import asyncio
import functools
import json
import time
import traceback
from collections import defaultdict
from copy import deepcopy


from sliders.document import Document
from sliders.experimental.aligner import align_quotes_for_chunk
from sliders.llm_models import Tables, NewFields, create_dynamic_extraction_relation_model, IsRelevantPage
from sliders.llm_models import Field as LLMField
from sliders.log_utils import logger
from sliders.utils import prepare_schema_repr

from sliders.llm.llm import get_llm_client
from sliders.llm.prompts import load_fewshot_prompt_template
from sliders.callbacks.logging import LoggingHandler


# Define ExtractedData class locally since it's not imported
class ExtractedData:
    def __init__(self, value, rationale, quote, metadata):
        self.value = value
        self.rationale = rationale
        self.quote = quote
        self.metadata = metadata


class ExtractSchema:
    def __init__(self, config: dict, model_config: dict):
        self.config = config
        self.model_config = model_config

    @staticmethod
    def create_extract_schema_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)

        extract_schema_template = load_fewshot_prompt_template(
            template_file="sliders/extract_schema.prompt",
            template_blocks=[],
        )
        extract_schema_chain = extract_schema_template | llm_client.with_structured_output(method="json_mode")
        return extract_schema_chain

    @staticmethod
    def create_is_relevant_chunk_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        is_relevant_chunk_template = load_fewshot_prompt_template(
            template_file="sliders/is_relevant_chunk.prompt",
            template_blocks=[],
        )
        is_relevant_chunk_chain = is_relevant_chunk_template | llm_client.with_structured_output(IsRelevantPage)
        return is_relevant_chunk_chain

    @staticmethod
    def create_decomposition_chain(**kwargs):
        llm_client = get_llm_client(**kwargs)
        decomposition_template = load_fewshot_prompt_template(
            template_file="sliders/derive_atomic_fields.prompt",
            template_blocks=[],
        )
        decomposition_chain = decomposition_template | llm_client.with_structured_output(NewFields)
        return decomposition_chain

    def convert_extracted_data_to_json(self, extracted_data: list[dict], document: Document, doc_id: int) -> list[dict]:
        final_json_repr = []
        for chunk_id, res in enumerate(extracted_data):
            if res is None:
                continue
            # Support both Pydantic model results and plain dicts
            if isinstance(res, str):
                res = json.loads(res)
            chunk = res.dict() if hasattr(res, "dict") else res
            if chunk is None:
                continue
            chunk_dict = {}
            for key, values in chunk.items():
                if isinstance(values, str):
                    continue
                if key == "extraction_plan":
                    continue
                for value in values:
                    if not isinstance(value, dict):
                        logger.info(
                            f"Value is not a dict: {value} for document {document.document_name} chunk {chunk_id}"
                        )
                        continue
                    value["__metadata__"] = {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "document_name": document.document_name,
                        "chunk_header": document.chunks[chunk_id]["metadata"],
                    }
                if len(values) > 0:
                    chunk_dict[key] = values
            if len(chunk_dict) > 0:
                final_json_repr.append(chunk_dict)
        return final_json_repr

    def _dedupe_rows(self, rows: list[dict]) -> list[dict]:
        """Merge rows by value-only compatibility.

        Merge two rows when for every field, either values are equal (after normalization)
        or at least one side is None. If any field has conflicting non-None values, do not merge.
        When merging, take the non-None value; union quotes; set is_explicit True if any True;
        prefer non-empty reasoning (keep the first non-empty or longer one).
        """

        def normalize_value(value):
            if isinstance(value, str):
                return value.strip()
            if isinstance(value, list):
                return tuple(value)
            if isinstance(value, dict):
                return tuple(sorted(value.items()))
            return value

        def merge_quotes(q1, q2):
            def to_list(q):
                if q is None:
                    return []
                if isinstance(q, list):
                    return q
                return [q]

            merged_list = []
            seen_quotes = set()
            for q in to_list(q1) + to_list(q2):
                if q is None:
                    continue
                if q not in seen_quotes:
                    seen_quotes.add(q)
                    merged_list.append(q)
            return merged_list if merged_list else None

        def merge_field_payload(p1: dict | None, p2: dict | None) -> tuple[dict | None, bool]:
            # Returns (merged_payload_or_none, conflict_flag)
            p1 = p1 or {}
            p2 = p2 or {}
            v1 = p1.get("value")
            v2 = p2.get("value")
            n1 = normalize_value(v1)
            n2 = normalize_value(v2)

            # Conflict if both non-None and different
            if v1 is not None and v2 is not None and n1 != n2:
                return None, True

            merged_value = v1 if v1 is not None else v2
            merged_quote = merge_quotes(p1.get("quote"), p2.get("quote"))
            r1 = p1.get("reasoning") or ""
            r2 = p2.get("reasoning") or ""
            merged_reasoning = r1 if len(r1) >= len(r2) else r2
            merged_is_explicit = bool(p1.get("is_explicit", True) or p2.get("is_explicit", True))

            merged = {
                "value": merged_value,
                "quote": merged_quote,
                "reasoning": merged_reasoning,
                "is_explicit": merged_is_explicit,
            }
            return merged, False

        def try_merge_rows(base: dict, incoming: dict) -> tuple[dict | None, bool]:
            # Returns (merged_row_or_none, did_merge)
            merged_row: dict = {}
            keys = set([k for k in base.keys() if k != "__metadata__"]) | set(
                [k for k in incoming.keys() if k != "__metadata__"]
            )
            for key in keys:
                if key == "__metadata__":
                    continue
                merged_field, conflict = merge_field_payload(base.get(key), incoming.get(key))
                if conflict:
                    return None, False
                merged_row[key] = merged_field or {}

            # Merge metadata if present by keeping none (will be added later) or prefer base
            if "__metadata__" in base or "__metadata__" in incoming:
                merged_row["__metadata__"] = base.get("__metadata__") or incoming.get("__metadata__")

            return merged_row, True

        # Greedy merging pass until stable
        working = list(rows)
        changed = True
        while changed:
            changed = False
            merged_list: list[dict] = []
            for row in working:
                merged_in = False
                for i, existing in enumerate(merged_list):
                    merged_row, did_merge = try_merge_rows(existing, row)
                    if did_merge and merged_row is not None:
                        merged_list[i] = merged_row
                        merged_in = True
                        changed = True
                        break
                if not merged_in:
                    merged_list.append(row)
            working = merged_list
        return working

    def _merge_chunk_results(self, results: list[dict]) -> dict:
        """Merge multiple extraction results for a single chunk, then value-only dedupe per relationship."""
        merged: dict = {}

        def return_non_none_rows(values: list[dict]) -> list[dict]:
            non_none_rows = []
            for value in values:
                for key, row_value in value.items():
                    if row_value.get("value") is not None:
                        non_none_rows.append(value)
                        break
            return non_none_rows

        for res in results:
            chunk = res.dict() if hasattr(res, "dict") else res
            if chunk is None:
                continue
            for key, values in chunk.items():
                if key == "extraction_plan":
                    continue
                if values:
                    # if any of the value is not None, then extend the values
                    non_none_rows = return_non_none_rows(values)
                    if non_none_rows:
                        merged.setdefault(key, []).extend(non_none_rows)
        # Value-only dedupe, enabled by default via config
        if self.config.get("dedupe_merged_rows", True):
            for key, rows in merged.items():
                merged[key] = self._dedupe_rows(rows)
        return merged

    async def _extract_chunk_single(
        self,
        question_id: str,
        extract_chain,
        schema_repr: str,
        document: Document,
        chunk: dict,
        chunk_id: int,
        question: str,
        task_guidelines: str,
    ):
        if self.config.get("is_relevant_chunk", False):
            is_relevant_handler = LoggingHandler(
                prompt_file="sliders/is_relevant_chunk.prompt",
                metadata={
                    "question_id": question_id,
                    "question": question,
                    "stage": "is_relevant_chunk",
                    "chunk_id": chunk_id,
                    "document_name": document.document_name,
                    "objective": str(chunk_id) + "_" + document.document_name,
                },
            )
            is_relevant_chain = self.create_is_relevant_chunk_chain(
                **self.model_config["is_relevant_chunk"],
            )
            is_relevant = await is_relevant_chain.ainvoke(
                {
                    "document": self._document_chunk_repr(document, chunk, chunk_id),
                    "question": question,
                },
                config={"callbacks": [is_relevant_handler]},
            )
            if not is_relevant.is_relevant:
                return "irrelevant"

        handler = LoggingHandler(
            prompt_file="sliders/extract_schema.prompt",
            metadata={
                "question_id": question_id,
                "question": question,
                "stage": "extract_chunk_single",
                "chunk_id": chunk_id,
                "document_name": document.document_name,
                "objective": str(chunk_id) + "_" + document.document_name,
            },
        )
        payload = {
            "schema": schema_repr,
            "document": self._document_chunk_repr(document, chunk, chunk_id),
            "document_name": document.document_name,
            "document_description": document.description,
            "question": question,
            "task_guidelines": task_guidelines,
        }
        return await extract_chain.ainvoke(payload, config={"callbacks": [handler]})

    async def _extract_chunk_multiple(
        self,
        question_id: str,
        extract_chain,
        schema_repr: str,
        document: Document,
        chunk: dict,
        chunk_id: int,
        question: str,
        task_guidelines: str,
        num_samples: int,
    ):
        """Extract a single chunk num_samples times and merge results. If num_samples==1, do a single call."""
        calls = [
            self._extract_chunk_single(
                question_id, extract_chain, schema_repr, document, chunk, chunk_id, question, task_guidelines
            )
            for _ in range(num_samples)
        ]
        results = await asyncio.gather(*calls, return_exceptions=True)
        filtered = [json.loads(r) for r in results if r is not None and not isinstance(r, Exception)]
        if not filtered:
            return None
        return self._merge_chunk_results(filtered)

    async def _annotate_alignments(self, json_repr: list[dict], document: Document) -> None:
        # json_repr is a list of per-chunk dicts: {relationship_name: [relationship dicts]}
        for chunk_dict in json_repr:
            # Recover chunk_id from any relationship's metadata
            any_rel_list = next((rels for rels in chunk_dict.values() if rels), None)
            if not any_rel_list:
                continue
            any_rel = any_rel_list[0]
            meta = any_rel.get("__metadata__", {})
            chunk_id = meta.get("chunk_id")
            if chunk_id is None:
                continue
            source_text = document.chunks[chunk_id]["content"]
            loop = asyncio.get_running_loop()

            for _, relationships in chunk_dict.items():
                for row in relationships:
                    for field_name, field_value in row.items():
                        quotes = field_value.get("quote")
                        if not quotes:
                            continue
                        align_call = functools.partial(align_quotes_for_chunk, quotes, source_text, True)
                        try:
                            aligned = await asyncio.wait_for(
                                loop.run_in_executor(None, align_call), timeout=0.5 * len(quotes)
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"Alignment timed out for document={document.document_name} chunk_id={chunk_id}"
                            )
                            aligned = []

                        if "alignment_metadata" not in row["__metadata__"]:
                            row["__metadata__"]["alignment_metadata"] = {}
                        row["__metadata__"]["alignment_metadata"][field_name] = [
                            {
                                "idx": idx,
                                "char_interval": {
                                    "start_pos": aligned_extraction.char_interval.start_pos
                                    if aligned_extraction.char_interval is not None
                                    else None,
                                    "end_pos": aligned_extraction.char_interval.end_pos
                                    if aligned_extraction.char_interval is not None
                                    else None,
                                },
                                "text_span": (
                                    source_text[
                                        aligned_extraction.char_interval.start_pos : aligned_extraction.char_interval.end_pos
                                    ]
                                    if aligned_extraction.char_interval is not None
                                    else None
                                ),
                                "alignment_status": aligned_extraction.alignment_status.value
                                if aligned_extraction.alignment_status is not None
                                else None,
                            }
                            for idx, aligned_extraction in enumerate(aligned)
                            if aligned_extraction is not None
                        ]

    async def handle_failed_extractions(
        self,
        question_id: str,
        question: str,
        extracted_data: list[dict],
        metadata: dict,
        schema: Tables,
        document: Document,
        schema_repr: str,
        successful_extractions: int,
        failed_extractions: int,
        retry_attempts: int,
    ):
        # For some chunks, the extraction might fail, so we need to retry with a higher temperature
        for i, res in enumerate(extracted_data):
            if res is None:
                failed_extractions += 1
                retry_attempts += 1
                try:
                    handler = LoggingHandler(
                        prompt_file="sliders/extract_schema.prompt",
                        metadata={
                            "question_id": question_id,
                            "question": question,
                            "stage": "extraction_retry",
                            "chunk_index": i,
                            "document": document.document_name,
                        },
                    )

                    # retry with a higher temperature (also won't hit cache)
                    payload = deepcopy(self.model_config["extract_schema"])
                    payload.pop("temperature")
                    extraction_chain = self.create_extract_schema_chain(
                        temperature=0.7,
                        **payload,
                    )
                    new_res = await extraction_chain.ainvoke(
                        {
                            "document": self._document_chunk_repr(document, document.chunks[i], i),
                            "document_name": document.document_name,
                            "document_description": document.description,
                            "schema": schema_repr,
                            "question": question,
                        },
                        config={"callbacks": [handler]},
                    )
                    extracted_data[i] = new_res
                    if new_res is not None:
                        successful_extractions += 1
                    else:
                        failed_extractions += 1
                except Exception as e:
                    metadata["errors"].append(
                        {
                            "stage": "extraction_retry",
                            "error": str(e),
                            "chunk_index": i,
                            "document": document.document_name,
                        }
                    )
                    failed_extractions += 1
            else:
                successful_extractions += 1

    def finalize_tables(self, per_document_extracted_data: list[dict]) -> list[dict]:
        tables = {}
        for document_data in per_document_extracted_data:
            if document_data is not None:
                for chunk_data in document_data:
                    if chunk_data is not None:
                        for table in chunk_data["tables"]:
                            if table["name"] not in tables:
                                tables[table["name"]] = []
                            for row in table["rows"]:
                                row["__metadata__"] = table["__metadata__"]
                                tables[table["name"]].append(row)
        return tables

    async def checkback_extracted_data(
        self,
        question_id: str,
        extracted_data: list[dict],
        document: Document,
        doc_id: int,
        schema: Tables,
        question: str,
    ):
        """This function checks if the quote in the extracted data is actually present in the document."""
        for chunk_id, extracted_chunk_data in enumerate(extracted_data):
            for table in extracted_chunk_data["tables"]:
                rows = []
                for row in table["rows"]:
                    if "fields" not in row:
                        continue
                    for field_name, field_value in row["fields"].items():
                        if "quote" in field_value and field_value["quote"] is None:
                            field_value["value"] = None
                    rows.append(row)
                table["rows"] = rows

        # Step 1: remove rows that contain information that is not useful, for example if field_1 is critical and it is
        # not present in the row, then remove the row. We might want to ask the llm given the schema, which fields are
        # critical and which are not.

        # extracted_data = await self.rules_engine.apply(extracted_data, schema, document)

        # Step 2: Once we have done that, we want to get atomic fields from the rows, i.e., split the field where
        # `is_explicit` is False into multiple fields such that `is_explicit = True` for those new fields.

        if self.config.get("decompose_fields", False):
            await self._decompose_fields(question_id, extracted_data, schema, document, doc_id, question)

    async def _decompose_fields(
        self,
        question_id: str,
        extracted_data: list[dict],
        schema: Tables,
        document: Document,
        doc_id: int,
        question: str,
    ):
        chunks_to_reextract = defaultdict(list)
        map_chunk_id_to_extracted_data_index = {}
        for i, extracted_chunk_data in enumerate(extracted_data):
            for relationship_name, rows in extracted_chunk_data.items():
                row_with_non_explicit_fields = []
                for row in rows:
                    for field_name, field_value in row.items():
                        if field_value.get("is_explicit", True) is False and field_value.get("value") is not None:
                            row_with_non_explicit_fields.append(row)
                            break
                if len(row_with_non_explicit_fields) > 0:
                    # since all the rows will be from the same chunk, we can get the chunk_id from the first row
                    chunk_id = row_with_non_explicit_fields[0]["__metadata__"].get("chunk_id")
                    if chunk_id is None:
                        continue
                    map_chunk_id_to_extracted_data_index[chunk_id] = i
                    chunks_to_reextract[chunk_id].append({relationship_name: row_with_non_explicit_fields})

        # Step 3.1: given the new table, update the schema
        # Step 3.2: re-extract data from the given chunk
        # Step 3.3: update the extracted data with new schema and re-extracted data

        if not chunks_to_reextract:
            return

        # Helper to propose atomic fields for a relationship based on problematic rows
        async def _propose_atomic_fields(
            relationship_name: str, rows: list[dict], current_schema: Tables
        ) -> list[LLMField]:
            # Create a compact preview omitting quotes to keep prompt size reasonable
            preview_rows = []
            for r in rows:
                preview_rows.append(
                    {
                        k: {
                            "value": v.get("value"),
                            "is_explicit": v.get("is_explicit"),
                            "quote": v.get("quote"),
                            "reasoning": v.get("reasoning"),
                        }
                        for k, v in r.items()
                        if k != "__metadata__"
                    }
                )

            derive_atomic_fields_chain = self.create_decomposition_chain(
                **self.model_config["derive_atomic_fields"],
            )
            schema_repr_local = prepare_schema_repr(current_schema)
            try:
                handler = LoggingHandler(
                    prompt_file="sliders/derive_atomic_fields.prompt",
                    metadata={
                        "question_id": question_id,
                        "question": question,
                        "stage": "derive_atomic_fields",
                        "relationship_name": relationship_name,
                    },
                )
                resp: NewFields = await derive_atomic_fields_chain.ainvoke(
                    {
                        "schema": schema_repr_local,
                        "relationship_name": relationship_name,
                        "rows": preview_rows,
                        "question": question,
                    },
                    config={"callbacks": [handler]},
                )
            except Exception:
                return []

            new_fields: list[LLMField] = []
            for f in resp.fields:
                new_fields.append(
                    LLMField(
                        name=f.name,
                        description=f.description,
                        extraction_guideline=f.extraction_guideline,
                        data_type=f.data_type,
                        unit=(f.unit or ""),
                        scale=(f.scale or ""),
                    )
                )
            return new_fields

        # Work on a copy of the schema to accumulate updates
        updated_schema = deepcopy(schema)

        # Accumulate proposed new atomic fields per relationship to avoid duplicate LLM calls
        relationship_to_new_fields: dict[str, list[LLMField]] = {}
        # Track decomposed/original fields per relationship
        relationship_to_decomposed_fields: dict[str, set[str]] = {}

        # Determine decomposed fields based on rows that triggered re-extraction
        for _, rels in chunks_to_reextract.items():
            for rel_map in rels:
                for relationship_name, rows in rel_map.items():
                    decomposed_names: set[str] = relationship_to_decomposed_fields.setdefault(relationship_name, set())
                    for row in rows:
                        for field_name, field_value in row.items():
                            if field_name == "__metadata__":
                                continue
                            if field_value.get("is_explicit", True) is False and field_value.get("value") is not None:
                                decomposed_names.add(field_name)

        # Propose atomic fields for each relationship once
        for relationship_name in relationship_to_decomposed_fields.keys():
            # Find any example rows for this relationship from chunks_to_reextract
            sample_rows: list[dict] = []
            for _, rels in chunks_to_reextract.items():
                for rel_map in rels:
                    if relationship_name in rel_map:
                        sample_rows = rel_map[relationship_name]
            proposed_fields = await _propose_atomic_fields(relationship_name, sample_rows, updated_schema)
            if not proposed_fields:
                continue
            # Update class fields in updated_schema (Step 3.1, add new fields)
            for cls in updated_schema.tables:
                if cls.name == relationship_name:
                    existing_names = {f.name for f in cls.fields}
                    for nf in proposed_fields:
                        if nf.name not in existing_names:
                            cls.fields.append(nf)
                    relationship_to_new_fields[relationship_name] = proposed_fields
                    break

        # Also remove decomposed/original fields from updated_schema (per request #2)
        if relationship_to_decomposed_fields:
            for cls in updated_schema.tables:
                if cls.name in relationship_to_decomposed_fields:
                    decomposed_set = relationship_to_decomposed_fields[cls.name]
                    if decomposed_set:
                        cls.fields = [f for f in cls.fields if f.name not in decomposed_set]

        # Update all extracted_data rows to include new fields (value=None) and remove decomposed fields
        if relationship_to_new_fields or relationship_to_decomposed_fields:
            for chunk_dict in extracted_data:
                for relationship_name, rows in list(chunk_dict.items()):
                    # Remove decomposed fields from every row of this relationship
                    decomposed_set = relationship_to_decomposed_fields.get(relationship_name, set())
                    if decomposed_set:
                        for row in rows:
                            for fname in list(row.keys()):
                                if fname in decomposed_set:
                                    row.pop(fname, None)
                    # Add missing new atomic fields to every row (with value=None)
                    new_fields = relationship_to_new_fields.get(relationship_name, [])
                    if new_fields:
                        for row in rows:
                            for nf in new_fields:
                                if nf.name not in row:
                                    row[nf.name] = {
                                        "value": None,
                                        "quote": None,
                                        "reasoning": "",
                                        "is_explicit": True,
                                    }

        # If no fields were proposed, and no decomposed fields, nothing to update/re-extract
        if not relationship_to_new_fields and not relationship_to_decomposed_fields:
            return

        # Re-extract the affected chunks with the updated schema (Step 3.2)
        updated_schema_repr = prepare_schema_repr(updated_schema)
        updated_model = create_dynamic_extraction_relation_model(updated_schema.tables)
        extract_chain_updated = self.create_extract_schema_chain(
            **self.model_config["extract_schema"],
            pydantic_class=updated_model,
        )

        for chunk_id in chunks_to_reextract.keys():
            try:
                handler = LoggingHandler(
                    prompt_file="sliders/extract_schema.prompt",
                    metadata={
                        "question_id": question_id,
                        "question": question,
                        "stage": "extract_chunk_updated",
                        "chunk_id": chunk_id,
                        "document_name": document.document_name,
                    },
                )
                new_res = await extract_chain_updated.ainvoke(
                    {
                        "schema": updated_schema_repr,
                        "document": self._document_chunk_repr(document, document.chunks[chunk_id], chunk_id),
                        "document_name": document.document_name,
                        "document_description": document.description,
                        "question": question,
                        "task_guidelines": "Prefer explicit values with direct quotes.",
                    },
                    config={"callbacks": [handler]},
                )
                # Convert and merge back (Step 3.3)
                rejson_list = self.convert_extracted_data_to_json([new_res], document, doc_id)
                if not rejson_list:
                    continue
                rejson = rejson_list[0]

                # Replace entire relationship entries for those targeted in this chunk
                for rel_map in chunks_to_reextract[chunk_id]:
                    for relationship_name in rel_map.keys():
                        if relationship_name in rejson:
                            extracted_data[map_chunk_id_to_extracted_data_index[chunk_id]][relationship_name] = rejson[
                                relationship_name
                            ]
                            # Ensure decomposed fields are removed and new fields exist in replaced rows
                            decomposed_set = relationship_to_decomposed_fields.get(relationship_name, set())
                            new_fields = relationship_to_new_fields.get(relationship_name, [])
                            for row in extracted_data[map_chunk_id_to_extracted_data_index[chunk_id]][
                                relationship_name
                            ]:
                                # Remove decomposed
                                if decomposed_set:
                                    for fname in list(row.keys()):
                                        if fname in decomposed_set:
                                            row[fname] = {
                                                "value": None,
                                                "quote": None,
                                                "reasoning": "Could be derived from the other fields.",
                                                "is_explicit": True,
                                            }
                                # Add new fields if missing (with value=None)
                                for nf in new_fields or []:
                                    if nf.name not in row:
                                        row[nf.name] = {
                                            "value": None,
                                            "quote": None,
                                            "reasoning": "",
                                            "is_explicit": True,
                                        }
            except Exception:
                # Keep original data for this chunk if re-extraction fails
                continue

    def _document_chunk_repr(self, document: Document, chunk: dict, page_number: int) -> str:
        return f"""Document: {document.document_name}
Headers: {chunk["metadata"]}
Page Number: {page_number}
Content:
{chunk["content"]}"""

    async def extract(
        self,
        question: str,
        schema: Tables,
        documents: list[Document],
        metadata: dict,
        task_guidelines: str,
    ) -> dict:
        extraction_start_time = time.time()

        extract_schema_chain = self.create_extract_schema_chain(
            **self.model_config["extract_schema"],
        )

        # await self.rules_engine.get_rules(schema, question)

        # Extract schema from each chunk of the document
        schema_repr = prepare_schema_repr(schema)
        # logger.info(schema_repr)
        per_document_extracted_data = []
        total_chunks = 0
        successful_extractions = 0
        failed_extractions = 0
        retry_attempts = 0

        for doc_id, document in enumerate(documents):
            document_tasks = []
            for chunk_id, chunk in enumerate(document.chunks):
                total_chunks += 1
                if self.config.get("num_samples_per_chunk", 1) == 1:
                    document_tasks.append(
                        self._extract_chunk_single(
                            question_id=metadata.get("question_id", None),
                            extract_chain=extract_schema_chain,
                            schema_repr=schema_repr,
                            document=document,
                            chunk=chunk,
                            chunk_id=chunk_id,
                            question=question,
                            task_guidelines=task_guidelines,
                        )
                    )
                else:
                    document_tasks.append(
                        self._extract_chunk_multiple(
                            question_id=metadata.get("question_id", None),
                            extract_chain=extract_schema_chain,
                            schema_repr=schema_repr,
                            document=document,
                            chunk=chunk,
                            chunk_id=chunk_id,
                            question=question,
                            task_guidelines=task_guidelines,
                            num_samples=self.config.get("num_samples_per_chunk", 1),
                        )
                    )

            try:
                extracted_data = await asyncio.gather(*document_tasks)

                extracted_data = [r for r in extracted_data if r != "irrelevant"]

                # Sometimes the extraction fails, so we need to retry with a higher temperature
                await self.handle_failed_extractions(
                    question_id=metadata.get("question_id", None),
                    question=question,
                    extracted_data=extracted_data,
                    metadata=metadata,
                    schema=schema,
                    document=document,
                    schema_repr=schema_repr,
                    successful_extractions=successful_extractions,
                    failed_extractions=failed_extractions,
                    retry_attempts=retry_attempts,
                )

                # Convert the extracted data to a JSON representation
                json_repr = self.convert_extracted_data_to_json(extracted_data, document, doc_id)
                # Annotate alignment metadata per relationship (with timeout)
                # await self._annotate_alignments(json_repr, document)

                await self.checkback_extracted_data(
                    question_id=metadata.get("question_id", None),
                    extracted_data=json_repr,
                    document=document,
                    doc_id=doc_id,
                    schema=schema,
                    question=question,
                )

                per_document_extracted_data.append(json_repr)

            except Exception as e:
                logger.error(f"Error extracting schema for document {document.document_name}: {e}")
                logger.error(traceback.format_exc())
                metadata["errors"].append(
                    {"stage": "extraction_batch", "error": str(e), "document": document.document_name}
                )
                failed_extractions += len(document_tasks)
                per_document_extracted_data.append([None] * len(document_tasks))

        # Update extraction metadata
        metadata["extraction"]["chunks_processed"] = total_chunks
        metadata["extraction"]["successful_extractions"] = successful_extractions
        metadata["extraction"]["failed_extractions"] = failed_extractions
        metadata["extraction"]["retry_attempts"] = retry_attempts
        metadata["extraction"]["extraction_time"] = time.time() - extraction_start_time
        metadata["extraction"]["success_rate"] = successful_extractions / total_chunks if total_chunks > 0 else 0

        # Finalizing the tables
        final_extracted_data = self.finalize_tables(per_document_extracted_data)

        return final_extracted_data
