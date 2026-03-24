import argparse
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os
import tiktoken
from sliders.log_utils import logger
from tqdm.asyncio import tqdm
from langchain_core.rate_limiters import InMemoryRateLimiter
from openai import RateLimitError

rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
    max_bucket_size=20,
)


llm = (
    AzureChatOpenAI(
        model="gpt-4.1",
        api_key=os.getenv("AZURE_NAIRR_KEY"),
        azure_endpoint=os.getenv("AZARE_URL_ENDPOINT"),
        api_version="2024-12-01-preview",
        max_tokens=None,
        temperature=0.0,
        rate_limiter=rate_limiter,
        timeout=60,
    )
    .with_structured_output(method="json_mode")
    .with_retry(
        retry_if_exception_type=(RateLimitError,),  # Retry only on RateLimitError
        wait_exponential_jitter=True,  # Add jitter to the exponential backoff
        stop_after_attempt=2,  # Try twice
        exponential_jitter_params={"initial": 3},  # if desired, customize backoff
    )
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt_file", type=str)
    parser.add_argument("--txt_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--page_tokens",
        type=int,
        default=4000,
        help="Target tokens per page (default: 4000)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum page navigation steps to execute (default: 20)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to write extracted tables JSON (default: <txt_file>.tables.json)",
    )
    return parser.parse_args()


def _paginate_text(text: str, page_tokens: int) -> Tuple[List[str], List[int]]:
    """Split text into pages by tokens if possible, otherwise by approximate characters.

    Returns a tuple of (pages, page_start_char_offsets).
    """
    if page_tokens <= 0:
        return [text], [0]

    if tiktoken is not None:
        try:
            # cl100k_base is broadly compatible with GPT models
            encoding = tiktoken.get_encoding("cl100k_base")
            token_ids = encoding.encode(text)
            pages: List[str] = []
            offsets: List[int] = []
            start_offset = 0
            for i in range(0, len(token_ids), page_tokens):
                chunk_ids = token_ids[i : i + page_tokens]
                page_text = encoding.decode(chunk_ids)
                offsets.append(start_offset)
                pages.append(page_text)
                start_offset += len(page_text)
            return pages, offsets
        except Exception:
            # Fallback to char-based below
            pass

    # Character-based approximation (~4 chars/token)
    approx_chars_per_token = 4
    page_chars = max(1, page_tokens * approx_chars_per_token)
    pages = [text[i : i + page_chars] for i in range(0, len(text), page_chars)]
    offsets = [i for i in range(0, len(text), page_chars)]
    if not pages:
        pages = [""]
        offsets = [0]
    return pages, offsets


def _parse_llm_response(raw: Any) -> Dict[str, Any]:
    """Normalize various response shapes into a Python dict."""
    if raw is None:
        return {}
    # LangChain structured output may already be a dict
    if isinstance(raw, dict):
        return raw
    # Sometimes returns a BaseMessage with .content
    content = None
    try:
        content = getattr(raw, "content", None)
    except Exception:
        content = None
    if isinstance(content, (dict, list)):
        return content if isinstance(content, dict) else {"data": content}
    if isinstance(content, str):
        try:
            return json.loads(content)
        except Exception:
            pass
    # Last resort: string cast
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def number_lines(text: str) -> str:
    """Return a new string where each line starts with its 1-based line number.
    Preserves blank lines and the file's original newline characters."""
    parts = []
    for i, line in enumerate(text.splitlines(True), start=1):  # keepends=True
        content = line.rstrip("\r\n")
        linebreak = line[len(content) :]  # original newline(s): "", "\n", "\r\n", etc.
        parts.append(f"{i}" if content == "" else f"{i} {content}")
        parts.append(linebreak)
    return "".join(parts)


async def convert_txt_to_markdown(
    txt_file: str,
    page_tokens: int,
    output_json: Optional[str],
):
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert that identifies tables in a given text document.
Given a page of a text document, identify all tables.

The page has line index starting from 1.

For each table, return:
- the first line text of the table starting from the table header.
- the last line text of the table including the table footer.
- the first line index of the table starting from the table header.
- the last line index of the table including the table footer.

If uncertain, about the first or last line text, set that field to null.
Also provide some table notes based on your reading of the table. These should be metadata about the table, like the source of the table, the date of the table, the currency of the table, etc.

You should output JSON with the following fields:
{{
    "tables": [
        {{
            "first_line_text": string | null,
            "first_line_index": int | null,
            "last_line_text": string | null,
            "last_line_index": int | null,
            "table_notes": string | null
        }}
    ]
}}
""",
            ),
            ("human", "# Document\n\n {txt}"),
        ]
    )

    with open(txt_file, "r") as f:
        raw_full_text = f.read()

    full_text = number_lines(raw_full_text)

    # Paginate the document
    pages, page_offsets = _paginate_text(full_text, page_tokens)
    total_pages = len(pages)

    # Decide output path
    if not output_json:
        output_json = f"{txt_file}.new.tables.json"

    results: Dict[str, Any] = {
        "source": os.path.abspath(txt_file),
        "page_tokens": page_tokens,
        "total_pages": total_pages,
        "tables": [],
    }

    page_index = 0
    for page_index in range(total_pages):
        page_text = pages[page_index]
        # chain = template | llm.with_retry()
        prompt = template.invoke({"txt": page_text})
        raw = await llm.ainvoke(prompt)
        # logger.info(f"Page {page_index} response: {raw}")
        data = _parse_llm_response(raw)

        # logger.info(f"LLM thinks there are {len(data.get('tables', []))} tables on this page")

        # Extract tables if present
        tables_spec = data.get("tables", []) if isinstance(data, dict) else []
        if isinstance(tables_spec, list) and tables_spec:
            for table_spec in tables_spec:
                first_line_index = table_spec.get("first_line_index")
                last_line_index = table_spec.get("last_line_index")
                first_line_text = table_spec.get("first_line_text")
                last_line_text = table_spec.get("last_line_text")
                table_notes = table_spec.get("table_notes")
                if (
                    first_line_index is None
                    or last_line_index is None
                    or first_line_text is None
                    or last_line_text is None
                ):
                    continue
                if first_line_index < 1 or last_line_index < 1:
                    continue
                if len(results["tables"]) > 0 and results["tables"][-1]["last_line_index"] > first_line_index:
                    continue
                results["tables"].append(
                    {
                        "first_line_index": first_line_index - 1,
                        "last_line_index": last_line_index - 1,
                        "first_line_text": first_line_text,
                        "last_line_text": last_line_text,
                        "table_notes": table_notes,
                    }
                )

    # Persist results
    with open(output_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # print(f"Extracted tables saved to: {output_json}")


async def main(txt_file, txt_dir, output_dir, page_tokens):
    if txt_file is not None:
        output_json = os.path.join(output_dir, os.path.basename(txt_file) + ".new.tables.json")
        await convert_txt_to_markdown(
            txt_file,
            page_tokens,
            output_json,
        )
    elif txt_dir is not None:
        tasks = []
        for txt_file in os.listdir(txt_dir):
            if txt_file.endswith(".txt"):
                output_json = os.path.join(output_dir, os.path.basename(txt_file) + ".new.tables.json")
                if os.path.exists(output_json):
                    continue
                tasks.append(
                    convert_txt_to_markdown(
                        os.path.join(txt_dir, txt_file),
                        page_tokens,
                        output_json,
                    )
                )
        await tqdm.gather(*tasks)
    else:
        raise ValueError("Either txt_file or txt_dir must be provided")


if __name__ == "__main__":
    args = parse_args()

    try:
        asyncio.run(
            main(
                txt_file=args.txt_file,
                txt_dir=args.txt_dir,
                output_dir=args.output_dir,
                page_tokens=args.page_tokens,
            )
        )

    except Exception as e:
        logger.error(e)
        import traceback

        traceback.print_exc()
        import ipdb

        ipdb.post_mortem()
