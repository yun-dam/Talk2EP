import collections
import dataclasses
import difflib
import enum
import functools
import itertools
from typing import Iterator, Sequence

from sliders.experimental import tokenizer
from sliders.log_utils import logger

_FUZZY_ALIGNMENT_MIN_THRESHOLD = 0.75


class AlignmentStatus(enum.Enum):
    MATCH_EXACT = "MATCH_EXACT"
    MATCH_LESSER = "MATCH_LESSER"
    MATCH_FUZZY = "MATCH_FUZZY"


@dataclasses.dataclass
class Extraction:
    extraction_text: str
    token_interval: tokenizer.TokenInterval | None = None
    char_interval: tokenizer.CharInterval | None = None
    alignment_status: AlignmentStatus | None = None


class WordAligner:
    """Aligns words between two sequences of tokens using Python's difflib."""

    def __init__(self):
        """Initialize the WordAligner with difflib SequenceMatcher."""
        self.matcher = difflib.SequenceMatcher(autojunk=False)
        self.source_tokens: Sequence[str] | None = None
        self.extraction_tokens: Sequence[str] | None = None

    def _set_seqs(
        self,
        source_tokens: Sequence[str],
        extraction_tokens: Sequence[str],
    ):
        """Sets the source and extraction tokens for alignment.

        Args:
          source_tokens: A nonempty sequence or iterator of word-level tokens from
            source text.
          extraction_tokens: A nonempty sequence or iterator of extraction tokens in
            order for matching to the source.
        """

        if not source_tokens or not extraction_tokens:
            raise ValueError("Source tokens and extraction tokens cannot be empty.")

        self.source_tokens = source_tokens
        self.extraction_tokens = extraction_tokens
        self.matcher.set_seqs(a=source_tokens, b=extraction_tokens)

    def _get_matching_blocks(self) -> Sequence[tuple[int, int, int]]:
        """Utilizes difflib SequenceMatcher and returns matching blocks of tokens.

        Returns:
          Sequence of matching blocks between source_tokens (S) and
          extraction_tokens
          (E). Each block (i, j, n) conforms to: S[i:i+n] == E[j:j+n], guaranteed to
          be monotonically increasing in j. Final entry is a dummy with value
          (len(S), len(E), 0).
        """
        if self.source_tokens is None or self.extraction_tokens is None:
            raise ValueError("Source tokens and extraction tokens must be set before getting matching blocks.")
        return self.matcher.get_matching_blocks()

    def _fuzzy_align_extraction(
        self,
        extraction: Extraction,
        source_tokens: list[str],
        tokenized_text: tokenizer.TokenizedText,
        token_offset: int,
        char_offset: int,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
    ) -> Extraction | None:
        """Fuzzy-align an extraction using difflib.SequenceMatcher on tokens.

        The algorithm scans every candidate window in `source_tokens` and selects
        the window with the highest SequenceMatcher `ratio`. It uses an efficient
        token-count intersection as a fast pre-check to discard windows that cannot
        meet the alignment threshold. A match is accepted when the ratio is ≥
        `fuzzy_alignment_threshold`. This only runs on unmatched extractions, which
        is usually a small subset of the total extractions.

        Args:
          extraction: The extraction to align.
          source_tokens: The tokens from the source text.
          tokenized_text: The tokenized source text.
          token_offset: The token offset of the current chunk.
          char_offset: The character offset of the current chunk.
          fuzzy_alignment_threshold: The minimum ratio for a fuzzy match.

        Returns:
          The aligned data.Extraction if successful, None otherwise.
        """

        extraction_tokens = list(_tokenize_with_lowercase(extraction.extraction_text))
        # Work with lightly stemmed tokens so pluralisation doesn't block alignment
        extraction_tokens_norm = [_normalize_token(t) for t in extraction_tokens]

        if not extraction_tokens:
            return None

        logger.info(f"Fuzzy aligning {extraction.extraction_text} ({len(extraction_tokens)} tokens)")

        best_ratio = 0.0
        best_span: tuple[int, int] | None = None  # (start_idx, window_size)

        len_e = len(extraction_tokens)
        max_window = len(source_tokens)

        extraction_counts = collections.Counter(extraction_tokens_norm)
        min_overlap = int(len_e * fuzzy_alignment_threshold)

        matcher = difflib.SequenceMatcher(autojunk=False, b=extraction_tokens_norm)

        for window_size in range(len_e, max_window + 1):
            if window_size > len(source_tokens):
                break

            # Initialize for sliding window
            window_deque = collections.deque(source_tokens[0:window_size])
            window_counts = collections.Counter([_normalize_token(t) for t in window_deque])

            for start_idx in range(len(source_tokens) - window_size + 1):
                # Optimization: check if enough overlapping tokens exist before expensive
                # sequence matching. This is an upper bound on the match count.
                if (extraction_counts & window_counts).total() >= min_overlap:
                    window_tokens_norm = [_normalize_token(t) for t in window_deque]
                    matcher.set_seq1(window_tokens_norm)
                    matches = sum(size for _, _, size in matcher.get_matching_blocks())
                    if len_e > 0:
                        ratio = matches / len_e
                    else:
                        ratio = 0.0
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_span = (start_idx, window_size)

                # Slide the window to the right
                if start_idx + window_size < len(source_tokens):
                    # Remove the leftmost token from the count
                    old_token = window_deque.popleft()
                    old_token_norm = _normalize_token(old_token)
                    window_counts[old_token_norm] -= 1
                    if window_counts[old_token_norm] == 0:
                        del window_counts[old_token_norm]

                    # Add the new rightmost token to the deque and count
                    new_token = source_tokens[start_idx + window_size]
                    window_deque.append(new_token)
                    new_token_norm = _normalize_token(new_token)
                    window_counts[new_token_norm] += 1

        if best_span and best_ratio >= fuzzy_alignment_threshold:
            start_idx, window_size = best_span

            try:
                extraction.token_interval = tokenizer.TokenInterval(
                    start_index=start_idx + token_offset,
                    end_index=start_idx + window_size + token_offset,
                )

                start_token = tokenized_text.tokens[start_idx]
                end_token = tokenized_text.tokens[start_idx + window_size - 1]
                extraction.char_interval = tokenizer.CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )

                extraction.alignment_status = AlignmentStatus.MATCH_FUZZY
                return extraction
            except IndexError:
                logger.exception("Index error while setting intervals during fuzzy alignment.")
                return None

        return None

    def align_extractions(
        self,
        extraction_groups: Sequence[Sequence[Extraction]],
        source_text: str,
        token_offset: int = 0,
        char_offset: int = 0,
        delim: str = "\u241f",  # Unicode Symbol for unit separator
        enable_fuzzy_alignment: bool = True,
        fuzzy_alignment_threshold: float = _FUZZY_ALIGNMENT_MIN_THRESHOLD,
        accept_match_lesser: bool = False,
    ) -> Sequence[Sequence[Extraction]]:
        """Aligns extractions with their positions in the source text.

        This method takes a sequence of extractions and the source text, aligning
        each extraction with its corresponding position in the source text. It
        returns a sequence of extractions along with token intervals indicating the
        start and
        end positions of each extraction in the source text. If an extraction cannot
        be
        aligned, its token interval is set to None.

        Args:
          extraction_groups: A sequence of sequences, where each inner sequence
            contains an Extraction object.
          source_text: The source text against which extractions are to be aligned.
          token_offset: The offset to add to the start and end indices of the token
            intervals.
          char_offset: The offset to add to the start and end positions of the
            character intervals.
          delim: Token used to separate multi-token extractions.
          enable_fuzzy_alignment: Whether to use fuzzy alignment when exact matching
            fails.
          fuzzy_alignment_threshold: Minimum token overlap ratio for fuzzy alignment
            (0-1).
          accept_match_lesser: Whether to accept partial exact matches (MATCH_LESSER
            status).

        Returns:
          A sequence of extractions aligned with the source text, including token
          intervals.
        """
        logger.info(
            "WordAligner: Starting alignment of extractions with the source text. Extraction groups to align: %s",
            extraction_groups,
        )
        if not extraction_groups:
            logger.info("No extraction groups provided; returning empty list.")
            return []

        source_tokens = list(_tokenize_with_lowercase(source_text))

        delim_len = len(list(_tokenize_with_lowercase(delim)))
        if delim_len != 1:
            raise ValueError(f"Delimiter {delim!r} must be a single token.")

        logger.info("Using delimiter %r for extraction alignment", delim)

        extraction_tokens = list(
            _tokenize_with_lowercase(
                f" {delim} ".join(extraction.extraction_text for extraction in itertools.chain(*extraction_groups))
            )
        )

        self._set_seqs(source_tokens, extraction_tokens)

        index_to_extraction_group = {}
        extraction_index = 0
        for group_index, group in enumerate(extraction_groups):
            logger.info(f"Processing extraction group {group_index} with {len(group)} extractions.")
            for extraction in group:
                # Validate delimiter doesn't appear in extraction text
                if delim in extraction.extraction_text:
                    raise ValueError(
                        f"Delimiter {delim!r} appears inside extraction text"
                        f" {extraction.extraction_text!r}. This would corrupt alignment"
                        " mapping."
                    )

                index_to_extraction_group[extraction_index] = (extraction, group_index)
                extraction_text_tokens = list(_tokenize_with_lowercase(extraction.extraction_text))
                extraction_index += len(extraction_text_tokens) + delim_len

        aligned_extraction_groups: list[list[Extraction]] = [[] for _ in extraction_groups]
        tokenized_text = tokenizer.tokenize(source_text)

        # Track which extractions were aligned in the exact matching phase
        aligned_extractions = []
        exact_matches = 0
        lesser_matches = 0

        # Exact matching phase
        for i, j, n in self._get_matching_blocks()[:-1]:
            extraction, _ = index_to_extraction_group.get(j, (None, None))
            if extraction is None:
                logger.info(f"No clean start index found for extraction index={j} iterating Difflib matching_blocks")
                continue

            extraction.token_interval = tokenizer.TokenInterval(
                start_index=i + token_offset,
                end_index=i + n + token_offset,
            )

            try:
                start_token = tokenized_text.tokens[i]
                end_token = tokenized_text.tokens[i + n - 1]
                extraction.char_interval = tokenizer.CharInterval(
                    start_pos=char_offset + start_token.char_interval.start_pos,
                    end_pos=char_offset + end_token.char_interval.end_pos,
                )
            except IndexError as e:
                raise IndexError(
                    "Failed to align extraction with source text. Extraction token"
                    f" interval {extraction.token_interval} does not match source text"
                    f" tokens {tokenized_text.tokens}."
                ) from e

            extraction_text_len = len(list(_tokenize_with_lowercase(extraction.extraction_text)))
            if extraction_text_len < n:
                raise ValueError(
                    "Delimiter prevents blocks greater than extraction length: "
                    f"extraction_text_len={extraction_text_len}, block_size={n}"
                )
            if extraction_text_len == n:
                extraction.alignment_status = AlignmentStatus.MATCH_EXACT
                exact_matches += 1
                aligned_extractions.append(extraction)
            else:
                # Partial match (extraction longer than matched text)
                if accept_match_lesser:
                    extraction.alignment_status = AlignmentStatus.MATCH_LESSER
                    lesser_matches += 1
                    aligned_extractions.append(extraction)
                else:
                    # Reset intervals when not accepting lesser matches
                    extraction.token_interval = None
                    extraction.char_interval = None
                    extraction.alignment_status = None

        # Collect unaligned extractions
        unaligned_extractions = []
        for extraction, _ in index_to_extraction_group.values():
            if extraction not in aligned_extractions:
                unaligned_extractions.append(extraction)

        # Apply fuzzy alignment to remaining extractions
        if enable_fuzzy_alignment and unaligned_extractions:
            logger.info(f"Starting fuzzy alignment for {len(unaligned_extractions)} unaligned extractions")
            for extraction in unaligned_extractions:
                aligned_extraction = self._fuzzy_align_extraction(
                    extraction,
                    source_tokens,
                    tokenized_text,
                    token_offset,
                    char_offset,
                    fuzzy_alignment_threshold,
                )
                if aligned_extraction:
                    aligned_extractions.append(aligned_extraction)
                    logger.info(f"Fuzzy alignment successful for extraction: {extraction.extraction_text}")

        for extraction, group_index in index_to_extraction_group.values():
            aligned_extraction_groups[group_index].append(extraction)

        logger.info(f"Final aligned extraction groups: {aligned_extraction_groups}")
        return aligned_extraction_groups


def _tokenize_with_lowercase(text: str) -> Iterator[str]:
    """Extract and lowercase tokens from the input text into words.

    This function utilizes the tokenizer module to tokenize text and yields
    lowercased words.

    Args:
      text (str): The text to be tokenized.

    Yields:
      Iterator[str]: An iterator over tokenized words.
    """
    tokenized_pb2 = tokenizer.tokenize(text)
    original_text = tokenized_pb2.text
    for token in tokenized_pb2.tokens:
        start = token.char_interval.start_pos
        end = token.char_interval.end_pos
        token_str = original_text[start:end]
        token_str = token_str.lower()
        yield token_str


@functools.lru_cache(maxsize=10000)
def _normalize_token(token: str) -> str:
    """Lowercases and applies light pluralisation stemming."""
    token = token.lower()
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def align_quotes_for_chunk(quotes: Sequence[str], source_text: str, accept_lesser: bool = False) -> list[Extraction]:
    """Convenience adapter to align a list of quote strings against a chunk's source text.

    Returns a list of Extraction objects with char/token intervals and status filled when aligned.
    """
    aligner = WordAligner()
    extraction_groups = [[Extraction(extraction_text=q) for q in quotes]]
    aligned_groups = aligner.align_extractions(
        extraction_groups=extraction_groups,
        source_text=source_text,
        accept_match_lesser=accept_lesser,
    )
    return aligned_groups[0] if aligned_groups else []


if __name__ == "__main__":
    import asyncio

    from sliders.document import Document
    from sliders.experimental.aligner import align_quotes_for_chunk

    async def main():
        document = await Document.from_markdown(
            "/data1/hypothesis_dataset/financebench/markdown/pdfs/3M_2023Q2_10Q.md",
            description="3M 2023 Q2 10Q",
            document_name="3M_2023Q2_10Q",
        )

        source_text = document.chunks[0]["content"]
        quotes = "Cash and cash equivalents | 4,258\nMarketable securities - current | 56\nAccounts receivable - net of allowances of  $160  and $174 | 4,947\nTotal current liabilities | 10,936".split(
            "\n"
        )

        print(quotes[0] in source_text)

        aligned = align_quotes_for_chunk(quotes, source_text, accept_lesser=True)

        print(aligned)

    asyncio.run(main())
