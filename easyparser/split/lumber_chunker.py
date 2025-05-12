import logging
from bisect import bisect_left
from typing import Callable, List, Optional

from tqdm import tqdm

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType
from easyparser.mime import MimeType
from easyparser.models import completion, parse_json_from_text
from easyparser.split.split import _non_whitespace_separators, split_text
from easyparser.split.utils import word_len

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<task> You are given a set of texts between the starting tag <passages> and ending tag </passages>. Each text is labeled as 'ID `N`' where 'N' is the passage number. Your task is to find the first passage where the content clearly separates from the previous passages in topic and/or semantics. </task>

<rules>
Follow the following rules while finding the splitting passage:
- Always return the answer as a JSON parsable object with the 'split_index' key having a value of the first passage where the topic changes.
- Avoid very long groups of paragraphs. Aim for a good balance between identifying content shifts and keeping groups manageable.
- If no clear `split_index` is found, return N + 1, where N is the index of the last passage.
</rules>

<passages>
{passages}
</passages>
"""  # noqa: E501
SPLIT_SCHEMA = {
    "properties": {
        "split_index": {"title": "ID of split position", "type": "integer"},
    },
    "required": ["split_index"],
    "title": "Split",
    "type": "object",
}
DEFAULT_CHUNK_JOIN_CHAR = " "
MIN_SPLITS_TO_REPORT_PROGRESS = 10


def _add_id_to_splits(splits: List[str]) -> List[str]:
    """Prepare the splits for the chunker."""
    return [
        f"ID {i}: " + split.replace("\n", "").strip()
        for (i, split) in enumerate(splits)
    ]


def _get_cumulative_token_counts(splits: List[str], length_fn: Callable) -> List[int]:
    """Get the cumulative token counts for the splits."""
    token_counts = []
    current_token_count = 0
    for split in splits:
        token_counts.append(current_token_count)
        current_token_count += length_fn(split)
    return token_counts


def _is_mime_text(chunk: Chunk) -> bool:
    return (
        chunk.mimetype == MimeType.text and chunk.content and chunk.ctype != CType.Root
    )


class LumberChunker(BaseOperation):
    _len_fns = {
        "len": len,
        "word_len": word_len,
    }

    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        model: str | None = None,
        chunk_size: int = 256,
        candidate_size: int | None = None,
        length_fn: Callable[[str], int] | None | str = word_len,
        separators: Optional[list[str]] = None,
        chunk_join_char: str = DEFAULT_CHUNK_JOIN_CHAR,
        verbose: bool = True,
        **kwargs,
    ) -> ChunkGroup:
        """Chunk the text into smaller pieces based on the content of the text."""
        # Resolve length function
        if isinstance(length_fn, str):
            length_fn = cls._len_fns[length_fn]
        elif length_fn is None:
            length_fn = word_len

        separators = separators or (
            ["\n\n", "\n", "\t"] + _non_whitespace_separators + [" ", ""]
        )
        candidate_size = max(10, chunk_size // 8)

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root_id, root in enumerate(chunks):
            child_chunks = [item for _, item in root.walk() if _is_mime_text(item)]
            split_chunks = []

            for child_chunk in tqdm(
                child_chunks,
                desc=f"easyparser item #{root_id}",
                total=len(child_chunks),
                disable=not verbose,
            ):
                # Get candidate splits
                splits = split_text(
                    child_chunk.content,
                    length_fn=length_fn,
                    chunk_size=candidate_size,
                    separators=separators,
                    keep_separator=True,
                    is_separator_regex=False,
                )
                # Show progress bar if there are enough splits
                if len(splits) >= MIN_SPLITS_TO_REPORT_PROGRESS and verbose:
                    progress_bar = tqdm(
                        total=len(splits),
                        desc="Processing splits",
                        unit="split",
                    )
                else:
                    progress_bar = None

                if len(splits) <= 1:
                    # If the split is too small, just return the original chunk
                    split_chunks.append(child_chunk.clone(no_relation=True))
                    continue

                num_splits = len(splits)
                splits_with_id = _add_id_to_splits(splits)
                # Get the cumulative token counts
                cumulative_token_counts = _get_cumulative_token_counts(
                    splits_with_id, length_fn
                )
                # Repeatedly call completion prompt to get the split index
                current_index = 0
                current_token_count = 0
                while current_index < num_splits:
                    group_end_index = min(
                        bisect_left(
                            cumulative_token_counts, current_token_count + chunk_size
                        ),
                        num_splits,
                    )

                    if group_end_index <= current_index + 1:
                        split_index = current_index + 1
                    else:
                        prompt = PROMPT_TEMPLATE.format(
                            passages="\n".join(
                                splits_with_id[current_index:group_end_index]
                            )
                        )
                        response = completion(
                            prompt,
                            model=model,
                            schema=SPLIT_SCHEMA,
                        )
                        logger.debug(f"LumberChunker prompt: {prompt}")
                        logger.debug(f"LumberChunker response: {response}")
                        decoded_dict = parse_json_from_text(response)
                        if decoded_dict is None:
                            split_index = group_end_index
                        else:
                            split_index = int(decoded_dict["split_index"])

                        if current_index >= split_index:
                            split_index = current_index + 1

                    new_split_text = chunk_join_char.join(
                        [split for split in splits[current_index:split_index]]
                    )
                    split_chunks.append(
                        Chunk(
                            text=new_split_text,
                            content=new_split_text,
                            mimetype=child_chunk.mimetype,
                            ctype=child_chunk.ctype,
                            summary=child_chunk.summary,
                            origin=child_chunk.origin,
                            metadata=child_chunk.metadata,
                        )
                    )

                    current_token_count = cumulative_token_counts[
                        min(split_index, num_splits - 1)
                    ]
                    current_index = split_index
                    if progress_bar:
                        progress_bar.update(current_index - progress_bar.n)

            new_root = root.clone(no_relation=True)
            new_root.add_children(split_chunks)
            output.append(new_root)

        return output
