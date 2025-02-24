import re
from typing import Callable, Literal


def split_with_regex(
    text: str, separator: str, keep_separator: bool | Literal["start", "end"]
) -> list[str]:
    if not separator:
        return [c for c in text if c]

    if keep_separator:
        _splits = re.split(f"({separator})", text)
        if keep_separator == "end":
            splits = [
                _splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)
            ]
            if len(_splits) % 2 == 0:
                splits.extend(_splits[-1:])
            if _splits and len(_splits) > 1:
                splits.append(_splits[-1])
        else:  # start
            splits = [_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)]
            if _splits:
                splits.insert(0, _splits[0])
    else:
        splits = re.split(separator, text)

    return [s for s in splits if s]


def merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    chunk_overlap: int,
    length_fn: Callable[[str], int],
) -> list[str]:
    """Merge smaller splits into larger chunks

    Args:
        splits: list of text splits
        separator: separator to use between splits
        chunk_size: maximum size of chunks to return
        chunk_overlap: overlap between chunks
        length_fn: function to measure text length

    Returns:
        list of text chunks
    """
    separator_len = length_fn(separator)
    chunks = []
    current: list[str] = []
    total = 0

    for split in splits:
        split_len = length_fn(split)
        if total + split_len + (separator_len if current else 0) > chunk_size:
            if current:
                # Join current chunks and add to final list
                chunk = separator.join(current).strip()
                if chunk:
                    chunks.append(chunk)

                # Remove chunks until we're under chunk_overlap
                while total > chunk_overlap or (
                    total + split_len + (separator_len if current else 0) > chunk_size
                    and total > 0
                ):
                    total -= length_fn(current[0]) + (
                        separator_len if len(current) > 1 else 0
                    )
                    current.pop(0)

            if not current and split_len > chunk_size:
                chunks.append(split)
                continue

        current.append(split)
        total += split_len + (separator_len if len(current) > 1 else 0)

    if current:
        chunk = separator.join(current).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def word_len(text: str) -> int:
    return len(text.split())
