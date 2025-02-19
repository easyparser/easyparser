from typing import Callable, Optional, Literal
import re


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


def by_characters(
    text: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    length_fn: Callable[[str], int] = len,
    separators: Optional[list[str]] = None,
    keep_separator: bool | Literal["start", "end"] = True,
    is_separator_regex: bool = False,
) -> list[str]:
    """Chunk the text recursively based on a list of characters.

    Args:
        text: text to split
        chunk_size: maximum size of chunks to return
        chunk_overlap: overlap between chunks
        length_fn: function to measure text length
        separators: list of separators to use for splitting, tries each one in order
        keep_separator: whether to keep separator in chunks and where to place it
        is_separator_regex: whether separators are regex patterns

    Returns:
        list of text chunks
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
            f"({chunk_size}), should be smaller."
        )

    separators = separators or ["\n\n", "\n", " ", ""]

    def _split_text(text: str, separators: list[str]) -> list[str]:
        """Recursive implementation of text splitting."""
        # Find the first separator that appears in text
        separator = separators[-1]
        _separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            _sep = sep if is_separator_regex else re.escape(sep)
            if sep == "":
                separator = sep
                break
            if re.search(_sep, text):
                separator = sep
                new_separators = separators[i + 1 :]
                break

        # Split text using the selected separator
        _separator = separator if is_separator_regex else re.escape(separator)
        splits = split_with_regex(text, _separator, keep_separator)

        final_chunks = []
        good_splits = []
        _separator = "" if not keep_separator else separator

        # Process each split
        for split in splits:
            if length_fn(split) < chunk_size:
                good_splits.append(split)
            else:
                if good_splits:
                    merged_text = merge_splits(good_splits, _separator, chunk_size, chunk_overlap, length_fn)
                    final_chunks.extend(merged_text)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(split)
                else:
                    other_chunks = _split_text(split, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged_text = merge_splits(good_splits, _separator, chunk_size, chunk_overlap, length_fn)
            final_chunks.extend(merged_text)

        return final_chunks

    return _split_text(text, separators)


__all__ = ["by_characters"]
