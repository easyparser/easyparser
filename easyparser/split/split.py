import re
import textwrap
from typing import Callable, Optional

from ..base import BaseOperation, Chunk, ChunkGroup
from .utils import merge_splits, split_with_regex, word_len

_whitespace_pattern = re.compile(r"\s+")
_non_whitespace_separators = [  # from https://github.com/isaacus-dev/semchunk
    ".",
    "?",
    "!",
    "*",  # Sentence terminators.
    ";",
    ",",
    "(",
    ")",
    "[",
    "]",
    "“",
    "”",
    "‘",
    "’",
    "'",
    '"',
    "`",  # Clause separators.
    ":",
    "—",
    "…",  # Sentence interrupters.
    "/",
    "\\",
    "–",
    "&",
    "-",  # Word joiners.
]


def _default_separators(text):
    """Get the default separators based on the text"""
    whitespaces = _whitespace_pattern.findall(text)
    if not whitespaces:
        return _non_whitespace_separators + [""]

    return (
        list(sorted(set(whitespaces), key=lambda x: len(x), reverse=True))
        + _non_whitespace_separators
        + [""]
    )


def split_text(
    text: str,
    length,
    chunk_size,
    separators: list[str],
    keep_separator,
    is_separator_regex,
) -> list[str]:
    """Recursive implementation of text splitting."""
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

    # split text using the selected separator
    _separator = separator if is_separator_regex else re.escape(separator)
    splits = split_with_regex(text, _separator, keep_separator=keep_separator)

    final_chunks = []
    good_splits = []
    _separator = "" if keep_separator else separator

    # process each split
    for split in splits:
        if length(split) < chunk_size:
            good_splits.append(split)
        else:
            if good_splits:
                merged_text = merge_splits(
                    good_splits, _separator, chunk_size, 0, length
                )
                final_chunks.extend(merged_text)
                good_splits = []

            if not new_separators:
                final_chunks.append(split)
            else:
                other_chunks = split_text(
                    split,
                    length,
                    chunk_size,
                    new_separators,
                    keep_separator,
                    is_separator_regex,
                )
                final_chunks.extend(other_chunks)

    if good_splits:
        merged_text = merge_splits(good_splits, _separator, chunk_size, 0, length)
        final_chunks.extend(merged_text)

    return final_chunks


def flatten_chunk_to_markdown(
    chunk: Chunk,
    max_size: int,
    length_fn: Callable,
    separators: list,
    _cache: dict,
    **kwargs,
) -> list[Chunk]:
    """Flatten a parent chunk into multiple child chunks

    Args:
        chunk: Chunk to flatten
        max_size: maximum size of chunks to return.
        length_fn: function to measure text length
        _cache: cache for already processed chunks
    """
    if not kwargs:
        parent = chunk.parent
        kwargs = {"header_level": 0}
        while parent:
            if parent.ctype == "header":
                kwargs["header_level"] += 1
            parent = parent.parent

    if chunk.ctype == "header":
        kwargs["header_level"] += 1

    if chunk.id in _cache:
        rendered_text = _cache[chunk.id]
        if length_fn(rendered_text) < max_size:
            # Best case scenario, no need to split.
            return [
                Chunk(
                    mimetype="plain/text",
                    # @TODO: general-purpose ctype that can be used in this strategy
                    # and other strategies
                    ctype="flattened_markdown",
                    content=rendered_text,
                    metadata={"originals": chunk.get_ids()},
                )
            ]
    elif chunk.text:
        if length_fn(chunk.text) < max_size:
            # Best case scenario, no need to split.
            return [
                Chunk(
                    mimetype="plain/text",
                    ctype="flattened_markdown",
                    content=chunk.text,
                    metadata={"originals": chunk.get_ids()},
                )
            ]

    result = []
    current_chunk = Chunk(
        mimetype="plain/text",
        ctype="flattened_markdown",
        content="",
        metadata={"originals": [chunk.id]},
    )
    rendered_text = chunk.content if isinstance(chunk.content, str) else ""
    if (
        chunk.ctype == "header"
        and kwargs.get("header_level", 0) > 0
        and not rendered_text.startswith("#")
    ):
        rendered_text = "#" * kwargs["header_level"] + " " + rendered_text

    if length_fn(rendered_text) > max_size:
        # Split the text recursively based on the separators
        seps = separators or _default_separators(rendered_text)
        splitteds = split_text(
            rendered_text,
            length_fn,
            max_size,
            seps,
            keep_separator=False,
            is_separator_regex=False,
        )
        result = [
            Chunk(
                mimetype="plain/text",
                ctype="flattened_markdown",
                content=split,
                metadata={"originals": [chunk.id]},
            )
            for split in splitteds[:-1]
        ]
        current_chunk.content = splitteds[-1]
    else:
        current_chunk.content = rendered_text

    child = chunk.child
    while child:
        if child.child:
            # If child has children, process them recursively
            rendered_children = flatten_chunk_to_markdown(
                child, max_size, length_fn, separators, _cache, **kwargs
            )

            # Each of the rendered children is a chunk, that is properly splitted
            # according to the layout
            if rendered_children:
                separator = "\n\n"
                if (
                    length_fn(current_chunk.content)
                    + length_fn(separator)
                    + length_fn(rendered_children[0].content)
                    <= max_size
                ):
                    # Can add the child to the current chunk
                    if len(rendered_children) > 1:
                        current_chunk.content += (
                            separator + rendered_children[0].content
                        )
                        current_chunk.content = current_chunk.content.strip("\n")
                        current_chunk.metadata["originals"].extend(
                            rendered_children[0].metadata["originals"]
                        )
                        result.append(current_chunk)
                        result.extend(rendered_children[1:])
                        current_chunk = Chunk(
                            mimetype="plain/text",
                            ctype="flattened_markdown",
                            content="",
                            metadata={"originals": [chunk.id]},
                        )
                    else:
                        current_chunk.content += (
                            separator + rendered_children[0].content
                        )
                        current_chunk.metadata["originals"].extend(
                            rendered_children[0].metadata["originals"]
                        )

                    child = child.next
                    continue

                if current_chunk.content:
                    # Push the current chunk to the result
                    current_chunk.content = current_chunk.content.strip("\n")
                    result.append(current_chunk)
                    current_chunk = Chunk(
                        mimetype="plain/text",
                        ctype="flattened_markdown",
                        content="",
                        metadata={"originals": [chunk.id]},
                    )
                result.extend(rendered_children)
            child = child.next
            continue

        # If the child doesn't have children, process it individually
        child_str = child.render(format="markdown", **kwargs).rstrip().lstrip("\n")
        child_str = textwrap.dedent(child_str)
        if not child_str:
            child = child.next
            continue

        if child.ctype == "inline":
            s = " "
        elif child.ctype == "list":
            child_str = textwrap.indent(child_str, " ")
            s = "\n"
        else:
            s = "\n\n"

        if (
            length_fn(child_str) + length_fn(s) + length_fn(current_chunk.content)
            <= max_size
        ):
            # Can add the child to the current chunk
            current_chunk.content += s + child_str
            current_chunk.metadata["originals"].extend(child.get_ids())
        else:
            # The resulting chunk will be too large, create a new chunk
            if current_chunk.content:
                current_chunk.content = current_chunk.content.strip("\n")
                result.append(current_chunk)

            current_chunk = Chunk(
                mimetype="plain/text",
                ctype="flattened_markdown",
                content="",
                metadata={"originals": child.get_ids()},
            )
            if length_fn(child_str) > max_size:
                # Split the text recursively based on the separators
                seps = separators or _default_separators(child_str)
                splitteds = split_text(
                    child_str,
                    length_fn,
                    max_size,
                    seps,
                    keep_separator=False,
                    is_separator_regex=False,
                )
                result.extend(
                    [
                        Chunk(
                            mimetype="plain/text",
                            ctype="flattened_markdown",
                            content=split,
                            metadata={"originals": child.get_ids()},
                        )
                        for split in splitteds[:-1]
                    ]
                )
                current_chunk.content = splitteds[-1]
            else:
                current_chunk.content = child_str

        child = child.next

    if current_chunk.content:
        current_chunk.content = current_chunk.content.strip("\n")
        result.append(current_chunk)

    return result


class FlattenToMarkdown(BaseOperation):

    _len_fns = {
        "len": len,
        "word_len": word_len,
    }

    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        max_size=-1,
        length_fn: Callable[[str], int] | None | str = word_len,
        separators: Optional[list[str]] = None,
        **kwargs,
    ) -> ChunkGroup:
        """Flatten the chunk tree to a single level of chunks, where each chunk has
        a length smaller than max_size. If a chunk is larger than max_size,
        it will be split into smaller chunks, based on character separators.

        Args:
            chunk: ChunkGroup to combine
            max_size: maximum size of chunks to return. If -1, no limit
            length_fn: function to measure text length
            separators: list of characters to split on. If None, default
                separators are used, which are decreasing consecutive whitespaces,
                sentence terminators (e.g. .,!), clause separators (e.g. :;),
                sentence interrupters (e.g. /,\\), and word joiners (e.g. -&).

        Returns:
            ChunkGroup: list of Chunk, where each Chunk will have the `.next` chunk
                and `.prev` chunk set to the next and previous chunk in the list.
        """
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        if max_size == -1:
            max_size = float("inf")

        # Resolve length function
        if isinstance(length_fn, str):
            length = cls._len_fns[length_fn]
        elif length_fn is None:
            length = word_len
        else:
            length = length_fn

        output = ChunkGroup()
        for root in chunks:
            flattened_chunks = flatten_chunk_to_markdown(
                root,
                max_size,
                length,
                separators,
                {},
                **kwargs,
            )
            for idx, chunk in enumerate(flattened_chunks[1:]):
                chunk.prev = flattened_chunks[idx - 1]
                flattened_chunks[idx].next = chunk

            output.append(flattened_chunks[0])

        return output
