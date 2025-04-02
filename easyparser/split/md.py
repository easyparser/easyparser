from typing import Callable

from easyparser.base import BaseOperation, Chunk, ChunkGroup

from .utils import word_len

_len_fns = {
    "len": len,
    "word_len": word_len,
}


def combine_lvl(heading_start_idx, heading_levels, min_size, length_fn, content):
    """
    Combine sections of the same consecutive level to be at least a given size.

    Args:
        heading_start_idx: List of character offsets of the headings
        heading_levels: List of heading levels
        min_size: Minimum section size
        length_fn: Function to calculate the length of a section
        content: The content string

    Returns:
        Tuple of (new_heading_start_idx, new_heading_levels)
    """
    if len(heading_start_idx) <= 1:
        return heading_start_idx, heading_levels

    # Create result arrays
    result_idx = [heading_start_idx[0]]
    result_levels = [heading_levels[0]]

    # Process each heading
    i = 1
    while i < len(heading_start_idx):
        current_idx = heading_start_idx[i]
        current_level = heading_levels[i]
        prev_idx = result_idx[-1]
        prev_level = result_levels[-1]

        # Calculate size of previous section
        section_size = length_fn(content[prev_idx:current_idx])

        # If the section is too small and both headings have the same level
        if section_size < min_size and current_level == prev_level:
            # Skip this heading (merge with previous section)
            i += 1
            continue

        # Add this heading to the result
        result_idx.append(current_idx)
        result_levels.append(current_level)
        i += 1

    return result_idx, result_levels


def collapse_lvl(heading_start_idx, heading_levels, min_size, length_fn, content):
    """Collapse the section if the section only contains one child section, and if
    either the section or the child section is smaller than the given size.
    """
    # Make copies of the input lists
    new_idx = heading_start_idx.copy()
    new_lvl = heading_levels.copy()

    if len(new_idx) <= 1:
        return new_idx, new_lvl

    # Process headings from the end to avoid index shifting problems
    i = len(new_idx) - 2  # Start from the second last element

    while i >= 0:
        current_level = new_lvl[i]

        # Find direct children of this heading
        direct_children, _clvl = [], -1
        j = i + 1
        while j < len(new_lvl) and new_lvl[j] > current_level:
            if not direct_children or new_lvl[j] == _clvl:
                direct_children.append(j)
                _clvl = new_lvl[j]
            j += 1

        # If there are no child sections or more than one child section, skip
        if len(direct_children) != 1:
            i -= 1
            continue

        child_idx = direct_children[0]
        child_level = new_lvl[child_idx]

        # Calculate section sizes
        # Parent section size is from its start to its child's start
        parent_size = length_fn(content[new_idx[i] : new_idx[child_idx]])

        # Child section size is from its start to the next heading's start
        if child_idx + 1 < len(new_idx):
            child_size = length_fn(content[new_idx[child_idx] : new_idx[child_idx + 1]])
        else:
            child_size = length_fn(content[new_idx[child_idx] :])

        # Check if the child has its own children
        has_grandchildren = False
        if child_idx + 1 < len(new_lvl) and new_lvl[child_idx + 1] > child_level:
            has_grandchildren = True

        # Collapse if child has no children and either section is smaller than min_size
        if not has_grandchildren and (parent_size < min_size or child_size < min_size):
            # Remove the child heading
            new_idx.pop(child_idx)
            new_lvl.pop(child_idx)
            # Start over since indices have changed
            i = len(new_idx) - 2
            continue

        i -= 1

    return new_idx, new_lvl


def map_bytestring_index_to_string_index(byte_string, byte_indices):
    """Convert the bytestring index to string index"""
    # Sort indices to process them in order
    sorted_indices = sorted(byte_indices)

    # Create a mapping from byte indices to character indices
    byte_to_char_map = {}

    # Initialize counters
    char_count = 0

    # Process each byte and track character positions
    start = 0
    for byte_pos in sorted_indices:
        char = byte_string[start:byte_pos].decode("utf-8")
        char_count += len(char)
        byte_to_char_map[byte_pos] = char_count
        start = byte_pos

    # Return the results in the original order
    return byte_to_char_map


class MarkdownSplitByHeading(BaseOperation):
    _len_fns = {
        "len": len,
        "word_len": word_len,
    }

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        min_chunk_size: int = -1,
        length_fn: Callable[[str], int] | None | str = "word_len",
        **kwargs,
    ) -> ChunkGroup:
        """Split large chunks of text into smaller chunks based on Markdown heading,
        where each chunk is not larger than a given size.

        Args:
            min_chunk_size: the minimum size of a chunk. If a chunk is smaller than
                this size, it will be merged with the next chunk. If -1, there is
                no minimum.
        """
        import tree_sitter_markdown
        from tree_sitter import Language, Parser

        # Resolve chunk
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        # Resolve length function
        if isinstance(length_fn, str):
            ln = _len_fns[length_fn]
        elif length_fn is None:
            ln = word_len
        else:
            ln = length_fn

        parser = Parser(Language(tree_sitter_markdown.language()))

        output = ChunkGroup()
        for mc in chunk:
            ct = mc.content
            ctb = ct.encode("utf-8")
            if not isinstance(ct, str) or (
                min_chunk_size != -1 and ln(ct) < min_chunk_size
            ):
                output.add_group(ChunkGroup(root=mc))
                continue

            tree = parser.parse(ctb)
            ts_root = tree.root_node

            # Get chunk header range
            stack, h_start, h_end, level = [ts_root], [], {}, []
            while stack:
                ts_node = stack.pop()
                if "heading" in ts_node.type:
                    lvl = None
                    for child in ts_node.children:
                        if child.type == "atx_h1_marker":
                            lvl = 1
                        elif child.type == "atx_h2_marker":
                            lvl = 2
                        elif child.type == "atx_h3_marker":
                            lvl = 3
                        elif child.type == "atx_h4_marker":
                            lvl = 4
                        elif child.type == "atx_h5_marker":
                            lvl = 5
                        elif child.type == "atx_h6_marker":
                            lvl = 6
                        elif child.type == "setext_h1_underline":
                            lvl = 1
                        elif child.type == "setext_h2_underline":
                            lvl = 2

                    if lvl is None:
                        continue

                    h_start.append(ts_node.start_byte)
                    h_end[ts_node.start_byte] = ts_node.end_byte
                    level.append(lvl)

                for i in range(ts_node.child_count - 1, -1, -1):
                    if child := ts_node.children[i]:
                        stack.append(child)

            if len(h_start) < 2:
                # 1 or 0 heading, so nothing to split
                output.add_group(ChunkGroup(root=mc))
                continue

            # Convert from byte index to character index
            byte_to_char_map = map_bytestring_index_to_string_index(
                ctb, h_start + list(h_end.values())
            )
            h_start = [byte_to_char_map[idx] for idx in h_start]
            h_end = {
                byte_to_char_map[idx]: byte_to_char_map[end]
                for idx, end in h_end.items()
            }

            # Combine small chunks to larger chunk
            if min_chunk_size != -1:
                while True:
                    new_h, new_l = collapse_lvl(h_start, level, min_chunk_size, ln, ct)
                    new_h, new_l = combine_lvl(new_h, new_l, min_chunk_size, ln, ct)
                    if len(new_h) == len(h_start):
                        break
                    h_start, level = new_h, new_l

            # Build the chunks
            result = []
            parents = []
            prev, prev_lvl = mc, 0
            for idx in range(len(h_start)):
                lvl = level[idx]
                start_idx = h_start[idx]
                end_idx = h_end[start_idx]
                heading = ct[start_idx:end_idx].strip()

                if idx == 0:
                    # Also include the content before the first heading
                    start_idx = 0

                if lvl > prev_lvl:
                    parents.append((prev, prev_lvl))
                    prev, prev_lvl = None, lvl
                elif lvl < prev_lvl:
                    prev, prev_lvl = parents.pop()

                if idx + 1 == len(h_start):
                    content = ct[start_idx:]
                    text = content
                else:
                    content = ct[start_idx : h_start[idx + 1]]
                    for i_ in range(idx, len(h_start)):
                        if level[i_] <= lvl:
                            text = ct[start_idx : h_start[i_]]
                    else:
                        text = content

                chunk = Chunk(
                    mimetype="plain/text",
                    content=content,
                    text=text,
                    origin=mc.origin,
                    parent=parents[-1][0],
                    prev=prev,
                    metadata={
                        "heading_level": lvl,
                        "heading": heading,
                    },
                    history=mc.history
                    + [cls.name(min_chunk_size=min_chunk_size, length_fn=length_fn)],
                )
                result.append(chunk)
                if prev is not None:
                    prev.next = chunk
                else:
                    parents[-1][0].child = chunk
                prev = chunk

            output.add_group(ChunkGroup(root=mc, chunks=result))

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["tree-sitter", "tree-sitter-markdown"]


class MarkdownTOC(BaseOperation):
    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Chunk the markdown text by the headings. Construct the TOC along the way.

        The chunking is done by splitting the text at the headings, ensure that each
        of the chunks is not smaller than the given size.
        """
        ...


class MarkdownAnnotateImage(BaseOperation): ...


class MarkdownAnnotateTable(BaseOperation): ...


class MarkdownAnnotateCodeBlock(BaseOperation): ...


class MardownSummarizeSection(BaseOperation): ...
