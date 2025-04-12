import logging

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType

logger = logging.getLogger(__name__)


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


def convert_setext_to_atx(markdown: str) -> str:
    """Convert setext-style headings to ATX-style headings in markdown text.

    Args:
        markdown: Markdown text with setext headings

    Returns:
        str: Markdown text with ATX headings
    """
    if not markdown:
        return markdown

    lines = [(each.strip(), each) for each in markdown.split("\n")]
    result = []
    i = 0
    in_code_block = False
    line_count = len(lines)

    while i < line_count:
        if lines[i][0].startswith("```"):
            in_code_block = not in_code_block
            result.append(lines[i][1])
            i += 1
            continue

        if in_code_block:
            result.append(lines[i][1])
            i += 1
            continue

        if i + 1 < line_count and lines[i][0]:
            next_line_stripped = lines[i + 1][0]

            if next_line_stripped:
                first_char = next_line_stripped[0]
                if (first_char == "=" or first_char == "-") and (
                    next_line_stripped.count(first_char) == len(next_line_stripped)
                ):

                    level = 1 if first_char == "=" else 2
                    result.append(f"{'#' * level} {lines[i][1]}")
                    i += 2
                    continue

        result.append(lines[i][1])
        i += 1

    return "\n".join(result)


def parse_tree_sitter_node(node, content) -> Chunk:
    """Parse a tree-sitter node and its children into a Chunk.

    Args:
        node: the tree-sitter node to parse
        content: the bytestring
    """
    if node.type == "pipe_table":
        text = content[node.start_byte : node.end_byte].decode("utf-8").strip()
        return Chunk(
            mimetype="text/markdown",
            ctype=CType.Table,
            content=text,
            text=text,
        )
    elif node.type == "paragraph":
        text = content[node.start_byte : node.end_byte].decode("utf-8").strip()
        return Chunk(
            mimetype="text/plain",
            ctype=CType.Para,
            content=text,
            text=text,
        )
    elif node.type == "html_block":
        text = content[node.start_byte : node.end_byte].decode("utf-8").strip()
        return Chunk(
            mimetype="text/html",
            ctype=CType.Code,
            content=text,
            text=text,
        )
    elif node.type == "fenced_code_block":
        text = content[node.start_byte : node.end_byte].decode("utf-8").strip()
        return Chunk(
            mimetype="text/plain",
            ctype=CType.Code,
            content=text,
            text=text,
        )
    elif node.type == "section":
        chunk = Chunk(mimetype="text/markdown", ctype=CType.Para, content="", text="")
        prev = None
        for idx, child in enumerate(node.children):
            child_chunk = parse_tree_sitter_node(child, content)
            if idx == 0 and child_chunk.ctype == CType.Header:
                chunk = child_chunk
                continue
            child_chunk.parent = chunk
            if chunk.child is None:
                chunk.child = child_chunk
            if prev is not None:
                prev.next = child_chunk
                child_chunk.prev = prev
            prev = child_chunk
        return chunk
    elif "heading" in node.type:
        text = content[node.start_byte : node.end_byte].decode("utf-8").strip()
        lvl = None
        for child in node.children:
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

        chunk = Chunk(
            mimetype="text/plain",
            content=text,
        )

        if lvl is None:
            chunk.ctype = CType.Para
        else:
            chunk.ctype = CType.Header
            chunk.metadata = {"level": lvl}

        return chunk
    elif node.type == "document":
        chunk = Chunk(mimetype="text/markdown", ctype=CType.Para, content="")
        prev = None
        for idx, child in enumerate(node.children):
            child_chunk = parse_tree_sitter_node(child, content)
            child_chunk.parent = chunk
            if idx == 0:
                chunk.child = child_chunk
            else:
                prev.next = child_chunk
                child_chunk.prev = prev
            prev = child_chunk
        return chunk
    elif node.type == "list":
        chunk = Chunk(mimetype="text/markdown", ctype=CType.List, content="")
        prev = None
        # handle list items
        for idx, child in enumerate(node.children):
            if child.children[0].type == "list_marker_minus":
                marker = "-"
            elif child.children[0].type == "list_marker_plus":
                marker = "+"
            elif child.children[0].type == "list_marker_star":
                marker = "*"
            elif child.children[0].type == "list_marker_parenthesis":
                marker = f"{idx+1})"
            elif child.children[0].type == "list_marker_period":
                marker = f"{idx+1}."
            else:
                raise NotImplementedError(
                    f"List marker {child.children[0].type} not implemented"
                )
            list_item = Chunk(
                mimetype="text/markdown", ctype=CType.List, content=marker
            )
            item_content = parse_tree_sitter_node(child.children[1], content)
            item_content.ctype = CType.Inline
            item_content.parent = list_item
            list_item.child = item_content
            if len(child.children) == 3:
                nested_list = parse_tree_sitter_node(child.children[2], content)

                nested_list.parent = list_item
                item_content.next = nested_list
                nested_list.prev = item_content

            list_item.parent = chunk
            if idx == 0:
                chunk.child = list_item
            if prev is not None:
                prev.next = list_item
                list_item.prev = prev
            prev = list_item
        return chunk
    else:
        raise NotImplementedError(f"Node type {node.type} not implemented")


class Markdown(BaseOperation):
    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Split large chunks of text into smaller chunks based on Markdown heading,
        where each chunk is not larger than a given size.
        """
        import tree_sitter_markdown
        from tree_sitter import Language, Parser

        # Resolve chunk
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        parser = Parser(Language(tree_sitter_markdown.language()))

        output = ChunkGroup()
        for mc in chunk:
            location = mc.origin.location
            with open(location) as f:
                ct = f.read()

            # Convert setext to atx
            ct_len = len(ct)
            ct = convert_setext_to_atx(ct)
            if len(ct) != ct_len:
                logger.warning("Converted setext to atx")
                mc.content = ct

            ctb = ct.encode("utf-8")
            if not isinstance(ct, str):
                output.add_group(ChunkGroup(root=mc))
                continue

            tree = parser.parse(ctb)
            ts_root = tree.root_node

            child = parse_tree_sitter_node(ts_root, ctb)
            mc.child = child
            child.parent = mc
            output.add_group(ChunkGroup(root=mc))

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["tree-sitter", "tree-sitter-markdown"]
