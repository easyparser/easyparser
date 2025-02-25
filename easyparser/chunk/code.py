from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node


def chunk_ts_node(
    node: "Node", original: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """Chunk the tree-sitter node recursively"""
    chunks = []
    current_chunk = ""
    for child in node.children:
        if child.end_byte - child.start_byte > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
            chunks.extend(chunk_ts_node(child, original, chunk_size, chunk_overlap))
        elif child.end_byte - child.start_byte + len(current_chunk) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = original[child.start_byte : child.end_byte]
        else:
            current_chunk += original[child.start_byte : child.end_byte]
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def chunk_code(
    text: str,
    language: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Chunk the code recursively

    Args:
        text: text to split
        language: language of the code
        chunk_size: maximum size of chunks to return
        chunk_overlap: overlap between chunks
        length_fn: function to measure text length

    Returns:
        list of text chunks
    """
    try:
        from tree_sitter import Node  # noqa: F401
        from tree_sitter_language_pack import get_parser
    except ImportError:
        raise ImportError(
            "Please install `pip install tree-sitter tree-sitter-language-pack`"
        )

    parser = get_parser(language)
    tree = parser.parse(text.encode("utf-8"))
    if not tree:
        return [text]

    root_node = tree.root_node
    if root_node.end_byte - root_node.start_byte <= chunk_size:
        return [text]

    chunks = chunk_ts_node(tree.root_node, text, chunk_size, chunk_overlap)
    return chunks
