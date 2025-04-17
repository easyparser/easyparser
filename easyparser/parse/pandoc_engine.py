import json
import logging
import tempfile
from pathlib import Path
from typing import Generator

try:
    import pypandoc
except ImportError:
    pass

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType
from easyparser.mime.guess import guess_mimetype

logger = logging.getLogger(__name__)


_BLOCK_ELEMENTS = {
    "BlockQuote",
    "BulletList",
    "CodeBlock",
    "DefinitionList",
    "Div",
    "Figure",
    "Header",
    "HorizontalRule",
    "LineBlock",
    "Null",
    "OrderedList",
    "Para",
    "Plain",
    "RawBlock",
    "Table",
}

_INLINE_ELEMENTS = {
    "Cite",
    "Code",
    "Emph",
    "Image",
    "LineBreak",
    "Link",
    "Math",
    "Note",
    "Quoted",
    "RawInline",
    "SmallCaps",
    "SoftBreak",
    "Space",
    "Span",
    "Str",
    "Strikeout",
    "Strong",
    "Superscript",
    "Subscript",
}


def parse_ordered_list(ordered_list_node: dict) -> Chunk:
    """Parse a Pandoc OrderedList node from JSON AST to a dictionary

    Args:
        ordered_list_node: A Pandoc OrderedList node from the JSON AST

    Returns:
        A dictionary containing parsed content with the following keys:
            - 'text': A single string containing the full formatted list text
            - 'references': List of references found in the list
            - 'images': List of images found in the list
            - 'formatting': List of formatting elements
            - 'links': List of links
            - 'citations': List of citations
            - 'other_attributes': Other Pandoc attributes
    """
    metadata = {
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    # List attributes
    list_attrs = ordered_list_node["c"][0]
    start_number = list_attrs[0]
    list_style = list_attrs[1]
    list_delimiter = list_attrs[2]
    metadata["other_attributes"].append(
        {
            "type": CType.List,
            "start": start_number,
            "style": list_style,
            "delimiter": list_delimiter,
        }
    )

    chunk = Chunk(mimetype="text/plain", ctype=CType.List, metadata=metadata)
    prev_chunk = None

    items = ordered_list_node["c"][1]
    for idx, item in enumerate(items):
        # Construct 1. or (a) or A. or i. etc.
        item_number = idx + start_number
        marker = _generate_list_marker(item_number, list_style, list_delimiter)

        # Process the item content
        list_item = process_blocks(item)
        list_item.ctype = CType.List

        # Pandoc represent list components as blocks, but we want to represent as
        # inlines because they aren't structurally children
        list_child = list_item.child
        while list_child:
            if list_child.ctype != CType.List:
                list_child.ctype = CType.Inline
            list_child = list_child.next

        if marker:
            list_item.content = marker + list_item.content

        if idx == 0:
            chunk.child = list_item

        if prev_chunk:
            prev_chunk.next = list_item
            list_item.prev = prev_chunk

        list_item.parent = chunk
        prev_chunk = list_item

    return chunk


def parse_bullet_list(bullet_list_node: dict) -> Chunk:
    """Parse a BulletList node, similar to OrderedList but with bullet markers."""
    metadata = {
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [{"type": "list_type", "value": "bullet"}],
    }
    chunk = Chunk(mimetype="text/plain", ctype=CType.List, metadata=metadata)
    prev_chunk = None

    # Add list type to other_attributes
    items = bullet_list_node["c"]

    for idx, item in enumerate(items):
        list_item = process_blocks(item)
        list_item.ctype = CType.List
        list_item.content = "• " + list_item.content
        list_item.content = list_item.content.strip()

        # Pandoc represent list components as blocks, but we want to represent as
        # inlines because they aren't structurally children
        list_child = list_item.child
        while list_child:
            if list_child.ctype != CType.List:
                list_child.ctype = CType.Inline
            list_child = list_child.next

        if idx == 0:
            chunk.child = list_item

        if prev_chunk:
            prev_chunk.next = list_item
            list_item.prev = prev_chunk

        list_item.parent = chunk
        prev_chunk = list_item

    return chunk


def parse_header(header_node) -> Chunk:
    """Convert a Pandoc header node from JSON AST into a structured dictionary

    Args:
        header_node: a Pandoc AST header node in Pandoc format

    Returns:
        A dictionary containing parsed content with the following keys:
            - 'type': The type of the element (e.g., header)
            - 'level': The header level (1, 2, 3, etc.)
            - 'text': A single string containing the full formatted list text
            - 'references': List of references found in the list
            - 'images': List of images found in the list
            - 'formatting': List of formatting elements
            - 'links': List of links
            - 'citations': List of citations
            - 'other_attributes': Other Pandoc attributes
    """
    metadata = {"level": 1}
    header_data = header_node["c"]

    # The first element is the header level (1, 2, 3, etc.)
    if len(header_data) > 0 and isinstance(header_data[0], int):
        metadata["level"] = header_data[0]

    # The second element is the attributes array [id, classes, key-value pairs]
    if len(header_data) > 1 and isinstance(header_data[1], list):
        attr = header_data[1]
        if len(attr) > 0:
            metadata["id"] = attr[0]
        if len(attr) > 1:
            metadata["classes"] = attr[1]
        if len(attr) > 2:
            for k, v in attr[2]:
                metadata[str(k)] = v

    # The third element is the content array
    chunk = process_inlines(header_data[2])
    chunk.mimetype = "text/plain"
    chunk.ctype = CType.Header
    if chunk.metadata is None:
        chunk.metadata = metadata
    else:
        chunk.metadata.update(metadata)

    return chunk


def parse_figure(figure_node: dict) -> Chunk:
    """Parse a Pandoc Figure node from JSON AST to Chunk"""
    metadata = {
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    # Extract attributes, caption, and content
    attrs, caption, figure_content = figure_node["c"]
    chunk = process_blocks(figure_content)
    chunk.ctype = CType.Figure
    chunk.metadata = metadata

    # Process attributes
    attr_id, classes, key_values = attrs
    if attr_id:
        metadata["other_attributes"].append({"id": attr_id})

    for cls in classes:
        metadata["formatting"].append({"class": cls})

    for key, value in key_values:
        metadata["other_attributes"].append({key: value})

    # Process caption
    if caption and caption[1]:
        caption_chunk = process_blocks(caption[1])
        caption_chunk.parent = chunk
        chunk.child = caption_chunk

    return chunk


def parse_table(table_node: dict) -> Chunk:
    """Parse table node into Chunk"""
    # Due to complexity of table, convert to markdown for now
    santinel = pypandoc.convert_text("", to="json", format="markdown")
    santinel = json.loads(santinel)
    santinel["blocks"] = [table_node]
    table_markdown = pypandoc.convert_text(
        json.dumps(santinel), to="markdown", format="json"
    )
    return Chunk(
        mimetype="text/plain",
        ctype=CType.Table,
        content=table_markdown,
        text=table_markdown,
    )


def _generate_list_marker(number, style, delimiter) -> str:
    """Generate the appropriate list marker (e.g. 1. or (a) or A. or i. etc.)

    Args:
        number: the item number
        style: the list style (e.g., Decimal, LowerRoman, UpperAlpha)
        delimiter: the list delimiter (e.g., Period, OneParen, TwoParens)

    Returns:
        the formatted list marker
    """
    marker = ""

    # Extract style and delimiter values if they are in Pandoc {'t': 'Type'} format
    if isinstance(style, dict) and "t" in style:
        style = style["t"]
    if isinstance(delimiter, dict) and "t" in delimiter:
        delimiter = delimiter["t"]

    # Convert number to appropriate style
    if style == "Decimal":
        marker = str(number)
    elif style == "LowerRoman":
        marker = _to_roman(number).lower()
    elif style == "UpperRoman":
        marker = _to_roman(number)
    elif style == "LowerAlpha":
        marker = _to_alpha(number).lower()
    elif style == "UpperAlpha":
        marker = _to_alpha(number)
    else:
        marker = str(number)

    # Apply delimiter
    if delimiter == "Period":
        marker += "."
    elif delimiter == "OneParen":
        marker += ")"
    elif delimiter == "TwoParens":
        marker = "(" + marker + ")"
    else:
        marker += "."

    return marker + " "


def _to_roman(num):
    """Convert number to Roman numeral"""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ""
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syms[i]
            num -= val[i]
        i += 1
    return roman_num


def _to_alpha(num):
    """Convert number to alphabetical representation (A, B, C, ... Z, AA, AB, ...)"""
    result = ""
    while num > 0:
        num, remainder = divmod(num - 1, 26)
        result = chr(65 + remainder) + result
    return result


def process_pandoc(blocks: list[dict]) -> Generator[Chunk, None, None]:
    """Parse a Pandoc JSON block AST file into a structured dictionary"""
    for idx, bl in enumerate(blocks):
        if bl["t"] in _BLOCK_ELEMENTS:
            yield process_block(bl)
        else:
            raise NotImplementedError(f"Block type {bl['t']} at {idx} not implemented.")


def process_blocks(blocks: list[dict]) -> Chunk:
    """Condense a list of blocks from Pandoc AST into single object

    Args:
        blocks: List of Pandoc AST block nodes

    Returns:
        A dictionary containing parsed content with the following keys:
            - 'text': A single string containing the full formatted list text
            - 'references': List of references found in the list
            - 'images': List of images found in the list
            - 'formatting': List of formatting elements
            - 'links': List of links
            - 'citations': List of citations
            - 'other_attributes': Other Pandoc attributes
    """
    chunk = Chunk(mimetype="text/plain", ctype=CType.Para, content="")
    if not blocks:
        return chunk

    chunks = []
    for block in blocks:
        block_result = process_block(block)
        chunks.append(block_result)

    if not chunk:
        return chunk
    if len(chunks) == 1:
        return chunks[0]

    prev_chunk = chunks[0]
    prev_chunk.parent = chunk
    chunk.child = prev_chunk
    for ch in chunks[1:]:
        prev_chunk.next = ch
        ch.prev = prev_chunk
        prev_chunk = ch
        ch.parent = chunk

    return chunk


def process_block(block: dict) -> Chunk:
    """Process a single block-level Pandoc AST element"""
    metadata = {
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    block_type = block.get("t")
    content = block.get("c", [])

    if block_type == "Plain":
        return process_inlines(content)
    elif block_type == "Para":
        chunk = process_inlines(content)
        chunk.ctype = CType.Para
        return chunk
    elif block_type == "Header":
        return parse_header(block)
    elif block_type == "OrderedList" or block_type == "BulletList":
        if block_type == "OrderedList":
            inline_result = parse_ordered_list(block)
        else:
            inline_result = parse_bullet_list(block)
        text = ""
        child = inline_result.child
        while child:
            text += child.text + "\n"
            child = child.next
        inline_result.text = text.strip()
        return inline_result
    elif block_type == "BlockQuote":
        return process_blocks(content)
    elif block_type == "RawBlock":
        attrs = content[0]
        raw_content = content[1]
        metadata = {
            "type": "raw_block",
            "format": attrs,
            "start": 0,
            "end": len(raw_content),
        }

        return Chunk(
            mimetype="text/plain",
            ctype="__pandoc__rawblock__",
            content=raw_content,
            text=raw_content,
            metadata=metadata,
        )
    elif block_type == "Div":
        attrs = content[0]
        chunk = process_blocks(content[1])
        chunk.ctype = "Para"
        return chunk
    elif block_type == "CodeBlock":
        attrs = content[0]

        chunk = Chunk(
            mimetype="text/plain",
            ctype=CType.Code,
            content=content[1],
            text=content[1],
        )

        metadata["formatting"].append(
            {
                "type": "code_block",
                "language": attrs[1][0] if len(attrs[1]) > 0 else "",
            }
        )

        if attrs[0]:
            metadata["other_attributes"].append(
                {"type": "identifier", "value": attrs[0]}
            )
        if attrs[2]:
            for attr in attrs[2]:
                metadata["other_attributes"].append(
                    {"type": "attribute", "key": attr[0], "value": attr[1]}
                )
        chunk.metadata = metadata
        return chunk
    elif block_type == "Figure":
        return parse_figure(block)
    elif block_type == "Table":
        return parse_table(block)
    elif block_type == "HorizontalRule":
        return Chunk(mimetype="text/plain", ctype=CType.Para, content="-----\n")
    else:
        raise NotImplementedError(f"Unknown pandoc block type: {block_type}")


def process_inlines(inlines: list[dict]) -> Chunk:
    """Condense a list of inline elements from Pandoc AST into single object

    Args:
        inlines: List of Pandoc AST inline nodes

    Returns:
        dict: A dictionary with text and metadata
    """
    metadata = {
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }
    chunk = Chunk(
        mimetype="text/plain", ctype=CType.Inline, content="", metadata=metadata
    )
    for inline in inlines:
        if not isinstance(inline, dict):
            logger.warning(f"Unknown pandoc inline type: {type(inline)}. Expect dict.")
            continue

        inline_type = inline.get("t")
        start_pos = len(chunk.content)
        content = inline.get("c", [])

        if inline_type == "Str":
            chunk.content += content
        elif inline_type == "Space":
            chunk.content += " "
        elif inline_type == "SoftBreak":
            chunk.content += " "
        elif inline_type == "LineBreak":
            chunk.content += "\n"
        elif (
            inline_type == "Emph"
            or inline_type == "Strong"
            or inline_type == "Underline"
        ):
            nested_chunk = process_inlines(content)
            chunk.merge(nested_chunk)
            if chunk.metadata is None:
                chunk.metadata = {"formatting": []}
            chunk.metadata["formatting"].append(
                {
                    "type": inline_type,
                    "position": start_pos,
                    "end_position": len(chunk.content),
                }
            )
        elif inline_type == "Link":
            nested = process_inlines(content[1])
            chunk.merge(nested)

            if chunk.metadata is None:
                chunk.metadata = {"links": [], "other_attributes": []}
            chunk.metadata["links"].append(
                {
                    "text": nested.content,
                    "url": content[2][0],
                    "title": content[2][1],
                    "start": start_pos,
                    "end": len(chunk.content),
                }
            )

            if attrs := content[0]:
                if attrs[0]:  # Identifier
                    chunk.metadata["other_attributes"].append(
                        {"type": "identifier", "value": attrs[0]}
                    )
                if attrs[1]:  # Classes
                    chunk.metadata["other_attributes"].append(
                        {"type": "classes", "value": attrs[1]}
                    )
                if attrs[2]:  # Key-value attributes
                    for attr in attrs[2]:
                        chunk.metadata["other_attributes"].append(
                            {"type": "attribute", "key": attr[0], "value": attr[1]}
                        )
        elif inline_type == "Note":
            nested = process_blocks(content)
            nested.content = f"[Note: {nested.content}]"
            chunk.merge(nested)
        elif inline_type == "Image":
            attrs = content[0]
            nested = process_inlines(content[1])

            chunk.content += f"[Image: {nested.content}]"
            nested.summary = nested.content
            nested.metadata = {
                "url": content[2][0],
                "title": content[2][1],
                "start_position": start_pos,
                "end_position": len(chunk.content),
                "other_attributes": {},
            }

            if Path(content[2][0]).exists():
                # If the file exists locally, load the file
                nested.mimetype = guess_mimetype(content[2][0])
                with open(content[2][0], "rb") as f:
                    nested.content = f.read()

            if attrs[0]:
                nested.metadata["other_attributes"]["identifier"] = attrs[0]
            if attrs[1]:
                nested.metadata["other_attributes"]["classes"] = attrs[1]
            if attrs[2]:
                for attr in attrs[2]:
                    nested.metadata["other_attributes"][str(attr[0])] = attr[1]

            nested.parent = chunk
            if chunk.child is None:
                chunk.child = nested
            else:
                last_child = chunk.child
                while last_child.next:
                    last_child = last_child.next
                last_child.next = nested
                nested.prev = last_child
        elif inline_type == "Code":
            attrs = content[0]
            code_text = content[1]

            chunk.content += code_text
            chunk.metadata["formatting"].append(
                {"type": "code", "start": start_pos, "end": start_pos + len(code_text)}
            )

            # Add identifier and attributes if present
            if attrs:
                if attrs[0]:  # Identifier
                    chunk.metadta["other_attributes"].append(
                        {
                            "type": "identifier",
                            "value": attrs[0],
                            "element_type": "code",
                            "element_position": start_pos,
                        }
                    )
                if attrs[1]:  # Classes
                    chunk.metadata["other_attributes"].append(
                        {
                            "type": "classes",
                            "value": attrs[1],
                            "element_type": "code",
                            "element_position": start_pos,
                        }
                    )
                if attrs[2]:  # Key-value attributes
                    for attr in attrs[2]:
                        chunk.metadata["other_attributes"].append(
                            {
                                "type": "attribute",
                                "key": attr[0],
                                "value": attr[1],
                                "element_type": "code",
                                "element_position": start_pos,
                            }
                        )
        elif inline_type == "Cite":
            citations = content[0]
            nested = process_inlines(content[1])
            chunk.merge(nested)

            for citation in citations:
                if isinstance(citation, dict):
                    cite_data = {
                        "start": start_pos,
                        "end": start_pos + len(chunk.content),
                    }

                    # Extract citation fields
                    if "citationId" in citation:
                        cite_data["id"] = citation["citationId"]
                    if "citationPrefix" in citation:
                        cite_data["prefix"] = process_inlines(
                            citation["citationPrefix"]
                        ).content
                    if "citationSuffix" in citation:
                        cite_data["suffix"] = process_inlines(
                            citation["citationSuffix"]
                        ).content
                    if "citationMode" in citation:
                        cite_data["mode"] = citation["citationMode"]
                    if "citationNoteNum" in citation:
                        cite_data["note_num"] = citation["citationNoteNum"]
                    if "citationHash" in citation:
                        cite_data["hash"] = citation["citationHash"]

                    chunk.metadata["citations"].append(cite_data)
        elif inline_type == "Quoted":
            quote_type = content[0]["t"] if isinstance(content[0], dict) else content[0]
            open_mark, close_mark = '"', '"'
            if quote_type == "SingleQuote":
                open_mark, close_mark = "'", "'"

            nested = process_inlines(content[1])
            nested.content += open_mark + nested.content + close_mark
            nested.text = open_mark + nested.text + close_mark
            chunk.merge(nested)
        elif inline_type == "RawInline":
            format_type = content[0]
            raw_content = content[1]

            # Add raw content to text
            chunk.content += raw_content

            # Track raw content
            chunk.metadata["formatting"].append(
                {
                    "type": "raw",
                    "format": format_type,
                    "start": start_pos,
                    "end": start_pos + len(chunk.content),
                }
            )
        elif inline_type == "Math":
            math_type = content[0]["t"] if isinstance(content[0], dict) else content[0]
            math_content = content[1]

            # Format based on math type
            if math_type == "InlineMath":
                math_content += f"${math_content}$"
            elif math_type == "DisplayMath":
                math_content = f"$${math_content}$$"

            chunk.content += math_content

            # Track math formatting
            chunk.metadata["formatting"].append(
                {
                    "type": "math",
                    "math_type": math_type,
                    "start": start_pos,
                    "end": start_pos + len(chunk.content),
                }
            )
        elif inline_type == "Span":
            attrs = content[0]
            span_content = content[1]

            # Process span content
            nested = process_inlines(span_content)
            chunk.merge(nested)

            # Track span formatting
            span_attr = {
                "type": "span",
                "start": start_pos,
                "end": start_pos + len(chunk.content),
            }

            # Add identifier and attributes
            if attrs[0]:  # Identifier
                span_attr["id"] = attrs[0]
            if attrs[1]:  # Classes
                span_attr["classes"] = attrs[1]
            if attrs[2]:  # Key-value attributes
                span_attr["attributes"] = {k: v for k, v in attrs[2]}

            if chunk.metadata is None:
                chunk.metadata = {"formatting": []}
            chunk.metadata["formatting"].append(span_attr)
        elif inline_type == "Superscript" or inline_type == "Subscript":
            nested = process_inlines(content)
            chunk.merge(nested)
            chunk.metadata["formatting"].append(
                {
                    "type": inline_type.lower(),
                    "start": start_pos,
                    "end": start_pos + len(chunk.content),
                }
            )
        else:
            logger.warning(f"Unknown pandoc inline node: {inline_type}")

    return chunk


class PandocEngine(BaseOperation):

    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        """Load structured document files using Pandoc.

        It is most suitable for text-heavy files while preserving document structure
        information, like Markdown, LaTeX or Word.
        """
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        output = ChunkGroup()
        for root in chunk:
            fp = root.origin.location

            media = tempfile.TemporaryDirectory(prefix="chunking_pandoc_media")
            json_string = pypandoc.convert_file(
                fp, to="json", extra_args=["--extract-media", media.name], sandbox=True
            )
            data = json.loads(json_string)
            bls = data["blocks"]

            parent_chunk_stacks = [root]
            last_chunk = None

            for ch in process_pandoc(bls):

                ch.parent = parent_chunk_stacks[-1]
                if ch.parent.child is None:
                    ch.parent.child = ch

                if ch.ctype == CType.Header:
                    lvl = ch.metadata["level"]
                    if lvl > parent_chunk_stacks[-1].metadata.get("level", 0):
                        # Go deeper
                        if (
                            last_chunk is not None
                            and last_chunk.id != parent_chunk_stacks[-1].id
                        ):
                            last_chunk.next = ch
                            ch.prev = last_chunk
                        parent_chunk_stacks.append(ch)
                    else:  # Go back to sibling of parent
                        while parent_chunk_stacks[-1].metadata.get("level", 0) >= lvl:
                            prev_sibling = parent_chunk_stacks.pop()
                        parent_chunk_stacks.append(ch)
                        prev_sibling.next = ch
                        ch.prev = prev_sibling

                    last_chunk = None
                    continue

                if last_chunk:
                    last_chunk.next = ch
                    ch.prev = last_chunk

                last_chunk = ch

            output.append(root)
            media.cleanup()

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["pypandoc"]
