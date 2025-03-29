import json
import logging
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Generator

from easyparser.base import BaseOperation, Chunk, ChunkGroup

logger = logging.getLogger(__name__)


_BLOCK_ELEMENTS = {
    "Plain",
    "Para",
    "CodeBlock",
    "BlockQuote",
    "OrderedList",
    "BulletList",
    "DefinitionList",
    "Header",
    "HorizontalRule",
    # "Table",
    "Div",
    "Null",
}

_INLINE_ELEMENTS = {
    "Str",
    "Space",
    "SoftBreak",
    "LineBreak",
    "Emph",
    "Strong",
    "Strikeout",
    "Superscript",
    "Subscript",
    "SmallCaps",
    "Quoted",
    "Cite",
    "Code",
    "Space",
    "Math",
    "RawInline",
    "Link",
    "Image",
    "Note",
    "Span",
}


def parse_ordered_list(ordered_list_node: dict) -> dict:
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
    result = {
        "text": "",
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
    result["other_attributes"].append(
        {
            "type": "list_attributes",
            "start": start_number,
            "style": list_style,
            "delimiter": list_delimiter,
        }
    )

    items = ordered_list_node["c"][1]
    for idx, item in enumerate(items):
        # Construct 1. or (a) or A. or i. etc.
        item_number = idx + start_number
        marker = _generate_list_marker(item_number, list_style, list_delimiter)

        # Process the item content
        item_result = process_blocks(item)
        if marker:
            item_result["text"] = marker + item_result["text"]

        if idx < len(items) - 1:
            item_result["text"] += "\n"

        result["text"] += item_result["text"]
        result["references"].extend(item_result["references"])
        result["images"].extend(item_result["images"])
        result["formatting"].extend(item_result["formatting"])
        result["links"].extend(item_result["links"])
        result["citations"].extend(item_result["citations"])
        result["other_attributes"].extend(item_result["other_attributes"])

    return result


def parse_bullet_list(bullet_list_node: dict) -> dict:
    """Parse a BulletList node, similar to OrderedList but with bullet markers."""
    result = {
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    # Add list type to other_attributes
    result["other_attributes"].append({"type": "list_type", "value": "bullet"})
    items = bullet_list_node["c"]

    for idx, item in enumerate(items):
        item_result = process_blocks(item)
        item_result["text"] = "• " + item_result["text"]
        if idx < len(items) - 1:
            item_result["text"] += "\n"

        result["text"] += item_result["text"]
        result["references"].extend(item_result["references"])
        result["images"].extend(item_result["images"])
        result["formatting"].extend(item_result["formatting"])
        result["links"].extend(item_result["links"])
        result["citations"].extend(item_result["citations"])
        result["other_attributes"].extend(item_result["other_attributes"])

    return result


def parse_header(header_node):
    """
    Convert a Pandoc header node from JSON AST into a structured dictionary.

    Args:
        header_node (dict): A Pandoc AST header node in Pandoc format

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
    result = {
        "type": "header",
        "level": 1,
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    header_data = header_node["c"]

    # The first element is the header level (1, 2, 3, etc.)
    if len(header_data) > 0 and isinstance(header_data[0], int):
        result["level"] = header_data[0]

    # The second element is the attributes array [id, classes, key-value pairs]
    if len(header_data) > 1 and isinstance(header_data[1], list):
        attr = header_data[1]
        if len(attr) > 0:
            result["id"] = attr[0]
        if len(attr) > 1:
            result["classes"] = attr[1]
        if len(attr) > 2:
            result["key_value_attributes"] = attr[2]

    # The third element is the content array
    if len(header_data) > 2 and isinstance(header_data[2], list):
        item_result = process_inlines(header_data[2])
        result["text"] = item_result["text"]
        result["references"].extend(item_result["references"])
        result["images"].extend(item_result["images"])
        result["formatting"].extend(item_result["formatting"])
        result["links"].extend(item_result["links"])
        result["citations"].extend(item_result["citations"])
        result["other_attributes"].extend(item_result["other_attributes"])

    return result


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


def process_pandoc(blocks: list[dict]) -> Generator[dict, None, list[dict]]:
    """Parse a Pandoc JSON block AST file into a structured dictionary"""
    result = []
    for idx, bl in enumerate(blocks):
        if bl["t"] in _BLOCK_ELEMENTS:
            yield process_block(bl)
        else:
            raise NotImplementedError(f"Block type {bl['t']} at {idx} not implemented.")

    return result


def process_blocks(blocks: list[dict]) -> dict:
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
    result = {
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    for idx, block in enumerate(blocks):
        block_result = process_block(block)

        # Append block text and metadata
        result["text"] += block_result["text"]
        result["references"].extend(block_result["references"])
        result["images"].extend(block_result["images"])
        result["formatting"].extend(block_result["formatting"])
        result["links"].extend(block_result["links"])
        result["citations"].extend(block_result["citations"])
        result["other_attributes"].extend(block_result["other_attributes"])

        # Add a space between blocks
        if not result["text"].endswith("\n") and idx < len(blocks) - 1:
            result["text"] += "\n"

    return result


def process_block(block: dict) -> dict:
    """Process a single block-level Pandoc AST element"""
    result = {
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

    if block_type == "Plain" or block_type == "Para":
        inline_result = process_inlines(content)
    elif block_type == "Header":
        return parse_header(block)
    elif block_type == "OrderedList":
        inline_result = parse_ordered_list(block)
        inline_result["text"] = textwrap.indent(inline_result["text"], "  ")
    elif block_type == "BulletList":
        inline_result = parse_bullet_list(block)
        inline_result["text"] = textwrap.indent(inline_result["text"], "  ")
    elif block_type == "BlockQuote":
        inline_result = process_blocks(content)
        inline_result["formatting"].append(
            {"type": "block_quote", "start": 0, "end": len(inline_result["text"])}
        )
    elif block_type == "CodeBlock":
        attrs = content[0]
        code = content[1]

        result["text"] = code
        result["formatting"].append(
            {
                "type": "code_block",
                "start": 0,
                "end": len(code),
                "language": attrs[1][0] if len(attrs[1]) > 0 else "",
            }
        )

        if attrs[0]:
            result["other_attributes"].append({"type": "identifier", "value": attrs[0]})
        if attrs[2]:
            for attr in attrs[2]:
                result["other_attributes"].append(
                    {"type": "attribute", "key": attr[0], "value": attr[1]}
                )
        return result
    else:
        logger.warning(f"Unknown pandoc block type: {block_type}")
        return result

    result["text"] = inline_result["text"]
    result["references"] = inline_result["references"]
    result["images"] = inline_result["images"]
    result["formatting"] = inline_result["formatting"]
    result["links"] = inline_result["links"]
    result["citations"] = inline_result["citations"]
    result["other_attributes"] = inline_result["other_attributes"]

    return result


def process_inlines(inlines: list[dict]) -> dict:
    """Condense a list of inline elements from Pandoc AST into single object

    Args:
        inlines: List of Pandoc AST inline nodes

    Returns:
        dict: A dictionary with text and metadata
    """
    result = {
        "text": "",
        "references": [],
        "images": [],
        "formatting": [],
        "links": [],
        "citations": [],
        "other_attributes": [],
    }

    for inline in inlines:
        if not isinstance(inline, dict):
            logger.warning(f"Unknown pandoc inline type: {type(inline)}. Expect dict.")
            continue

        inline_type = inline.get("t")
        start_pos = len(result["text"])
        content = inline.get("c", [])

        if inline_type == "Str":
            result["text"] += content
        elif inline_type == "Space":
            result["text"] += " "
        elif inline_type == "SoftBreak":
            result["text"] += " "
        elif inline_type == "LineBreak":
            result["text"] += "\n"
        elif (
            inline_type == "Emph"
            or inline_type == "Strong"
            or inline_type == "Underline"
        ):
            nested = process_inlines(content)
            result["text"] += nested["text"]
            result["formatting"].append(
                {
                    "type": inline_type,
                    "start": start_pos,
                    "end": start_pos + len(nested["text"]),
                }
            )
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])
        elif inline_type == "Link":
            # Link with format [attr, [content], [url, title]]
            attrs = content[0]
            link_content = content[1]
            target = content[2]

            link_url = target[0]
            link_title = target[1]

            # Process link text
            nested = process_inlines(link_content)
            result["text"] += nested["text"]

            # Add link metadata
            result["links"].append(
                {
                    "text": nested["text"],
                    "url": link_url,
                    "title": link_title,
                    "start": start_pos,
                    "end": start_pos + len(nested["text"]),
                }
            )

            # Copy other metadata
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])

            # Add identifier and attributes if present
            if attrs:
                if attrs[0]:  # Identifier
                    result["other_attributes"].append(
                        {"type": "identifier", "value": attrs[0]}
                    )
                if attrs[1]:  # Classes
                    result["other_attributes"].append(
                        {"type": "classes", "value": attrs[1]}
                    )
                if attrs[2]:  # Key-value attributes
                    for attr in attrs[2]:
                        result["other_attributes"].append(
                            {"type": "attribute", "key": attr[0], "value": attr[1]}
                        )
        elif inline_type == "Note":
            note_content = process_blocks(content)
            result["text"] += f" [Note: {note_content['text']}]"
            result["references"].extend(note_content["references"])
            result["images"].extend(note_content["images"])
            result["formatting"].extend(note_content["formatting"])
            result["links"].extend(note_content["links"])
            result["citations"].extend(note_content["citations"])
            result["other_attributes"].extend(note_content["other_attributes"])
        elif inline_type == "Image":
            # Image with format [attr, [alt text content], [url, title]]
            attrs = content[0]
            alt_content = content[1]
            target = content[2]

            image_url = target[0]
            image_title = target[1]

            # Process alt text
            nested = process_inlines(alt_content)
            alt_text = nested["text"]

            # Add image placeholder to text
            result["text"] += f"[Image: {alt_text}]"

            # Add image metadata
            result["images"].append(
                {
                    "alt_text": alt_text,
                    "url": image_url,
                    "title": image_title,
                    "position": start_pos,
                }
            )

            # Add identifier and attributes if present
            if attrs:
                if attrs[0]:  # Identifier
                    result["other_attributes"].append(
                        {
                            "type": "identifier",
                            "value": attrs[0],
                            "element_type": "image",
                            "element_position": start_pos,
                        }
                    )
                if attrs[1]:  # Classes
                    result["other_attributes"].append(
                        {
                            "type": "classes",
                            "value": attrs[1],
                            "element_type": "image",
                            "element_position": start_pos,
                        }
                    )
                if attrs[2]:  # Key-value attributes
                    for attr in attrs[2]:
                        result["other_attributes"].append(
                            {
                                "type": "attribute",
                                "key": attr[0],
                                "value": attr[1],
                                "element_type": "image",
                                "element_position": start_pos,
                            }
                        )
        elif inline_type == "Code":
            # Inline code with format [attr, code]
            attrs = content[0]
            code_text = content[1]

            result["text"] += code_text
            result["formatting"].append(
                {"type": "code", "start": start_pos, "end": start_pos + len(code_text)}
            )

            # Add identifier and attributes if present
            if attrs:
                if attrs[0]:  # Identifier
                    result["other_attributes"].append(
                        {
                            "type": "identifier",
                            "value": attrs[0],
                            "element_type": "code",
                            "element_position": start_pos,
                        }
                    )
                if attrs[1]:  # Classes
                    result["other_attributes"].append(
                        {
                            "type": "classes",
                            "value": attrs[1],
                            "element_type": "code",
                            "element_position": start_pos,
                        }
                    )
                if attrs[2]:  # Key-value attributes
                    for attr in attrs[2]:
                        result["other_attributes"].append(
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
            cite_content = content[1]

            # Process citation text
            nested = process_inlines(cite_content)
            result["text"] += nested["text"]

            # Add citation metadata
            for citation in citations:
                if isinstance(citation, dict):
                    cite_data = {
                        "start": start_pos,
                        "end": start_pos + len(nested["text"]),
                    }

                    # Extract citation fields
                    if "citationId" in citation:
                        cite_data["id"] = citation["citationId"]
                    if "citationPrefix" in citation:
                        prefix_text = process_inlines(citation["citationPrefix"])[
                            "text"
                        ]
                        cite_data["prefix"] = prefix_text
                    if "citationSuffix" in citation:
                        suffix_text = process_inlines(citation["citationSuffix"])[
                            "text"
                        ]
                        cite_data["suffix"] = suffix_text
                    if "citationMode" in citation:
                        cite_data["mode"] = citation["citationMode"]
                    if "citationNoteNum" in citation:
                        cite_data["note_num"] = citation["citationNoteNum"]
                    if "citationHash" in citation:
                        cite_data["hash"] = citation["citationHash"]

                    result["citations"].append(cite_data)

            # Copy other metadata
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])
        elif inline_type == "Quoted":
            # Quoted text with format [quote_type, content]
            quote_type = content[0]["t"] if isinstance(content[0], dict) else content[0]
            quote_content = content[1]

            # Get quote marks based on type
            if quote_type == "DoubleQuote":
                open_mark, close_mark = '"', '"'
            elif quote_type == "SingleQuote":
                open_mark, close_mark = """, """
            else:
                open_mark, close_mark = '"', '"'  # Default to double quotes

            # Process quoted content
            nested = process_inlines(quote_content)
            result["text"] += open_mark + nested["text"] + close_mark

            # Add quote formatting
            result["formatting"].append(
                {
                    "type": "quote",
                    "quote_type": quote_type,
                    "start": start_pos,
                    "end": start_pos
                    + len(nested["text"])
                    + len(open_mark)
                    + len(close_mark),
                }
            )

            # Copy other metadata
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])
        elif inline_type == "RawInline":
            # Raw inline content with format [format, content]
            format_type = content[0]
            raw_content = content[1]

            # Add raw content to text
            result["text"] += raw_content

            # Track raw content
            result["formatting"].append(
                {
                    "type": "raw",
                    "format": format_type,
                    "start": start_pos,
                    "end": start_pos + len(raw_content),
                }
            )
        elif inline_type == "Math":
            # Math content with format [math_type, content]
            math_type = content[0]["t"] if isinstance(content[0], dict) else content[0]
            math_content = content[1]

            # Format based on math type
            if math_type == "InlineMath":
                result["text"] += f"${math_content}$"
            elif math_type == "DisplayMath":
                result["text"] += f"$${math_content}$$"
            else:
                result["text"] += math_content

            # Track math formatting
            result["formatting"].append(
                {
                    "type": "math",
                    "math_type": math_type,
                    "start": start_pos,
                    "end": start_pos + len(result["text"]) - start_pos,
                }
            )
        elif inline_type == "Span":
            # Span with format [attr, content]
            attrs = content[0]
            span_content = content[1]

            # Process span content
            nested = process_inlines(span_content)
            result["text"] += nested["text"]

            # Track span formatting
            span_attr = {
                "type": "span",
                "start": start_pos,
                "end": start_pos + len(nested["text"]),
            }

            # Add identifier and attributes
            if attrs[0]:  # Identifier
                span_attr["id"] = attrs[0]
            if attrs[1]:  # Classes
                span_attr["classes"] = attrs[1]
            if attrs[2]:  # Key-value attributes
                span_attr["attributes"] = {k: v for k, v in attrs[2]}

            result["formatting"].append(span_attr)

            # Copy other metadata
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])
        elif inline_type == "Superscript" or inline_type == "Subscript":
            nested = process_inlines(content)
            result["text"] += nested["text"]
            result["formatting"].append(
                {
                    "type": inline_type.lower(),
                    "start": start_pos,
                    "end": start_pos + len(nested["text"]),
                }
            )
            result["references"].extend(nested["references"])
            result["images"].extend(nested["images"])
            result["formatting"].extend(nested["formatting"])
            result["links"].extend(nested["links"])
            result["citations"].extend(nested["citations"])
            result["other_attributes"].extend(nested["other_attributes"])
        else:
            logger.warning(f"Unknown pandoc inline node: {inline_type}")

    return result


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
            result = []
            fp = root.origin.location

            temp_dir = tempfile.TemporaryDirectory(prefix="easyparser_")
            ast = str(Path(temp_dir.name) / "output.json")
            media = str(Path(temp_dir.name) / "media")
            subprocess.run(
                ["pandoc", "-t", "json", "--extract-media", media, fp, "-o", ast],
                check=True,
            )
            with open(ast) as f:
                data = json.load(f)
                bls = data["blocks"]

            for element in process_pandoc(bls):
                # @TODO: Convert to our structure
                result.append(element)

            output.add_group(ChunkGroup(chunks=result, root=root))
            temp_dir.cleanup()

        return output
