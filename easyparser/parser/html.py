import logging
import tempfile
from copy import copy
from pathlib import Path

from easyparser.base import BaseOperation, Chunk, ChunkGroup

from .pandoc_engine import PandocEngine

logger = logging.getLogger(__name__)


def preprocess_html(input_file, output_file):
    """Preprocess HTML file to make it more suitable for Pandoc conversion"""
    from bs4 import BeautifulSoup, Comment
    from bs4.element import NavigableString

    # Read the HTML file
    with open(input_file, encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove hidden elements (display: none or visibility: hidden)
    hidden_elements_removed = 0
    for tag in soup.find_all(True):
        if tag.has_attr("style"):
            style = tag["style"]

            # Parse the style attribute
            try:
                style_dict = {}
                for item in style.split(";"):
                    if ":" in item:
                        prop, val = item.split(":", 1)
                        style_dict[prop.strip().lower()] = val.strip().lower()

                # Check for display: none or visibility: hidden
                if "display" in style_dict and style_dict["display"] == "none":
                    tag.extract()
                    hidden_elements_removed += 1
                    continue

                if "visibility" in style_dict and style_dict["visibility"] == "hidden":
                    tag.extract()
                    hidden_elements_removed += 1
                    continue
            except Exception as e:
                logger.warning(f"Error parsing style attribute: {e}")

        # Check for hidden attribute
        if tag.has_attr("hidden"):
            tag.extract()
            hidden_elements_removed += 1
            continue

    # Remove empty elements and elements without content
    empty_elements_removed = 0
    text_elements = [
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "span",
        "div",
        "li",
        "caption",
        "label",
    ]

    for tag_name in text_elements:
        for tag in soup.find_all(tag_name):
            # Check if the element has any text content or meaningful children
            if tag.get_text(strip=True):
                continue
            if tag.find_all(["img", "svg", "canvas", "video", "audio", "iframe", "br"]):
                continue
            tag.extract()
            empty_elements_removed += 1

    # Remove table row that has all empty cells
    for tr in soup.find_all("tr"):
        if tr.get_text(strip=True):
            continue
        if tr.find_all(["img", "svg", "canvas", "video", "audio", "iframe", "br"]):
            continue
        tr.extract()
        empty_elements_removed += 1

    # Remove scripts, style tags and unnecessary metadata
    for tag in soup.find_all(["script", "style", "meta", "link"]):
        tag.extract()

    # Simplify table
    tables_unpacked = 0
    tables_to_lists = 0
    tables_removed = 0
    tables = soup.find_all("table")

    for table in tables:
        rows = table.find_all("tr")

        # Skip completely empty tables
        if not rows:
            table.extract()
            tables_removed += 1
            continue

        # Table with single cell containing text - unpack the content
        if len(rows) == 1:
            cells = rows[0].find_all(["td", "th"])
            if len(cells) == 1 and cells[0].get_text(strip=True):
                # Create a div to replace the table with the cell content
                content_div = soup.new_tag("div")
                # Handle both Tag and NavigableString children
                for child in cells[0].children:
                    if isinstance(child, NavigableString):
                        content_div.append(NavigableString(child))
                    else:
                        content_div.append(copy(child))
                table.replace_with(content_div)
                tables_unpacked += 1
                continue

        # Strip style attributes but keep the table structure
        for tag in table.find_all(True):
            if tag.has_attr("style"):
                del tag["style"]

        # Add proper headers if missing
        if rows[0].find("th"):
            # Check if first row is likely a header row
            first_row = rows[0]
            if first_row.find_all("td"):
                for td in first_row.find_all("td"):
                    # Convert td to th if it appears to be a header
                    if td.get("style") and (
                        "bold" in td.get("style") or "center" in td.get("style")
                    ):
                        new_th = soup.new_tag("th")
                        new_th.string = td.get_text(strip=True)
                        td.replace_with(new_th)

        # Simplify nested elements in table cells
        for cell in table.find_all(["td", "th"]):
            # Get the text content
            text = cell.get_text(strip=True)
            if text:
                cell.clear()
                cell.string = text

    logger.debug(f"Removed {tables_removed} empty tables")
    logger.debug(f"Unpacked {tables_unpacked} single-cell tables")
    logger.debug(f"Converted {tables_to_lists} single-column tables to lists")

    # Clean up spans and divs
    for tag in soup.find_all(["span", "div"]):
        # Replace with its contents if it only contains text or simple elements
        if len(tag.find_all(["div", "table", "ul", "ol"])) == 0:
            tag.replace_with_children()

    # Replace multiple consecutive <br> tags with a single one
    for br in soup.find_all("br"):
        next_siblings = list(br.next_siblings)
        if next_siblings and next_siblings[0].name == "br":
            next_siblings[0].extract()

    # Write the processed HTML to the output file
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(soup))

    return output_file


class PandocHtmlParser(BaseOperation):
    @classmethod
    def run(cls, chunk: Chunk | ChunkGroup, **kwargs) -> ChunkGroup:
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])

        output = ChunkGroup()
        for root in chunk:
            logger.info(f"Parsing {root.origin.location}")
            temp_file_path = tempfile.mktemp(suffix=".html")
            try:
                # Preprocess the HTML file
                original_file_path = root.origin.location
                preprocess_html(original_file_path, temp_file_path)
                root.origin.location = temp_file_path

                # Convert HTML to Markdown using Pandoc
                pandoc_output = PandocEngine.run(root)
                root.origin.location = original_file_path
                output.append(pandoc_output[0])
            finally:
                # Clean up temporary file
                Path(temp_file_path).unlink(missing_ok=True)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        return ["beautifulsoup4", "pypandoc"]
