import csv
import io
import logging
import xml.etree.ElementTree as ET
import zipfile

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType

logger = logging.getLogger(__name__)


def convert_cell(cell):
    """Convert from Openpyxl cell into Python built-in types"""
    from openpyxl.cell.cell import TYPE_ERROR, TYPE_NUMERIC, TYPE_STRING

    if cell.value is None:
        return ""
    elif cell.data_type == TYPE_STRING:
        return cell.value.replace("\n", "\\n")
    elif cell.data_type == TYPE_NUMERIC:
        val = int(cell.value)
        if val == cell.value:
            return val
        return float(cell.value)
    elif cell.data_type == TYPE_ERROR:
        return ""

    return cell.value


def parse_drawing_texts(xlsx_path, drawing_path) -> list[Chunk]:
    """Parse drawing relationships by directly accessing the XML files"""

    # Open the Excel file as a zip
    with zipfile.ZipFile(xlsx_path, "r") as zip_ref:
        # Get the drawing file content
        drawing_xml = zip_ref.read(drawing_path).decode("utf-8")
        root = ET.fromstring(drawing_xml)

        # Find all shape elements (oneCellAnchor, twoCellAnchor, absoluteAnchor)
        ns = {
            "xdr": (
                "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
            ),
            "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
            "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        }
        shapes = (
            root.findall("./xdr:twoCellAnchor", ns)
            + root.findall("./xdr:oneCellAnchor", ns)
            + root.findall("./xdr:absoluteAnchor", ns)
        )

        # Parse texts
        chunks = []
        for shape in shapes:
            sp = shape.find(".//xdr:sp", ns)
            if sp is None:
                continue

            txBody = sp.find(".//xdr:txBody", ns)
            if txBody is None:
                continue
            text_parts = []
            for p in txBody.findall(".//a:p", ns):
                ts = [t.text for t in p.findall(".//a:t", ns) if t.text]
                text = " ".join(ts).strip()
                if text:
                    text_parts.append(text)

            if text_parts:
                chunk = Chunk(
                    mimetype="plain/text",
                    ctype=CType.Para,
                    content="\n\n".join(text_parts),
                )
                chunks.append(chunk)

        return chunks


class XlsxOpenpyxlParser(BaseOperation):

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup) -> ChunkGroup:
        """Parse xlsx file into Chunk objects using openpyxl.

        SUPPORTED FORMATS: xlsx, xlsm, xltx, xltm.

        Each sheet in the xlsx file corresponds to a div object element. Within the div
        there can be:
            - Tables. A table consists of rows. A row consists of cells.
            - Non-cell elements.

        This method only extracts calculated values from the xlsx file. It does not
        extract formulas.
        """
        import openpyxl
        from openpyxl.drawing.spreadsheet_drawing import SpreadsheetDrawing

        # Resolve chunk
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            workbook = openpyxl.load_workbook(root.origin.location)
            sheet_chunks = []
            for sheet in workbook.worksheets:

                # Create a new chunk for the sheet
                sheet_chunk = Chunk(
                    mimetype="plain/text",
                    ctype=CType.Header,
                    content=sheet.title,
                )
                children = []

                # Read cells
                if workbook.read_only:
                    sheet.reset_dimensions()

                data = []
                first_row_with_data = -1
                last_row_with_data = -1
                for row_number, row in enumerate(sheet.rows):
                    converted_row = [convert_cell(cell) for cell in row]
                    while converted_row and converted_row[-1] == "":
                        converted_row.pop()
                    if converted_row:
                        if first_row_with_data == -1:
                            first_row_with_data = row_number
                        else:
                            last_row_with_data = row_number
                    data.append(converted_row)

                # Trim trailing empty rows
                data = data[first_row_with_data : last_row_with_data + 1]

                # Get tables
                if len(data) > 0:
                    # Extend rows to max width
                    max_width = max(len(data_row) for data_row in data)
                    if min(len(data_row) for data_row in data) < max_width:
                        empty_cell = [""]
                        data = [
                            data_row + (max_width - len(data_row)) * empty_cell
                            for data_row in data
                        ]

                    # Create table chunk
                    table_chunk = Chunk(
                        mimetype="plain/text",
                        ctype=CType.Table,
                        content="",
                        parent=sheet_chunk,
                        metadata={
                            "merged_cells": [],
                            "first_row_with_data": first_row_with_data,
                            "last_row_with_data": last_row_with_data,
                        },
                    )

                    # Construct rows
                    rows = []
                    string_io = io.StringIO()
                    csv_writer = csv.writer(
                        string_io,
                        quoting=csv.QUOTE_MINIMAL,
                        escapechar="\\",
                        doublequote=True,
                    )
                    for row_number, row in enumerate(data):
                        # Create a new row chunk
                        csv_writer.writerow(row)
                        rows.append(
                            Chunk(
                                mimetype="plain/text",
                                ctype=CType.TableRow,
                                content=string_io.getvalue().strip(),
                                parent=table_chunk,
                                metadata={"row_number": row_number},
                            )
                        )
                        string_io.seek(0)
                        string_io.truncate(0)
                    table_chunk.add_children(rows)

                    # Gather merged rows, columns
                    merged_cells = []
                    for merged_cell in sheet.merged_cells.ranges:
                        start_col, start_row, end_col, end_row = merged_cell.bounds
                        # 0-indexed
                        start_row, end_row = (start_row - 1, end_row - 1)
                        start_col, end_col = (start_col - 1, end_col - 1)

                        if start_row > last_row_with_data:
                            continue
                        if start_col > max_width:
                            continue
                        if max_width == 0:
                            continue

                        start_row -= first_row_with_data
                        end_row = min(end_row - first_row_with_data, len(data))
                        start_col = min(start_col, max_width - 1)
                        end_col = min(end_col, max_width - 1)
                        merged_cells.append((start_row, start_col, end_row, end_col))
                    table_chunk.metadata["merged_cells"] = merged_cells

                    children.append(table_chunk)

                # Get drawing images
                for img in sheet._images:
                    image_chunk = Chunk(
                        mimetype=f"image/{img.format.lower()}",
                        ctype=CType.Figure,
                        content=img._data(),
                        parent=sheet_chunk,
                    )
                    children.append(image_chunk)

                drawings = sheet._rels.find(SpreadsheetDrawing._rel_type)
                for drawing in drawings:
                    tx_chs = parse_drawing_texts(root.origin.location, drawing.target)
                    children.extend(tx_chs)

                # Add the root chunk to output
                sheet_chunk.add_children(children)
                sheet_chunks.append(sheet_chunk)

            root.add_children(sheet_chunks)
            output.append(root)

        return output

    @classmethod
    def py_dependency(cls) -> list[str]:
        """Return the python dependency for this operation."""
        return ["openpyxl"]
