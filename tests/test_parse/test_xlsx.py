from pathlib import Path

from easyparser.base import Chunk
from easyparser.mime import mime_xlsx
from easyparser.parse.xlsx import XlsxOpenpyxlParser

asset_folder = Path(__file__).parent.parent / "assets"
multi_sheets = str(asset_folder / "multi_sheets.xlsx")
drawing_text_image = str(asset_folder / "drawing_text_image.xlsx")


def test_multi_sheet():
    root = mime_xlsx.as_root_chunk(multi_sheets)
    chunks = XlsxOpenpyxlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_drawing_text_image():
    root = mime_xlsx.as_root_chunk(drawing_text_image)
    chunks = XlsxOpenpyxlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
