from pathlib import Path

from easyparser.base import Chunk, ChunkGroup
from easyparser.controller import Controller
from easyparser.parser.audio import AudioWhisperParser
from easyparser.parser.csv import CsvParser
from easyparser.parser.dict_list import JsonParser, TomlParser, YamlParser
from easyparser.parser.html import PandocHtmlParser
from easyparser.parser.image import RapidOCRImageText
from easyparser.parser.md import Markdown
from easyparser.parser.pandoc_engine import PandocEngine
from easyparser.parser.pdf import DoclingPDF, FastPDF, SycamorePDF, UnstructuredPDF
from easyparser.parser.pptx import PptxParser
from easyparser.parser.text import TextParser
from easyparser.parser.video import VideoWhisperParser
from easyparser.parser.xlsx import XlsxOpenpyxlParser

asset_folder = Path(__file__).parent / "assets"

pdf_path1 = str(asset_folder / "short.pdf")
pdf_path2 = str(asset_folder / "short_image.pdf")
jpg_path1 = str(asset_folder / "with_table.jpg")
docx_path = str(asset_folder / "with_image.docx")
pptx_path = str(asset_folder / "normal.pptx")
pptx_short = str(asset_folder / "short_image.pptx")
multi_sheets = str(asset_folder / "multi_sheets.xlsx")
drawing_text_image = str(asset_folder / "drawing_text_image.xlsx")
csv_path = str(asset_folder / "contains_empty_cell.csv")
json_path = str(asset_folder / "long.json")
toml_path = str(asset_folder / "long.toml")
yaml_path = str(asset_folder / "long.yaml")
mp3_path = str(asset_folder / "jfk_apollo_49.mp3")
wav_path = str(asset_folder / "jfk_apollo_49.wav")
epub_path = str(asset_folder / "long.epub")
html_path = str(asset_folder / "long.html")
md_path = str(asset_folder / "lz.md")
txt_path = str(asset_folder / "long.txt")
mp4_path = str(asset_folder / "jfk_30.mp4")

ctrl = Controller()


def test_sycamore():
    root = ctrl.as_root_chunk(pdf_path1)
    chunks = SycamorePDF.run(root, use_ocr=False)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_unstructured():
    root = ctrl.as_root_chunk(pdf_path1)
    chunks = UnstructuredPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_docling():
    root = ctrl.as_root_chunk(pdf_path1)
    chunks = DoclingPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_fastpdf():
    root = ctrl.as_root_chunk(pdf_path1)
    chunks = FastPDF.run(root)
    assert len(chunks) > 0
    assert root.id in chunks.groups
    assert isinstance(chunks[0], Chunk)


def test_sycamore_multiple():
    root1 = ctrl.as_root_chunk(pdf_path1)
    root2 = ctrl.as_root_chunk(pdf_path2)
    chunks = SycamorePDF.run(ChunkGroup(chunks=[root1, root2]))
    assert root1.id in chunks.groups
    assert root2.id in chunks.groups
    assert len(chunks.groups[root1.id]) > 0
    assert len(chunks.groups[root2.id]) > 0


def test_jpg():
    root = ctrl.as_root_chunk(jpg_path1)
    chunks = RapidOCRImageText.run(root)
    chunk = chunks[0]
    chunk.print_graph()
    assert isinstance(chunk, Chunk)


def test_pandoc_docx():
    root = ctrl.as_root_chunk(docx_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_pptx_fast():
    root = ctrl.as_root_chunk(pptx_path)
    chunks = PptxParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_pptx_image():
    root = ctrl.as_root_chunk(pptx_short)
    chunks = PptxParser.run(root, caption=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_multi_sheet():
    root = ctrl.as_root_chunk(multi_sheets)
    chunks = XlsxOpenpyxlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_drawing_text_image():
    root = ctrl.as_root_chunk(drawing_text_image)
    chunks = XlsxOpenpyxlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_csv():
    root = ctrl.as_root_chunk(csv_path)
    chunks = CsvParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_parse_json():
    root = ctrl.as_root_chunk(json_path)
    chunks = JsonParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)


def test_parse_toml():
    root = ctrl.as_root_chunk(toml_path)
    chunks = TomlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)


def test_parse_yaml():
    root = ctrl.as_root_chunk(yaml_path)
    chunks = YamlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
    assert isinstance(chunk.content, str)


def test_mp3():
    root = ctrl.as_root_chunk(mp3_path)
    chunks = AudioWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_wav():
    root = ctrl.as_root_chunk(wav_path)
    chunks = AudioWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_pandoc_epub():
    root = ctrl.as_root_chunk(epub_path)
    chunks = PandocEngine.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_pandoc_html():
    root = ctrl.as_root_chunk(html_path)
    chunks = PandocHtmlParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_md():
    root = ctrl.as_root_chunk(md_path)
    chunks = Markdown.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_txt():
    root = ctrl.as_root_chunk(txt_path)
    chunks = TextParser.run(root)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)


def test_mp4():
    root = ctrl.as_root_chunk(mp4_path)
    chunks = VideoWhisperParser.run(root, include_segments=True)
    chunk = chunks[0]
    assert isinstance(chunk, Chunk)
