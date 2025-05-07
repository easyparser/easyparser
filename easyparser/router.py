import hashlib
import logging
import mimetypes
from pathlib import Path
from typing import Generator

try:
    from magika import Magika

    _m = Magika()
except ImportError:
    _m = None

try:
    import magic
except ImportError:
    magic = None

from easyparser.base import Chunk, CType, Origin

if hasattr(mimetypes, "guess_file_type"):
    _mimetypes_guess_file = mimetypes.guess_file_type
else:
    _mimetypes_guess_file = mimetypes.guess_type


logger = logging.getLogger(__name__)


class MimeType:
    # Text
    text = "text/plain"
    html = "text/html"
    md = "text/markdown"

    # Image
    jpeg = "image/jpeg"
    png = "image/png"

    # Document
    pdf = "application/pdf"
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    # Data interchange
    json = "application/json"
    toml = "application/toml"
    yaml = "application/yaml"
    yaml_x = "application/x-yaml"

    # Sound & video
    wav = "audio/wav"
    mp3 = "audio/mpeg"
    mp4 = "audio/mp4"

    # Archive
    # zip = "application/zip"
    # tar = "application/x-tar"
    epub = "application/epub+zip"
    directory = "inode/directory"


class FileCoordinator:
    """Coordinator to parse from raw "binary" into chunk

    Functions:
        - Lazily load all the possible parsers
        - Iterate over the parsers
    """

    def __init__(self, **extras):
        self._extras = extras
        self._parsers: dict[str, list] = self._load_parsers()

    def _load_parsers(self) -> dict[str, list]:
        from easyparser.parse.audio import AudioWhisperParser
        from easyparser.parse.dict_list import JsonParser, TomlParser, YamlParser
        from easyparser.parse.directory import DirectoryParser
        from easyparser.parse.html import PandocHtmlParser
        from easyparser.parse.image import RapidOCRImageText
        from easyparser.parse.md import Markdown
        from easyparser.parse.pandoc_engine import PandocEngine
        from easyparser.parse.pdf import FastPDF
        from easyparser.parse.pptx import PptxParser
        from easyparser.parse.text import TextParser
        from easyparser.parse.video import VideoWhisperParser
        from easyparser.parse.xlsx import XlsxOpenpyxlParser

        return {
            MimeType.text: [TextParser],
            MimeType.html: [PandocHtmlParser, PandocEngine, TextParser],
            MimeType.md: [Markdown, TextParser],
            MimeType.jpeg: [RapidOCRImageText],
            MimeType.png: [RapidOCRImageText],
            MimeType.pdf: [FastPDF],
            MimeType.docx: [PandocEngine],
            MimeType.pptx: [PptxParser],
            MimeType.xlsx: [XlsxOpenpyxlParser],
            MimeType.json: [JsonParser],
            MimeType.toml: [TomlParser],
            MimeType.yaml: [YamlParser],
            MimeType.yaml_x: [YamlParser],
            MimeType.wav: [AudioWhisperParser],
            MimeType.mp3: [AudioWhisperParser],
            MimeType.mp4: [VideoWhisperParser],
            MimeType.epub: [PandocEngine],
            MimeType.directory: [DirectoryParser],
        }

    def iter_parser(
        self,
        path: str | Path | None = None,
        mimetype: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> Generator[dict, None, None]:
        """For a given file or folder, iterate over eligible parsers

        There can be multiple parsers for a given file type, so if 1 parser fails,
        the next one will be tried.
        """
        if not mimetype:
            if not path:
                raise ValueError("Either mimetype or path must be provided.")
            mimetype = self.guess_mimetype(path)

        _miss = True
        if mimetype in self._extras:
            _miss = False
            yield from self._extras[mimetype]

        if mimetype in self._parsers:
            _miss = False
            yield from self._parsers[mimetype]

        if _miss:
            message = (
                f"Unsupported mimetype: {mimetype}. "
                "Please register in **extras, or make a Github issue"
            )
            if strict:
                raise ValueError(message)
            logger.warning(message)

    def guess_mimetype(self, path, default: str = "application/octet-stream") -> str:
        """Guess mimetype based on file path, prioritize magika > magic > mimetypes.

        Args:
            path: the path to the file
            default: the mimetype to return if the mimetype cannot be guessed

        Returns:
            The mimetype of the file.
        """
        if Path(path).is_dir():
            return MimeType.directory

        if _m:
            return _m.identify_path(path).output.mime_type
        if magic:
            return magic.from_file(path, mime=True)

        guessed = _mimetypes_guess_file(path)[0]
        if guessed:
            return guessed

        return default

    def as_root_chunk(self, path: str | Path, mimetype: str | None = None) -> Chunk:
        """Convert a file or directory to a chunk."""
        path_str = str(path)
        path = Path(path_str).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File/Directory not found: {path_str}")

        if path.is_file():
            if mimetype is None:
                mimetype = self.guess_mimetype(path)

            with open(path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            chunk = Chunk(
                ctype=CType.Div,
                mimetype=mimetype,
                origin=Origin(location=path_str, protocol="file"),
            )
            chunk.id = file_hash

            return chunk

        else:
            chunk = Chunk(
                ctype=CType.Div,
                origin=Origin(location=path_str, protocol="directory"),
            )
            chunk.id = f"dir_{chunk.id}"
            return chunk


_coordinators = []


def register_coordinator(coordinator: FileCoordinator) -> FileCoordinator:
    """Register a router."""
    global _coordinators

    _coordinators.append(coordinator)
    return coordinator


def get_coordinators() -> list:
    global _coordinators

    if not _coordinators:
        _coordinators = [FileCoordinator()]

    return _coordinators
