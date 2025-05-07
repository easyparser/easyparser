import hashlib
import logging
import mimetypes
from contextlib import contextmanager
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


def guess_mimetype(path, default: str = "application/octet-stream") -> str:
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


class Controller:
    """Coordinator to parse from raw "binary" into chunk

    Functions:
        - Lazily load all the possible parsers
        - Iterate over the parsers
    """

    def __init__(
        self, extras: dict[str, list] | None = None, callbacks: list | None = None
    ):
        self._extras = extras or {}
        self._callbacks = callbacks or []
        self._parsers: dict[str, list] = self._load_parsers()

        self._temp_extras = []
        self._temp_callbacks = []

    def _load_parsers(self) -> dict[str, list]:
        from easyparser.parser.audio import AudioWhisperParser
        from easyparser.parser.dict_list import JsonParser, TomlParser, YamlParser
        from easyparser.parser.directory import DirectoryParser
        from easyparser.parser.html import PandocHtmlParser
        from easyparser.parser.image import RapidOCRImageText
        from easyparser.parser.md import Markdown
        from easyparser.parser.pandoc_engine import PandocEngine
        from easyparser.parser.pdf import FastPDF
        from easyparser.parser.pptx import PptxParser
        from easyparser.parser.text import TextParser
        from easyparser.parser.video import VideoWhisperParser
        from easyparser.parser.xlsx import XlsxOpenpyxlParser

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
    ) -> Generator["Controller", None, None]:
        """For a given file or folder, iterate over eligible parsers

        There can be multiple parsers for a given file type, so if 1 parser fails,
        the next one will be tried.
        """
        if not mimetype:
            if not path:
                raise ValueError("Either mimetype or path must be provided.")
            mimetype = self.guess_mimetype(path)

        _miss = True

        # Prioritize the temporary callbacks and extras
        if self._temp_callbacks:
            for _temp_callbacks in self._temp_callbacks:
                for callback in _temp_callbacks:
                    matched = callback(path, mimetype)
                    if matched:
                        _miss = False
                        yield matched

        if self._temp_extras:
            for _temp_extras in self._temp_extras:
                if mimetype in _temp_extras:
                    _miss = False
                    yield from _temp_extras[mimetype]

        for callback in self._callbacks:
            matched = callback(path, mimetype)
            if matched:
                _miss = False
                yield matched

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
        return guess_mimetype(path, default)

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

    def register(
        self, extras: dict[str, list] | None = None, callbacks: list | None = None
    ) -> None:
        """Add extra parsers to the controller"""
        if extras:
            for key, value in extras.items():
                if key in self._extras:
                    self._extras[key].extend(value)
                else:
                    self._extras[key] = value

        if callbacks:
            self._callbacks.extend(callbacks)

    @contextmanager
    def temporary(
        self, extras: dict[str, list] | None = None, callbacks: list | None = None
    ):
        """Temporarily add extra parsers to the controller"""
        if extras:
            self._temp_extras.append(extras)
        if callbacks:
            self._temp_callbacks.append(callbacks)

        yield

        if extras:
            self._temp_extras.remove(extras)
        if callbacks:
            self._temp_callbacks.remove(callbacks)


_ctrl = None


def get_controller() -> Controller:
    """Get the controller instance"""
    global _ctrl
    if _ctrl is None:
        _ctrl = Controller()
    return _ctrl


def set_controller(ctrl: Controller) -> None:
    """Set the controller instance"""
    global _ctrl
    _ctrl = ctrl
