import io
import mimetypes
from pathlib import Path

try:
    from magika import Magika

    _m = Magika()
except ImportError:
    _m = None

try:
    import magic
except ImportError:
    magic = None


if hasattr(mimetypes, "guess_file_type"):
    _mimetypes_guess_file = mimetypes.guess_file_type
else:
    _mimetypes_guess_file = mimetypes.guess_type


def guess_mimetype(path, default: str = "application/octet-stream") -> str:
    """Guess mimetype based on file path, prioritize magika > magic > mimetypes.

    Args:
        path: the path to the file
        default: the mimetype to return if the mimetype cannot be guessed

    Returns:
        The mimetype of the file.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if p.is_dir():
        return MimeType.directory

    if ext == ".ipynb":
        # .ipynb is a JSON file that magic and magika usually mistake
        return MimeType.ipynb

    if ext == ".rst":
        # .rst is a text file that magic usually mistake
        return MimeType.rst

    if magic:
        return magic.from_file(path, mime=True)
    if _m:
        return _m.identify_path(path).output.mime_type

    guessed = _mimetypes_guess_file(path)[0]
    if guessed:
        return guessed

    return default


class MimeType:
    # Text
    text = "text/plain"
    html = "text/html"
    md = "text/markdown"
    rst = "text/x-rst"

    # Image
    jpeg = "image/jpeg"
    png = "image/png"
    tiff = "image/tiff"

    # Document
    pdf = "application/pdf"
    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    # Data interchange
    csv = "text/csv"
    json = "application/json"
    toml = "application/toml"
    yaml = "application/yaml"
    yaml_x = "application/x-yaml"

    # Sound & video
    wav = "audio/wav"
    wav_x = "audio/x-wav"
    mp3 = "audio/mpeg"
    mp4 = "video/mp4"

    # Archive
    # zip = "application/zip"
    # tar = "application/x-tar"
    epub = "application/epub+zip"
    directory = "inode/directory"

    # Code
    ipynb = "application/x-ipynb+json"


class MimeManager:
    """Helper to access and modify mime-specific behaviors"""

    def to_base64(self, chunk) -> str | None:
        """Convert to base64 string"""
        return

    def to_python(self, chunk):
        """Convert to Python, human-friendly object, e.g. PIL image"""
        if chunk.mimetype.startswith("image/"):
            from PIL import Image

            return Image.open(io.BytesIO(chunk.content))

    def metadata(self, mimetype: str) -> dict:
        """Get default chunk metadata for a mimetype

        The default metadata will be combined in a flat manner with the chunk metadata
        to ensure common information reuse. If a metadata key isn't meant to be reused
        by other operations, the convention is to prefix the key with the file type,
        or whatever you think appropriate, separated by a slash. Example: `html/key1`

        Args:
            mimetype: mimetype name

        Returns:
            dict: the default metadata for such mimetype
        """
        metadata: dict = {}

        match mimetype:
            case MimeType.pdf | MimeType.pptx:
                metadata["idx"] = 0  # page number, 0-based
                metadata["x1"] = 0  # as percentage
                metadata["x2"] = 0
                metadata["y1"] = 0
                metadata["y2"] = 0

            case mimetype if mimetype.startswith("image/"):
                metadata["x1"] = 0  # as percentage
                metadata["x2"] = 0
                metadata["y1"] = 0
                metadata["y2"] = 0

        return metadata


mime_manager = None


def get_mime_manager() -> MimeManager:
    """Get the global mime manager instance"""
    global mime_manager
    if mime_manager is None:
        mime_manager = MimeManager()
    return mime_manager


def set_mime_manager(mime_manager_instance: MimeManager) -> None:
    """Set the global mime manager instance"""
    global mime_manager
    mime_manager = mime_manager_instance
