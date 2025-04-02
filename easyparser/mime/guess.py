import mimetypes

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
    """Guess the mimetype of a file based on its path.

    Args:
        path: the path to the file
        default: the mimetype to return if the mimetype cannot be guessed

    Returns:
        The mimetype of the file.
    """
    if _m:
        return _m.identify_path(path).output.mime_type
    if magic:
        return magic.from_file(path, mime=True)

    guessed = _mimetypes_guess_file(path)[0]
    if guessed:
        return guessed

    return default
