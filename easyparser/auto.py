import logging
import os
from pathlib import Path
from typing import Callable

from easyparser.base import Chunk
from easyparser.controller import get_controller
from easyparser.mime import MimeType

logger = logging.getLogger(__name__)


def parse_as_graph(
    path: str | Path,
    extras: dict[str, list] | None = None,
    callbacks: list[Callable] | None = None,
) -> Chunk:
    """Parse a file or directory into chunks

    Args:
        path: the path to file or directory
        extras: a dictionary mapping mimetype to list of parsers to use for that
            mimetype
        callbacks: a list of callback functions, where each function takes in a
            path and a mimetype, and returns a single parser to use, or return None
            if no parser is found
    """
    ctrl = get_controller()
    with ctrl.temporary(extras=extras, callbacks=callbacks):
        # Parse the path into chunk
        chunk = ctrl.as_root_chunk(path)

        # Attempt to parse the chunk
        attempted = 0
        for parser in ctrl.iter_parser(path):
            attempted += 1
            try:
                parser.run(chunk)
                break
            except Exception as e:
                chunk = ctrl.as_root_chunk(path)
                logger.warning(f"Parser {parser} failed for {path}: {e}")
                continue
        else:
            if attempted == 0:
                # No parser found
                logger.warning(f"No parser found for {path}. Skipping.")

    return chunk


def parse(
    path: str | Path,
    skip_hidden: bool = True,
    extras: dict[str, list] | None = None,
    callbacks: list[Callable] | None = None,
) -> list[Chunk]:
    """Parse a directory into chunks, where each file is a chunk

    Args:
        path: the path to the directory
        skip_hidden: whether to skip hidden files and directories
        extras: a dictionary mapping mimetype to list of parsers to use for that
        callbacks: a list of callback functions, where each function takes in a
            path and a mimetype, and returns a single parser to use, or return None
            if no parser is found

    Returns:
        A list of chunks, where each chunk is a file in the directory
    """
    path = Path(path)
    if skip_hidden and path.name.startswith("."):
        logger.debug(f"Skipping hidden file/directory: {path}")
        return []

    if path.is_file():
        return [parse_as_graph(path, extras=extras, callbacks=callbacks)]

    # Don't process directories
    ctrl = get_controller()
    ctrl._parsers.pop(MimeType.directory, None)

    result = []
    with ctrl.temporary(extras=extras, callbacks=callbacks):
        # Parse the path into chunk
        for root, dirs, files in os.walk(path):
            for each_dir in list(sorted(dirs)):
                if skip_hidden and each_dir.startswith("."):
                    # Don't traverse hidden directories
                    logger.debug(f"Skipping hidden directory: {each_dir}")
                    dirs.remove(each_dir)
                    continue

                dir_path = os.path.join(root, each_dir)
                parser = list(ctrl.iter_parser(dir_path))
                if not parser:
                    continue

                # A directory might be processed in case `extras` and/or `callbacks`
                # decide to process it
                chunk = ctrl.as_root_chunk(dir_path)
                for p in parser:
                    try:
                        p.run(chunk)
                        result.append(chunk)
                        dirs.remove(each_dir)  # Don't process this directory again
                        break
                    except Exception as e:
                        chunk = ctrl.as_root_chunk(dir_path)
                        logger.warning(f"Parser {p} failed for {dir_path}: {e}")
                        continue

            # Parse each file in the directory
            for each_file in sorted(files):
                if skip_hidden and each_file.startswith("."):
                    # Don't process hidden files
                    logger.debug(f"Skipping hidden file: {each_file}")
                    continue

                file_path = os.path.join(root, each_file)
                chunk = ctrl.as_root_chunk(file_path)

                # Attempt to parse the chunk
                for parser in ctrl.iter_parser(file_path):
                    try:
                        parser.run(chunk)
                        break
                    except Exception as e:
                        chunk = ctrl.as_root_chunk(file_path)
                        logger.warning(f"Parser {parser} failed for {file_path}: {e}")
                        continue

                # Add the chunk to the result
                result.append(chunk)

    return result
