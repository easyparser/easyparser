import logging
from pathlib import Path
from typing import Callable

from easyparser.base import Chunk
from easyparser.controller import get_controller

logger = logging.getLogger(__name__)


def parse(
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
            except Exception as e:
                logger.warning(f"Parser {parser} failed for {path}: {e}")
                continue
            break
        else:
            if attempted == 0:
                # No parser found
                logger.warning(f"No parser found for {path}. Skipping.")

    return chunk
