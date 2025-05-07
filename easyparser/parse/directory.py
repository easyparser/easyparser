import logging
from pathlib import Path

from easyparser.base import BaseOperation, Chunk, ChunkGroup
from easyparser.router import get_coordinators

logger = logging.getLogger(__name__)


class DirectoryParser(BaseOperation):
    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        ctrls: list | None = None,
        **kwargs,
    ) -> ChunkGroup:
        """Parse a directory recursively.

        Args:
            chunks: the chunk or chunk group to process
            ctrls: list of coordinators to use for parsing
        """
        # Resolve chunk
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            path = Path(root.origin.location)
            if not path.is_dir():
                raise ValueError(f"Path {root.origin.location} is not a directory.")

            if not ctrls:
                ctrls = get_coordinators()

            children = []
            for _p in path.glob("*"):
                if _p.is_dir():
                    if _p.name == ".git":
                        # Skip .git directories
                        continue

                child = ctrls[-1].as_root_chunk(_p)
                attempted = 0
                for parser in ctrls[-1].iter_parser(_p):
                    attempted += 1
                    try:
                        parser.run(child)
                    except Exception as e:
                        logger.warning(f"Parser {parser} failed for {_p}: {e}")
                        continue
                    break
                else:
                    if attempted == 0:
                        # No parser found
                        logger.warning(f"No parser found for {_p}. Skipping.")

                children.append(child)

            root.add_children(children)
            output.append(root)

        return output
