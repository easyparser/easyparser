from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType


class TextParser(BaseOperation):
    """Parse .txt files into Chunk."""

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup) -> ChunkGroup:
        """Parse txt file into Chunk objects."""
        # Resolve chunk
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:

            with open(root.origin.location, encoding="utf-8") as fi:
                lines = fi.readlines()
                content = "".join(lines)

            # Create a text chunk as the main container
            text_chunk = Chunk(ctype=CType.Div, content=content)
            root.add_children(text_chunk)
            output.append(root)

        return output
