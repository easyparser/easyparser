import csv
import io
import logging

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType

logger = logging.getLogger(__name__)


class CsvParser(BaseOperation):
    """Parse CSV files into Chunk objects using Python's built-in csv module."""

    @classmethod
    def run(cls, chunks: Chunk | ChunkGroup) -> ChunkGroup:
        """Parse CSV file into Chunk objects.

        Each CSV file corresponds to a table object element.
        The table consists of rows, and each row consists of cells.
        """
        # Resolve chunk
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            logger.info(f"Parsing {root.origin.location}")
            # Create a table chunk as the main container
            table_chunk = Chunk(
                mimetype="plain/text",
                ctype=CType.Table,
                content="",
                metadata={
                    "merged_cells": [],  # CSV doesn't support merged cells
                    "first_row_with_data": 0,
                    "last_row_with_data": 0,  # Will be updated after processing
                },
            )

            # Read the CSV file
            with open(root.origin.location, newline="", encoding="utf-8-sig") as fi:
                # Try to detect the dialect
                try:
                    dialect = csv.Sniffer().sniff(fi.read(1024))
                    fi.seek(0)
                    csv_reader = csv.reader(fi, dialect)
                except csv.Error:
                    # Fall back to default dialect
                    fi.seek(0)
                    csv_reader = csv.reader(fi)

                # Process rows
                rows = []
                row_count = 0
                for row_number, row in enumerate(csv_reader):
                    # Create a string representation of the row
                    string_io = io.StringIO()
                    csv_writer = csv.writer(
                        string_io,
                        quoting=csv.QUOTE_MINIMAL,
                        escapechar="\\",
                        doublequote=True,
                    )
                    csv_writer.writerow(row)

                    # Create a new row chunk
                    row_chunk = Chunk(
                        mimetype="plain/text",
                        ctype=CType.TableRow,
                        content=string_io.getvalue().strip(),
                        parent=table_chunk,
                        metadata={"row_number": row_number},
                    )
                    rows.append(row_chunk)
                    row_count += 1

            # Update the table metadata with the last row index
            table_chunk.metadata["last_row_with_data"] = row_count - 1

            # Add rows to the table
            table_chunk.add_children(rows)

            # Add the table to the root
            root.add_children([table_chunk])
            output.append(root)

        return output
