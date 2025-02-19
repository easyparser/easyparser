import time
from pathlib import Path

from unstructured.partition.pdf import partition_pdf

from easyparser.load.pdf import PyMuPDFLoader
from easyparser.chunk.text import by_characters



if __name__ == "__main__":
    f = Path(__file__).parent.parent / "tests" / "assets" / "test.pdf"

    # unstructured
    start = time.time()
    result = partition_pdf(str(f), strategy="fast", chunking_strategy="basic")
    print(f"Time [Unstructured]: {time.time() - start} - # chunks: {len(result)}")

    # easyparser
    start = time.time()
    loader = PyMuPDFLoader()
    text = loader.load(f)[0].text
    chunks = by_characters(
        text, chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    print(f"Time [easyparser]: {time.time() - start} - # chunks: {len(chunks)}")

