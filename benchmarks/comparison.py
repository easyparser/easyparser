import time
from pathlib import Path

start = time.time()
from unstructured.partition.pdf import partition_pdf
print(f"Import unstructured {time.time() - start}")

start = time.time()
from easyparser.load.pdf import pdf_by_pymupdf
from easyparser.chunk.text import chunk_by_characters
print(f"Import easyparser {time.time() - start}")



if __name__ == "__main__":
    f = Path(__file__).parent.parent / "tests" / "assets" / "test.pdf"

    # easyparser
    start = time.time()
    text = pdf_by_pymupdf(f)[0].text
    chunks = chunk_by_characters(
        text, chunk_size=500, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    print(f"Time [easyparser]: {time.time() - start} - # chunks: {len(chunks)}")

    # unstructured
    start = time.time()
    result = partition_pdf(str(f), strategy="fast", chunking_strategy="basic")
    print(f"Time [Unstructured]: {time.time() - start} - # chunks: {len(result)}")
