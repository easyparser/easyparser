import time
from concurrent.futures import ProcessPoolExecutor

import pymupdf4llm
from img2table.document import PDF
from plot import plot_blocks

from easyparser.parse.fastpdf.pdf_parser import pages_to_markdown, parition_pdf

if __name__ == "__main__":
    executor = ProcessPoolExecutor()

    # download from https://arxiv.org/pdf/1706.03762
    # or https://www.w3.org/WAI/WCAG20/Techniques/working-examples/PDF20/table.pdf

    input_path = "1706.03762.pdf"
    debug_path = "debug"

    # count the page
    doc = PDF(input_path)
    num_pages = len(doc.images)

    start_time = time.time()
    text = pymupdf4llm.to_markdown(input_path)
    end_time = time.time()
    print("PyMuPDF4LLM")
    print(f"Average time per page: {(end_time - start_time) / num_pages:.2f}s")

    start_time = time.time()
    pages = parition_pdf(input_path, executor=executor, extract_table=True)
    end_time = time.time()

    num_pages = len(pages)
    print("WIP parser")
    print(f"Average time per page: {(end_time - start_time) / num_pages:.2f}s")
    executor.shutdown()

    # preview the results
    import json

    with open("output.json", "w") as f:
        json.dump(pages, f, indent=4)
    plot_blocks(input_path, pages, debug_path)

    # start_time = time.time()
    # from unstructured.partition.pdf import partition_pdf
    # output = partition_pdf(input_path, strategy="hi_res")
    # end_time = time.time()
    # print("Unstructured")
    # print(f"Average time per page: {(end_time - start_time) / num_pages:.2f}s")

    md_text = pages_to_markdown(pages)
    with open("output.md", "w") as f:
        f.write(md_text)
    print("Exported to output.md")
