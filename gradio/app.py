import functools
import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import pymupdf

import gradio as gr
from easyparser.base import CType
from easyparser.controller import Controller
from easyparser.parser import DoclingPDF, FastPDF, SycamorePDF, UnstructuredPDF
from easyparser.parser.fastpdf.util import OCRMode, bytes_to_base64
from easyparser.split.toc_builder import TOCHierarchyBuilder
from easyparser.util.plot import plot_img, plot_pdf

MAX_PAGES = os.getenv("MAX_PAGES", 10)
MAX_PAGES_CHUNKING = os.getenv("MAX_PAGES", 50)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
METHOD_MAP = {
    "easyparser_fastpdf": FastPDF,
    "unstructured": UnstructuredPDF,
    "docling": DoclingPDF,
    "sycamore": SycamorePDF,
}
METHOD_LIST = list(METHOD_MAP.keys())
TMP_DIR = Path("/tmp/visualize")
TMP_DIR.mkdir(exist_ok=True)
ctrl = Controller()

DEBUG_DIR = Path("debug")
PDF_JS_DIR = "pdfjs-4.0.379-dist"

assert Path(PDF_JS_DIR).exists(), (
    f"Missing `{PDF_JS_DIR}` directory. Please download it from"
    "https://github.com/mozilla/pdf.js/releases"
    "/download/v4.0.379/pdfjs-4.0.379-dist.zip"
)

HEAD_HTML = """
<link href='https://fonts.googleapis.com/css?family=PT Mono' rel='stylesheet'>
<script type='module' src='https://cdn.skypack.dev/pdfjs-viewer-element'></script>"
"""
LOAD_PDF_JS = """
function onPDFLoad(path) {
    if (path == null) {
        return;
    }
    path_str = "/gradio_api/file=" + path["path"];
    pdf_js_viewer_element = document.getElementById("pdf-viewer");
    pdf_js_viewer_element.setAttribute("src", path_str);
    pdf_js_viewer_element.setAttribute("page", 1);
    return path_str;
}
"""
CLEAR_PDF_JS = """
function clearPDF() {
    var pdfViewer = document.getElementById("pdf-viewer");
    pdfViewer.setAttribute("src", "");
}
"""
ASSIGN_CLICK_JS = """
function assignClickEvent() {
    var elements = document.querySelectorAll("a.chunk-ref");
    elements.forEach(function(element) {
        element.addEventListener("click", function() {
            var id = this.id;
            var pdfViewer = document.getElementById("pdf-viewer");
            var innerDoc = pdfViewer.iframe.contentDocument
                ? pdfViewer.iframe.contentDocument
                : pdfViewer.iframe.contentWindow.document;
            var page = id.split("-")[0];
            var bbox = id.split("-")[1];
            var left, top, right, bottom;
            if (bbox) {
                // remove first and last char "[]" from bbox
                bbox = bbox.substring(1, bbox.length - 1);
                split_bbox = bbox.split(",");
                left = parseFloat(split_bbox[0]) * 100;
                top = parseFloat(split_bbox[1]) * 100;
                right = parseFloat(split_bbox[2]) * 100;
                bottom = parseFloat(split_bbox[3]) * 100;
                width = right - left;
                height = bottom - top;
            }

            pdfViewer.setAttribute("page", page);
            var query_selector =
                "#viewer > div[data-page-number='" +
                page +
                "'] > div.textLayer"
            var textLayer = innerDoc.querySelector(query_selector);

            // remove all previous annotation sections
            if (!textLayer) {
                return;
            }
            var previous_annotation_sections = textLayer.querySelectorAll("section.chunk-ref");
            if (previous_annotation_sections.length > 0) {
                previous_annotation_sections.forEach(function(section) {
                    section.remove();
                });
            }

            var new_annotation_section = document.createElement("section");
            // set the position of the new section
            new_annotation_section.setAttribute("class", "chunk-ref");
            new_annotation_section.setAttribute("style", "position: absolute; z-index: 0; background-color: rgba(255, 255, 0, 1.0); left: " + left + "%; top: " + top + "%; width: " + width + "%; height: " + height + "%;");
            textLayer.appendChild(new_annotation_section);
        });
    });
}
"""  # noqa: E501


def format_chunk(chunk, include_image=True):
    if chunk.ctype == CType.Root or not chunk.content:
        return ""

    if chunk.mimetype in ["image/png", "image/jpeg", "image/jpg"]:
        block_img_base64 = bytes_to_base64(
            chunk.content,
            mime_type=chunk.mimetype,
        )
        block_text = chunk.text
        block_type = chunk.ctype

        if include_image:
            block_img_elem = f'<img src="{block_img_base64}" />'
            block_text = (
                "{}\n\n<details><summary>({} image)</summary>" "{}</details>"
            ).format(
                block_text,
                block_type,
                block_img_elem,
            )
    elif chunk.ctype == CType.Header:
        block_text = f"### {chunk.content}"
    else:
        block_text = chunk.content
    return block_text


@functools.lru_cache(maxsize=None)
def trim_pages(pdf_path, output_path=None, start_page=0, trim_pages=5):
    doc = pymupdf.open(pdf_path)
    if not output_path:
        output_path = pdf_path[:-4] + "_trimmed.pdf"

    num_pages = len(doc)
    if trim_pages > 0 and num_pages > trim_pages:
        to_select = list(range(start_page, min(start_page + trim_pages, num_pages)))
        doc.select(to_select)
        try:
            doc.ez_save(output_path)
        except:  # noqa: E722
            for xref in range(1, doc.xref_length()):
                try:
                    _ = doc.xref_object(xref)
                except:  # noqa: E722
                    doc.update_object(xref, "<<>>")
            doc.ez_save(output_path)

        print("Trimmed pdf to path", output_path)
    else:
        return pdf_path

    return output_path


def convert_document(
    pdf_path,
    method,
    use_full_page=False,
    force_ocr=False,
    generate_toc=False,
    add_reference_links=True,
    enabled=True,
):
    if enabled:
        print("Processing file", pdf_path, "with method", method)
        gr.Info(f"Processing file with method `{method}`")
    else:
        return "", "", "", []

    if not pdf_path:
        raise ValueError("No file provided")

    method_cls = METHOD_MAP[method]
    # trim to MAX_PAGES
    if MAX_PAGES > 0 and MAX_PAGES_CHUNKING > 0:
        max_pages = MAX_PAGES_CHUNKING if method_cls == FastPDF else MAX_PAGES
        old_pdf_path = pdf_path
        pdf_path = trim_pages(
            pdf_path,
            start_page=0,
            trim_pages=max_pages,
        )
        if pdf_path != old_pdf_path:
            gr.Info(f"Method `{method}` will process only first {max_pages} pages.")

    # benchmarking
    start = time.time()
    debug_image_paths = []

    root = ctrl.as_root_chunk(pdf_path)
    if method_cls == FastPDF:
        chunks = method_cls.run(
            root,
            # preset=ParserPreset.BEST,
            render_full_page=use_full_page,
            ocr_mode=OCRMode.AUTO if not force_ocr else OCRMode.ON,
            use_layout_parser=True,
            render_2d_text_paragraph=True,
            extract_image=True,
            extract_table=True,
            debug_path=DEBUG_DIR,
        )
    else:
        chunks = method_cls.run(root)

    if generate_toc:
        print("Generating Table-of-Content")
        toc_text = (
            "<details open='true'><summary><big>Table-of-Content</big></summary>\n\n"
        )
        new_chunks = TOCHierarchyBuilder.run(
            chunks,
            use_llm=True,
            model=MODEL_NAME,
        )
        for level, chunk in new_chunks[0].walk():
            if chunk.ctype != CType.Header:
                continue

            chunk_page = chunk.origin.location["page"]
            chunk_bbox = chunk.origin.location["bbox"]
            chunk_text = (
                "<h5>" + "".join(["&nbsp;&nbsp;&nbsp;"] * level) + chunk.content
            )
            ref_link = (
                f" <a class='chunk-ref' id='{chunk_page}-{chunk_bbox}'>[↗]</a>"
                if add_reference_links
                else ""
            )
            chunk_text += ref_link + "</h5>"
            toc_text += chunk_text + "\n\n"

        toc_text += "</details>\n\n---\n\n"
    else:
        toc_text = ""

    # serialize the child chunks to list
    child_chunks = [chunk for _, chunk in chunks[0].walk() if chunk.ctype != CType.Root]
    print("Total chunks:", len(child_chunks))

    max_page = 1
    print("Table Of Content")
    for chunk in child_chunks:
        if chunk.ctype == CType.Header:
            print(chunk.content)
        max_page = max(max_page, chunk.origin.location["page"])

    rendered_texts = []
    for chunk in child_chunks:
        chunk_page = chunk.origin.location["page"]
        chunk_bbox = chunk.origin.location["bbox"]
        ref_link = (
            f"\n<a class='chunk-ref' id='{chunk_page}-{chunk_bbox}'>[↗]</a>"
            if add_reference_links
            else ""
        )
        text = format_chunk(chunk) + ref_link
        rendered_texts.append(text)

    rendered_texts_no_img = [
        format_chunk(chunk, include_image=False) for chunk in child_chunks
    ]
    # remove empty strings
    combined_text = toc_text + "\n\n".join([text for text in rendered_texts if text])
    combined_text_no_img = "\n\n".join([text for text in rendered_texts_no_img if text])

    # alternative rendering
    # combined_text = combined_text_no_img = chunks[0].render("markdown")

    duration = time.time() - start
    duration_per_page = duration / max_page
    duration_message = (
        f"Conversion with {method} ({max_page} pages) took *{duration:.2f}s* total - "
        f"*{duration_per_page:.2f}s* per page"
    )
    print(duration_message)

    # create named temporary folder
    debug_dir = TMP_DIR / Path(pdf_path).stem
    debug_dir.mkdir(exist_ok=True)

    if pdf_path.endswith(".pdf"):
        plot_pdf(pdf_path, child_chunks, debug_dir)
    else:
        plot_img(pdf_path, child_chunks, debug_dir)

    debug_image_paths = list(debug_dir.glob("*.png"))

    return (
        duration_message,
        combined_text,
        combined_text_no_img,
        debug_image_paths,
    )


def to_zip_file(file_path, methods, *output_components):
    if not file_path:
        return gr.update(visible=False)

    markdown_text_dict = dict()
    debug_images_dict = defaultdict(list)
    for idx, method_name in enumerate(METHOD_LIST):
        if method_name not in methods:
            continue

        markdown_text = output_components[idx * 4 + 2]
        debug_images = output_components[idx * 4 + 3]

        markdown_text_dict[method_name] = markdown_text
        debug_images_dict[method_name] = debug_images

    # create new temp directory using Python's tempfile module
    temp_dir = Path(file_path).parent
    zip_file_path = temp_dir / "output.zip"

    markdown_path = temp_dir / f"{method_name}.md"
    with open(markdown_path, "w") as f:
        f.write(markdown_text)

    # create a zip file in write mode
    with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for method_name, markdown_text in markdown_text_dict.items():
            debug_image_paths = debug_images_dict[method_name]

            # write the markdown text to the zip file
            zipf.write(
                markdown_path,
                f"{method_name}/{method_name}.md",
            )
            if debug_image_paths:
                for idx, (debug_image_path, _) in enumerate(debug_image_paths):
                    debug_image_name = Path(debug_image_path).name
                    zipf.write(
                        debug_image_path,
                        f"{method_name}/{debug_image_name}",
                    )

    return gr.update(
        value=str(zip_file_path),
        visible=True,
    )


def show_tabs(selected_methods):
    visible_tabs = []
    for method in METHOD_MAP:
        visible_tabs.append(gr.update(visible=method in selected_methods))

    return visible_tabs


with gr.Blocks(
    theme=gr.themes.Ocean(
        font_mono="PT Mono",
    ),
    head=HEAD_HTML,
) as demo:
    with open("header.html") as file:
        header = file.read()
    gr.HTML(header)
    output_components = []
    output_tabs = []
    visualization_sub_tabs = []

    with gr.Row():
        with gr.Column(variant="panel", scale=4):
            input_file = gr.File(
                label="Upload PDF document",
                file_types=[
                    ".pdf",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".tiff",
                    ".tif",
                    ".bmp",
                ],
            )
            with gr.Accordion("Examples:"):
                example_root = "pdfs/"
                gr.Examples(
                    examples=[
                        os.path.join(example_root, _)
                        for _ in os.listdir(example_root)
                        if _.endswith(".pdf")
                    ],
                    inputs=input_file,
                )
            progress_status = gr.Markdown("", show_label=False, container=False)
            output_file = gr.File(
                label="Download output",
                interactive=False,
                visible=False,
            )

        with gr.Column(variant="panel", scale=5):
            methods = gr.Dropdown(
                METHOD_LIST,
                label=("Conversion methods"),
                value=["easyparser_fastpdf"],
                multiselect=True,
            )
            with gr.Row():
                full_page_render = gr.Checkbox(
                    label="Use full page layout-preserved rendering",
                    value=False,
                )
                force_page_ocr = gr.Checkbox(
                    label="Force OCR",
                    value=False,
                )
            with gr.Row():
                generate_toc = gr.Checkbox(
                    label="Generate Table-of-Content",
                    value=False,
                    visible=False,
                )
                add_reference_links = gr.Checkbox(
                    label="Add reference links to original PDF",
                    value=True,
                )
            with gr.Row():
                convert_btn = gr.Button("Convert", variant="primary", scale=2)
                clear_btn = gr.ClearButton(value="Clear", scale=1)

    with gr.Row():
        with gr.Column(variant="panel", scale=4):
            pdf_preview = gr.HTML(
                f"""
                <div class="pdf-wrapper">
                <pdfjs-viewer-element
                    id="pdf-viewer"
                    viewer-path="gradio_api/file={PDF_JS_DIR}"
                    locale="en"
                    style="width: 100%; height: 800px;"
                >
                </pdfjs-viewer-element>
                </div>
                """
            )

        with gr.Column(variant="panel", scale=5):
            with gr.Tabs():
                for method in METHOD_MAP:
                    with gr.Tab(method, visible=False) as output_tab:
                        with gr.Tabs():
                            with gr.Tab("Markdown render"):
                                markdown_render = gr.Markdown(
                                    label="Markdown rendering",
                                    height=800,
                                    show_copy_button=True,
                                )
                            with gr.Tab("Markdown text"):
                                markdown_text = gr.TextArea(
                                    lines=45, show_label=False, container=False
                                )
                            with gr.Tab(
                                "Debug visualization",
                            ) as visual_sub_tab:
                                output_description = gr.Markdown(
                                    container=False,
                                    show_label=False,
                                )
                                debug_images = gr.Gallery(
                                    show_label=False,
                                    container=False,
                                    interactive=False,
                                )

                    output_components.extend(
                        [
                            output_description,
                            markdown_render,
                            markdown_text,
                            debug_images,
                        ]
                    )
                    output_tabs.append(output_tab)
                    visualization_sub_tabs.append(visual_sub_tab)

    click_event = convert_btn.click(
        fn=show_tabs,
        inputs=[methods],
        outputs=output_tabs,
    )
    for idx, method in enumerate(METHOD_LIST):

        def progress_message(selected_methods, method=method):
            selected_methods_indices = [
                idx
                for idx, current_method in enumerate(METHOD_LIST)
                if current_method in selected_methods
            ]
            try:
                current_method_idx = selected_methods_indices.index(
                    METHOD_LIST.index(method)
                )
                msg = (
                    f"Processing ({current_method_idx + 1} / "
                    f"{len(selected_methods)}) **{method}**...\n\n"
                )
            except ValueError:
                msg = gr.update()

            return msg

        def process_method(
            input_file,
            selected_methods,
            use_full_page,
            force_ocr,
            generate_toc,
            add_reference_links,
            method=method,
        ):
            return convert_document(
                input_file,
                method=method,
                use_full_page=use_full_page,
                force_ocr=force_ocr,
                generate_toc=generate_toc,
                add_reference_links=add_reference_links,
                enabled=method in selected_methods,
            )

        click_event = click_event.then(
            fn=lambda methods, method=method: progress_message(methods, method),
            inputs=[methods],
            outputs=[progress_status],
        ).then(
            fn=lambda input_file, methods, use_full_page, force_ocr, generate_toc, add_reference_links, method=method: process_method(  # noqa: E501
                input_file,
                methods,
                use_full_page,
                force_ocr,
                generate_toc,
                add_reference_links,
                method,
            ),
            inputs=[
                input_file,
                methods,
                full_page_render,
                force_page_ocr,
                generate_toc,
                add_reference_links,
            ],
            outputs=output_components[idx * 4 : (idx + 1) * 4],
        )

    click_event.then(
        lambda: "All tasks completed.",
        outputs=[progress_status],
    ).then(
        fn=to_zip_file,
        inputs=[
            input_file,
            methods,
        ]
        + output_components,
        outputs=[output_file],
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js=ASSIGN_CLICK_JS,
    )

    clear_btn.add(
        [
            input_file,
            output_file,
        ]
        + output_components
    )
    clear_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[output_file],
    ).then(
        fn=None,
        inputs=None,
        outputs=output_components,
        js=CLEAR_PDF_JS,
    )

    input_file.change(
        fn=None,
        inputs=input_file,
        outputs=None,
        js=LOAD_PDF_JS,
    )

    demo.queue(default_concurrency_limit=4).launch(
        show_error=True,
        max_file_size="50mb",
        allowed_paths=[
            TMP_DIR,
            PDF_JS_DIR,
        ],
    )
