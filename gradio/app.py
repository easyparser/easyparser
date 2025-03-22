import os
import time
import zipfile
from collections import defaultdict
from pathlib import Path

from gradio_pdf import PDF

import gradio as gr
from easyparser.mime import mime_pdf
from easyparser.parse import DoclingPDF, FastPDF, SycamorePDF, UnstructuredPDF
from easyparser.util.plot import plot_pdf

METHOD_MAP = {
    "easyparser_fastpdf": FastPDF,
    "unstructured": UnstructuredPDF,
    "docling": DoclingPDF,
    "sycamore": SycamorePDF,
}
METHOD_LIST = list(METHOD_MAP.keys())
TMP_DIR = Path("/tmp/visualize")
TMP_DIR.mkdir(exist_ok=True)


def convert_document(pdf_path, method, enabled=True):
    if enabled:
        print("Processing file", pdf_path, "with method", method)
    else:
        return "", "", "", []

    # benchmarking
    start = time.time()
    debug_image_paths = []

    root = mime_pdf.as_root_chunk(pdf_path)
    method = METHOD_MAP[method]
    chunks = method.run(root)

    text = "\n\n".join([chunk.text for chunk in chunks])

    duration = time.time() - start
    duration_message = f"Conversion with {method} took *{duration:.2f} seconds*"
    print(duration_message)

    # create named temporary folder
    debug_dir = TMP_DIR / Path(pdf_path).stem
    debug_dir.mkdir(exist_ok=True)
    plot_pdf(pdf_path, chunks, debug_dir)

    debug_image_paths = list(debug_dir.glob("*.png"))

    return (
        duration_message,
        text,
        text,
        debug_image_paths,
    )


def to_zip_file(file_path, methods, *output_components):
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
    theme=gr.themes.Ocean(),
) as demo:
    with open("gradio/header.html") as file:
        header = file.read()
    gr.HTML(header)
    output_components = []
    output_tabs = []
    visualization_sub_tabs = []

    with gr.Row():
        with gr.Column(variant="panel", scale=5):
            input_file = gr.File(
                label="Upload PDF document",
                file_types=[
                    ".pdf",
                ],
            )
            with gr.Accordion("Examples:"):
                example_root = "./"
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
            with gr.Row():
                methods = gr.Dropdown(
                    METHOD_LIST,
                    label=("Conversion methods"),
                    value=["easyparser_fastpdf"],
                    multiselect=True,
                )
            with gr.Row():
                convert_btn = gr.Button("Convert", variant="primary", scale=2)
                clear_btn = gr.ClearButton(value="Clear", scale=1)

    with gr.Row():
        with gr.Column(variant="panel", scale=5):
            pdf_preview = PDF(
                label="PDF preview",
                interactive=False,
                visible=True,
                height=800,
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

    input_file.change(fn=lambda x: x, inputs=input_file, outputs=pdf_preview)

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

        def process_method(input_file, selected_methods, method=method):
            return convert_document(
                input_file,
                method=method,
                enabled=method in selected_methods,
            )

        click_event = click_event.then(
            fn=lambda methods, method=method: progress_message(methods, method),
            inputs=[methods],
            outputs=[progress_status],
        ).then(
            fn=lambda input_file, methods, method=method: process_method(
                input_file, methods, method
            ),
            inputs=[input_file, methods],
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
    )

    clear_btn.add(
        [
            input_file,
            pdf_preview,
            output_file,
        ]
        + output_components
    )
    clear_btn.click(
        fn=lambda: gr.update(visible=False),
        outputs=[output_file],
    )

    demo.queue(default_concurrency_limit=4).launch(
        show_error=True,
        max_file_size="50mb",
        allowed_paths=[
            TMP_DIR,
        ],
    )
