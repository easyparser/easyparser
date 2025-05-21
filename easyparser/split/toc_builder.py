import logging
import re
from collections import defaultdict
from typing import List

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType
from easyparser.mime.base import MimeType
from easyparser.models import completion

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """<task> You are given a set of headers from a document between the starting tag <passages> and ending tag </passages>. Each header is labeled as 'ID `N`' where 'N' is the header number. Your task is to organize the header to markdown tree with (#, ##) tag to represent the correct structure of the table of contents. Some header is not correctly detected and should be removed. </task>

<rules>
Follow the following rules:
- Return in the format
# ID `1`
## ID `2`
### ID `3`
(more levels if needed)
etc
- MUST use correct ID tag
- Only output the header IDs with the format ID `` without any other text
- If the header is a noisy one, remove the ID from the output
</rules>

<passages>
{passages}
</passages>
"""  # noqa: E501


def _add_id_to_headers(headers: List[str]) -> List[str]:
    """Prepare the splits for the chunker."""
    return [
        f"ID `{i}`: " + header.content.replace("\n", "").strip()
        for (i, header) in enumerate(headers)
    ]


def _build_toc_tree(
    headers: List[Chunk],
    model: str | None,
) -> List[Chunk]:
    """Build the table of contents tree from the headers."""
    headers_text_with_id = _add_id_to_headers(headers)
    headers_text = "\n".join(headers_text_with_id)
    response = completion(
        PROMPT_TEMPLATE.format(passages=headers_text),
        model=model,
    )
    logger.debug(f"Response: {response}")
    # parse the response
    id_pattern = r"ID `(\d+)`"
    headers_level = {}
    for line in response.split("\n"):
        # count number of leading # to determine the level
        level = line.count("#")
        # use regex to get the header id
        match = re.search(id_pattern, line)
        if match:
            header_id = int(match.group(1))
            # find the header with the id
            headers_level[header_id] = level
            logger.debug("#" * level + " " + headers[header_id].content)

    # find parent_id for each header
    # based on the reading order
    child_indices = defaultdict(list)
    for header_id, level in headers_level.items():
        # find the parent header
        for i in range(header_id - 1, -1, -1):
            if headers_level.get(i) is not None:
                if headers_level[i] < level:
                    child_indices[i].append(header_id)
                    break

    to_remove_header_ids = set(range(len(headers))) - set(headers_level.keys())
    return headers_level, child_indices, to_remove_header_ids


def _format_toc_tree(
    headers: List[Chunk],
    headers_level: dict[int, int],
) -> str:
    """Format the table of contents tree for output."""
    toc = []
    for header_id, header in enumerate(headers):
        header_level = headers_level.get(header_id)
        if header_level is None:
            continue
        toc.append(f"{'#' * (header_level)} {header.content}")
    return "\n".join(toc)


def _pdfium_get_toc(
    pdf_path: str,
    max_depth: int = 15,
):
    """Get the table of contents from the pdfium."""
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_path)
    toc = doc.get_toc(max_depth=max_depth)
    output = ""

    for bm in toc:
        line = "#" * (bm.level + 1) + " " + bm.title
        output += line + "\n"

    return output


class TOCExtractor(BaseOperation):
    """Extract the table of contents from the document."""

    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        use_llm: bool = False,
        model: str | None = None,
        **kwargs,
    ) -> ChunkGroup:
        """Extract the table of contents and output
        as a new Chunk for each Root."""
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            if root.ctype != CType.Root:
                continue

            header_chunks = []
            for _, child_chunk in root.walk():
                if child_chunk.ctype == CType.Header:
                    header_chunks.append(child_chunk)

            # first check if the pdfium can get the TOC
            pdf_path = root.origin.location
            toc_content = _pdfium_get_toc(
                pdf_path,
            )
            # if the toc_content is empty, use the LLM
            if len(toc_content) == 0 and use_llm:
                header_levels, _, _ = _build_toc_tree(
                    header_chunks,
                    model=model,
                )
                toc_content = _format_toc_tree(
                    header_chunks,
                    header_levels,
                )

            new_root = root.clone(no_relation=True)
            toc_chunk = Chunk(
                mimetype=MimeType.text,
                content=toc_content,
                ctype=CType.Div,
            )
            new_root.add_children([toc_chunk])
            output.append(new_root)

        return output


class TOCHierarchyBuilder(BaseOperation):

    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        use_llm: bool = False,
        model: str | None = None,
        **kwargs,
    ) -> ChunkGroup:
        """Group chunk by preceding header,
        assuming the reading order is correct"""

        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            cur_root_children = []
            parent_to_child_mapping = defaultdict(list)
            all_headers = []

            increment_header_id = -1
            cur_header_id = -1
            cur_header_children = []

            if use_llm:
                headers = [
                    chunk for _, chunk in root.walk() if chunk.ctype == CType.Header
                ]
                _, child_header_indices, to_remove_header_ids = _build_toc_tree(
                    headers, model=model
                )

            for _, child_chunk in root.walk():
                new_child_chunk = child_chunk.clone(no_relation=True)

                if new_child_chunk.ctype in [CType.Root, CType.Div, CType.Page]:
                    continue

                if new_child_chunk.ctype == CType.Header:
                    increment_header_id += 1
                    all_headers.append(new_child_chunk)

                if new_child_chunk.ctype == CType.Header and (
                    not use_llm or increment_header_id not in to_remove_header_ids
                ):
                    # commit the previous header
                    if cur_header_id >= 0 and cur_header_children:
                        parent_to_child_mapping[cur_header_id].extend(
                            cur_header_children
                        )

                    cur_header_id = increment_header_id
                    cur_header_children = []
                else:
                    if (
                        new_child_chunk.ctype == CType.Header
                        and increment_header_id in to_remove_header_ids
                    ):
                        # change ctype to Para
                        new_child_chunk.ctype = CType.Para

                    if cur_header_id >= 0:
                        cur_header_children.append(new_child_chunk)
                    else:
                        cur_root_children.append(new_child_chunk)

            # commit the last header
            if cur_header_id >= 0 and cur_header_children:
                parent_to_child_mapping[cur_header_id].extend(cur_header_children)

            # get root headers and add them to the root children
            if use_llm:
                header_child_indices = set()
                for header_id, child_header_ids in child_header_indices.items():
                    header_child_indices.update(child_header_ids)
                root_header_indices = sorted(
                    set(range(len(all_headers)))
                    - set(to_remove_header_ids)
                    - set(header_child_indices)
                )
            else:
                root_header_indices = range(len(all_headers))
            root_headers = [all_headers[i] for i in root_header_indices]
            cur_root_children.extend(root_headers)

            if use_llm:
                # update the child headers from child_header_indices
                for header_id, child_header_ids in child_header_indices.items():
                    # remove the header from the mapping
                    parent_to_child_mapping[header_id].extend(
                        [all_headers[i] for i in child_header_ids]
                    )

            # set the children
            for header_id, child_chunks in parent_to_child_mapping.items():
                all_headers[header_id].add_children(child_chunks)

            # remove children from root
            new_root = root.clone(no_relation=True)
            new_root.add_children(cur_root_children)
            output.append(new_root)

        return output
