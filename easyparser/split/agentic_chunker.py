# flake8: noqa: E501
"""
Adapt from Greg Kamradt's agentic_chunker.py implementation.
Source: https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/a4570f3c4883eb9b835b0ee18990e62298f518ef/tutorials/LevelsOfTextSplitting/agentic_chunker.py
"""
import uuid

from easyparser.base import BaseOperation, Chunk, ChunkGroup
from easyparser.models import completion

from .propositionizer import Propositionizer

NEW_GROUP_SUMMARY = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the new chunk summary, nothing else.
"""

NEW_GROUP_TITLE = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about. You will be given a summary of a chunk which needs a title.

Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.
"""


UPDATE_GROUP_SUMMARY = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. A new proposition was just added to one of your chunks, you should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a group of propositions which are in the chunk and the chunks current summary.

Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Proposition: Greg likes to eat pizza
Output: This chunk contains information about the types of food Greg likes to eat.

Only respond with the chunk new summary, nothing else.
"""


UPDATE_GROUP_TITLE = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic. A new proposition was just added to one of your chunks, you should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

A good title will say what the chunk is about.

You will be given a group of propositions which are in the chunk, chunk summary and the chunk title.

Your title should anticipate generalization. If you get a proposition about apples, generalize it to food. Or month, generalize it to "date and times".

Example:
Input: Summary: This chunk is about dates and times that the author talks about
Output: Date & Times

Only respond with the new chunk title, nothing else.
"""


FIND_RELEVANT_PROP_GROUP = """Determine whether or not the "Proposition" should belong to any of the existing chunks. A proposition should belong to a chunk of their meaning, direction, or intention are similar. The goal is to group similar propositions and chunks. If you think a proposition should be joined with a chunk, return the chunk id. The returned chunk id should be a single text on a line. If you do not think an item should be joined with an existing chunk, just return "No chunks"

Example:
Input:
    - Proposition: "Greg really likes hamburgers"
    - Current Chunks:
        - Chunk ID: 2n4l3d
        - Chunk Name: Places in San Francisco
        - Chunk Summary: Overview of the things to do with San Francisco Places

        - Chunk ID: 93833k
        - Chunk Name: Food Greg likes
        - Chunk Summary: Lists of the food and dishes that Greg likes
Output: 93833k
"""


def create_prop_group(chunk: Chunk, prop_groups: dict, model: str | None = None):
    """Create proposition group and add it to prop_groups"""
    prop_id = str(uuid.uuid4())[:6]
    summary = completion(
        "Determine the summary of the new chunk that this proposition will go "
        f"into:\n{chunk.content}",
        system=NEW_GROUP_SUMMARY,
        model=model,
    )
    title = completion(
        f"Determine the title of the chunk that this summary belongs to:\n{summary}",
        system=NEW_GROUP_TITLE,
        model=model,
    )

    # store the prop
    prop_groups[prop_id] = {
        "chunk_id": prop_id,
        "propositions": [chunk],
        "title": title,
        "summary": summary,
        "chunk_index": len(prop_groups),
    }


def find_relevant_props_group(chunk: Chunk, prop_group: dict, model: str | None) -> str:
    current_chunk_outline = get_chunk_outline(prop_group)
    resp = completion(
        "Current Chunks:\n"
        "--Start of current chunks--\n"
        f"{current_chunk_outline}\n"
        "--End of current chunks--\n\n"
        "Determine if the following statement should belong to one of the chunks "
        f"outlined:\n{chunk.content}",
        system=FIND_RELEVANT_PROP_GROUP,
        model=model,
    )

    for chunk_id in prop_group:
        if chunk_id in resp:
            return chunk_id

    return ""


def get_chunk_outline(prop_group: dict) -> str:
    """Get a string which represents the chunks you currently have.
    This will be empty when you first start off
    """
    chunk_outline = ""

    for _, chunk in prop_group.items():
        single_chunk_string = (
            f"Chunk ID: {chunk['chunk_id']}\n"
            f"Chunk Name: {chunk['title']}\n"
            f"Chunk Summary: {chunk['summary']}\n\n"
            ""
        )
        chunk_outline += single_chunk_string

    return chunk_outline


def update_group(group_id: str, prop_group: dict, model: str | None):
    group = prop_group[group_id]
    prop_str = "\n".join(prop.content for prop in group["propositions"])
    summary = completion(
        f"Chunk's propositions:\n{prop_str}\n\n"
        f"Current chunk summary:\n{group['summary']}",
        system=UPDATE_GROUP_SUMMARY,
        model=model,
    )
    title = completion(
        f"Chunk's propositions:\n{prop_str}\n\n"
        f"Chunk summary:\n{summary}\n\n"
        f"Current chunk title:\n{group['title']}",
        system=UPDATE_GROUP_TITLE,
        model=model,
    )
    group["summary"] = summary
    group["title"] = title


class AgenticChunker(BaseOperation):

    @classmethod
    def run(
        cls,
        chunks: Chunk | ChunkGroup,
        regenerate_metadata: bool = False,
        model: str | None = None,
        **kwargs,
    ) -> ChunkGroup:
        """Chunk the text into smaller pieces based on the content of the text."""
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            propositions = Propositionizer.run(root, model=model, **kwargs)
            prop_group = {}
            for idx_, (_, prop) in enumerate(propositions[0].walk()):
                from pprint import pprint

                print(f"==== {idx_}")
                pprint(prop_group)
                if not prop_group:
                    create_prop_group(prop, prop_group, model=model)
                    continue
                group_id = find_relevant_props_group(prop, prop_group, model=model)
                print(f"Group id: {group_id}")
                if not group_id:
                    create_prop_group(prop, prop_group, model=model)
                    continue

                prop_group[group_id]["propositions"].append(prop)
                if regenerate_metadata:
                    update_group(group_id, prop_group, model=model)

            if not prop_group:
                continue

            chs = [
                Chunk(
                    mimetype="text/plain",
                    content=" ".join(e.content for e in group["propositions"]),
                    summary=group["summary"],
                    metadata={
                        "originals": [prop.id for prop in group["propositions"]],
                        "title": group["title"],
                        "group_id": group_id,
                        "group_index": group["chunk_index"],
                    },
                )
                for group_id, group in prop_group.items()
            ]
            for idx, ch in enumerate(chs[1:], start=1):
                ch.prev = chs[idx - 1]
                chs[idx - 1].next = ch

            # Return the first chunk
            output.append(chs[0])

        return output
