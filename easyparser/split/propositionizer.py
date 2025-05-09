# flake8: noqa: E501
import json
import re

from easyparser.base import BaseOperation, Chunk, ChunkGroup
from easyparser.models import completion, parse_json_from_text

CAPTION_PROMPT = """Decompose the "Input" into clear and simple propositions, ensuring they are interpretable out of context.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Input: "The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare’s scratch or form and a lapwing’s nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output:
```
[ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare’s scratch or form and a lapwing’s nest look very similar.", "Both
hares and lapwing’s nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America." ]
```

Input: {text}
Output:"""


def completion_json(message: str, alias=None) -> list:
    """Run LLM and extract JSON from the triple backticks"""
    output = completion(message, model=alias)
    output_list = parse_json_from_text(output)
    if isinstance(output_list, list):
        return output_list
    return []


def get_proposition(chunk: Chunk, model: str | None = None) -> list[Chunk]:
    """Get chunk propositions

    Args:
        chunk: chunk to be propositionized
        model: registered model alias to be used

    Returns:
        list of propositions, each is a Chunk
    """
    if chunk.text:
        props = completion_json(CAPTION_PROMPT.format(text=chunk.text), alias=model)
        propositions = [
            Chunk(
                mimetype="text/plain", content=prop, metadata={"originals": [chunk.id]}
            )
            for prop in props
        ]
    else:
        content = chunk.content
        if isinstance(content, str) and content:
            props = completion_json(CAPTION_PROMPT.format(text=content, alias=model))
            propositions = [
                Chunk(
                    mimetype="text/plain",
                    content=prop,
                    metadata={"originals": [chunk.id]},
                )
                for prop in props
            ]
        else:
            propositions = []
        child = chunk.child
        if child:
            propositions += get_proposition(child, model=model)

    if chunk.next:
        propositions += get_proposition(chunk.next, model=model)

    return propositions


class Propositionizer(BaseOperation):

    @classmethod
    def run(
        cls, chunks: Chunk | ChunkGroup, model: str | None = None, **kwargs
    ) -> ChunkGroup:
        """Build propositions from the chunk.

        Suggest to flatten the chunks first before running this operation so that each
        chunk has good enough context for LLM to generate the propositions.

        Ref: Dense X Retrieval: What Retrieval Granularity Should We Use?
        """
        if isinstance(chunks, Chunk):
            chunks = ChunkGroup([chunks])

        output = ChunkGroup()
        for root in chunks:
            ch = root
            propositions = get_proposition(ch, model=model)
            for idx, ch_ in enumerate(propositions[1:], start=1):
                ch_.prev = propositions[idx - 1]
                propositions[idx - 1].next = ch_

            if propositions:
                output.append(propositions[0])

        return output
