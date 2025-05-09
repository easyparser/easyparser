import io
import json
import re

import llm
from PIL import Image

DEFAULT_LLM_ALIAS = "easyparser-llm"
_models = {}


def get_model(alias: str | None = None):
    """Get the specific model"""
    if alias is None:
        alias = DEFAULT_LLM_ALIAS

    if alias not in _models:
        try:
            _models[alias] = llm.get_model(alias)
        except llm.UnknownModelError as e:
            if alias == DEFAULT_LLM_ALIAS:
                raise ValueError(
                    "Can't find default model 'easyparser-llm'. Please check the \"Add "
                    'LLM support" section.'
                ) from None

            raise ValueError(f"Model with alias '{alias}' not found.") from e

    return _models[alias]


def completion(
    message: str, attachments=None, system=None, model=None, schema=None
) -> str:
    """Chat with a model"""
    model_obj = get_model(model)
    if attachments is None:
        return model_obj.prompt(message, system=system).text()

    if attachments is not None and not isinstance(attachments, (list, tuple)):
        attachments = [attachments]

    processed_attachments = []
    for idx, attachment in enumerate(attachments):
        if isinstance(attachment, llm.Attachment):
            processed_attachments.append(attachment)
        elif isinstance(attachment, Image.Image):
            buf = io.BytesIO()
            attachment.save(buf, format="PNG")
            processed_attachments.append(
                llm.Attachment(type="image/png", content=buf.getvalue())
            )
        else:
            raise ValueError(
                f"Unsupported attachment type: {type(attachment)} for attachment {idx}."
                " Please provide llm.Attachment or PIL.Image."
            )

    resp = model_obj.prompt(
        message, attachments=processed_attachments, system=system, schema=schema
    )
    return resp.text()


def parse_json_from_text(text) -> list | dict | None:
    """Parses a JSON object from a text string that is surrounded by code fences (```)

    Assumes the content within the code fences is JSON, even if the fence doesn't
    explicitly state "json".

    Args:
        text: the input text string potentially containing a JSON object within
    code fences.

    Returns:
        A Python dictionary or list representing the parsed JSON object if found,
        or None if no valid JSON object within code fences is found.
        Returns the raw string if it's not a valid JSON.
    """
    rematch = re.search(r"```.*?\n(.*?)\n```", text, re.DOTALL)

    if rematch:
        json_string = rematch.group(1).strip()
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e, "in text:", json_string)
            return None

    return None
