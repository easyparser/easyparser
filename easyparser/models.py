import io

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
                    "Can't find default model 'easyparser-llm'. Please run "
                    "`easyparser setup-llm` for step-by-step setup."
                ) from None

            raise ValueError(f"Model with alias '{alias}' not found.") from e

    return _models[alias]


def completion(message: str, attachments=None, alias=None) -> str:
    """Chat with a model"""
    model = get_model(alias)
    if attachments is None:
        return model.prompt(message).text

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

    resp = model.prompt(message, attachments=processed_attachments)

    return resp.text()
