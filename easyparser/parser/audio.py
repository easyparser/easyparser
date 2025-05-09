import logging
from typing import Any, Dict, Optional

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType

logger = logging.getLogger(__name__)


class AudioWhisperParser(BaseOperation):

    _loaded_models: Dict[str, Any] = {}

    @classmethod
    def _get_model(cls, model_name: str = "base"):
        """Get or load a Whisper model, caching it for reuse"""
        if model_name not in cls._loaded_models:
            import whisper

            logger.debug(f"Loading Whisper model: {model_name}")
            cls._loaded_models[model_name] = whisper.load_model(model_name)
        return cls._loaded_models[model_name]

    @classmethod
    def transcribe_audio(
        cls,
        audio_file_path: str,
        model_name: str = "base",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio file to text using OpenAI Whisper"""
        options = options or {}
        try:
            model = cls._get_model(model_name)
            result = model.transcribe(audio_file_path, **options)
            return result
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {"text": "", "segments": []}

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        model_name: str = "base",
        language: str = "",
        **kwargs: Any,
    ) -> ChunkGroup:
        """Parse audio files and convert to text transcription.

        Args:
            chunk: the chunk or chunk group to process
            model_name: whisper model to use ('tiny', 'base', 'small', 'medium',
                'large', 'turbo')
            language: optional language code to help with transcription

        Returns:
            ChunkGroup containing the processed chunks
        """
        # Resolve chunk
        if isinstance(chunk, Chunk):
            chunk = ChunkGroup(chunks=[chunk])
        try:
            options = {}
            if language:
                options["language"] = language

            output = ChunkGroup()
            for mc in chunk:
                # Get file info
                file_path = mc.origin.location

                logger.info(f"Parsing {file_path}")

                # Perform transcription - using the cached model
                result = cls.transcribe_audio(file_path, model_name, options)
                transcription = result.get("text", "")
                segments = result.get("segments", [])

                if not transcription:
                    logger.warning(f"Failed to transcribe audio file: {file_path}")
                    # Still include the chunk in output
                    output.append(mc)
                    continue

                # Process segments for metadata
                processed_segments = []
                for segment in segments:
                    processed_segments.append(
                        {
                            "start": segment.get("start", 0),
                            "end": segment.get("end", 0),
                            "text": segment.get("text", ""),
                        }
                    )

                # Create a transcription chunk with segments in metadata
                transcript_chunk = Chunk(
                    mimetype="text/plain",
                    ctype=CType.Para,
                    content=transcription,
                    origin=mc.origin,
                    parent=mc,
                    metadata={
                        "transcription_model": f"whisper:{model_name}",
                        "segments": processed_segments,
                        "language": result.get("language"),
                        "duration": result.get("duration", 0),
                    },
                )

                mc.child = transcript_chunk
                output.append(mc)

            return output

        finally:
            cls.cleanup()

    @classmethod
    def cleanup(cls):
        """Clean up loaded models to free memory"""
        cls._loaded_models.clear()

    @classmethod
    def py_dependency(cls) -> list[str]:
        """Return the list of Python dependencies required by this converter."""
        return ["openai-whisper"]
