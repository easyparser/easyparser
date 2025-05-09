import logging
import os
import tempfile
from typing import Any, Dict, Optional

from easyparser.base import BaseOperation, Chunk, ChunkGroup, CType

logger = logging.getLogger(__name__)


class VideoWhisperParser(BaseOperation):

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
    def _extract_audio(cls, video_file_path: str) -> str:
        """Extract audio from video file to a temporary mp3 file"""
        from pydub import AudioSegment

        try:
            # Create a temporary file for the extracted audio
            fd, temp_audio_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)

            logger.debug(f"Extracting audio from video: {video_file_path}")
            audio = AudioSegment.from_file(video_file_path)
            audio.export(temp_audio_path, format="mp3")

            return temp_audio_path
        except Exception as e:
            logger.error(f"Error extracting audio from video: {e}")
            return ""

    @classmethod
    def transcribe_video(
        cls,
        video_file_path: str,
        model_name: str = "base",
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transcribe video file to text using OpenAI Whisper"""
        options = options or {}
        audio_file = None

        try:
            # Extract audio from video
            audio_file = cls._extract_audio(video_file_path)
            if not audio_file:
                return {"text": "", "segments": []}

            # Transcribe the extracted audio
            model = cls._get_model(model_name)
            result = model.transcribe(audio_file, **options)
            return result
        except Exception as e:
            logger.error(f"Error transcribing video: {e}")
            return {"text": "", "segments": []}
        finally:
            # Clean up the temporary audio file regardless of success or failure
            if audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                    logger.debug(f"Removed temporary audio file: {audio_file}")
                except Exception as del_err:
                    logger.warning(
                        f"Failed to remove temporary audio file {audio_file}: {del_err}"
                    )

    @classmethod
    def run(
        cls,
        chunk: Chunk | ChunkGroup,
        model_name: str = "base",
        language: str = "",
        **kwargs: Any,
    ) -> ChunkGroup:
        """Parse video files and convert to text transcription.

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

                # Perform transcription - extract audio and use the cached model
                result = cls.transcribe_video(file_path, model_name, options)
                transcription = result.get("text", "")
                segments = result.get("segments", [])

                if not transcription:
                    logger.warning(f"Failed to transcribe video file: {file_path}")
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
                        "source_type": "video",
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
        return ["openai-whisper", "pydub"]
