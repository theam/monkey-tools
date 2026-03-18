"""Transcription MCP Server — Local-first using WhisperX + pyannote."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from mcp.server.fastmcp import FastMCP

# Bundled pyannote models — no HuggingFace token needed
_PLUGIN_DIR = Path(__file__).parent
_MODELS_DIR = _PLUGIN_DIR / "models"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# Use MPS (Apple Silicon GPU) if available
if torch.backends.mps.is_available():
    DEVICE = "mps"
    COMPUTE_TYPE = "float16"

# Detect MLX availability (Apple Silicon GPU acceleration)
USE_MLX = False
try:
    import mlx_whisper
    USE_MLX = True
    print("MLX backend available — using Apple Silicon GPU for ASR")
except ImportError:
    print("MLX not available — using faster-whisper/CPU backend")

# MLX model defaults
MLX_MODEL = os.environ.get(
    "MLX_MODEL", "mlx-community/whisper-large-v3-mlx"
)

# State
_model = None
_align_model = None
_align_metadata = None
_diarize_pipeline = None
_transcriptions: dict[str, dict] = {}

mcp = FastMCP(
    "transcription",
    instructions=(
        "Audio transcription server with speaker diarization. "
        "Use transcribe_audio to transcribe files with speaker identification. "
        "Use list_audios to see available audio files. "
        "Use list_transcriptions to see completed transcriptions. "
        "Use get_transcription to view a specific transcription. "
        "Use set_speaker_name to assign names to identified speakers."
    ),
)


def _get_model():
    """Lazy-load the Whisper model (faster-whisper/CPU path only)."""
    global _model
    if _model is None:
        print("Loading Whisper large-v3-turbo model (CPU)...")
        _model = whisperx.load_model(
            "large-v3-turbo",
            device=DEVICE if DEVICE != "mps" else "cpu",
            compute_type=COMPUTE_TYPE if DEVICE != "mps" else "int8",
            language=None,
        )
        print("Model loaded.")
    return _model


def _transcribe_mlx(
    audio_path: str, language: str | None = None, model_repo: str | None = None,
) -> dict:
    """Transcribe using MLX backend (Apple Silicon GPU)."""
    repo = model_repo or MLX_MODEL
    print(f"Transcribing with MLX ({repo})...")
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
    )
    # Convert mlx-whisper output to WhisperX-compatible format
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        })
    detected_lang = result.get("language", language or "unknown")
    print(f"MLX transcription complete. Language: {detected_lang}")
    return {"segments": segments, "language": detected_lang}


def _get_diarize_pipeline():
    """Lazy-load the pyannote diarization pipeline."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        print("Loading pyannote diarization pipeline...")
        # Temporarily set offline mode + cache dir for pyannote only
        # (avoids blocking Whisper MLX model downloads)
        old_cache = os.environ.get("HF_HUB_CACHE")
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            if _MODELS_DIR.is_dir():
                os.environ["HF_HUB_CACHE"] = str(_MODELS_DIR)
                os.environ["HF_HUB_OFFLINE"] = "1"
            _diarize_pipeline = DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                device="cpu",
            )
        finally:
            # Restore original env
            if old_cache is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = old_cache
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
        print("Diarization pipeline loaded.")
    return _diarize_pipeline


def _format_transcript(result: dict, include_timestamps: bool = True) -> str:
    """Format a WhisperX result into readable text."""
    lines = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if include_timestamps:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            ts = f"[{_fmt_time(start)} - {_fmt_time(end)}]"
            lines.append(f"{ts} {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n\n".join(lines)


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _find_audio_files(directory: str) -> list[Path]:
    """Find audio files in a directory."""
    audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac", ".mp4"}
    path = Path(directory)
    if not path.is_dir():
        return []
    files = []
    for f in sorted(path.iterdir()):
        if f.suffix.lower() in audio_extensions and f.is_file():
            files.append(f)
    return files


@mcp.tool()
def list_audios(directory: str) -> str:
    """List audio files in a directory with their duration estimates.

    Args:
        directory: Path to directory containing audio files.
    """
    files = _find_audio_files(directory)
    if not files:
        return f"No audio files found in {directory}"

    lines = [f"Audio files in {directory}:\n"]
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines.append(f"{i}. {f.name} ({size_mb:.1f} MB)")

    lines.append(f"\nTotal: {len(files)} files")
    return "\n".join(lines)


@mcp.tool()
def transcribe_audio(
    file_path: str,
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    skip_diarization: bool = False,
    model: str | None = None,
) -> str:
    """Transcribe an audio file with optional speaker diarization.

    All processing happens locally on your machine. No data is sent to any cloud service.
    Uses MLX (Apple Silicon GPU) when available for ~5-10x speedup over CPU.

    Args:
        file_path: Path to the audio file to transcribe.
        language: Language code (e.g., 'en', 'es'). Auto-detected if not specified.
        num_speakers: Exact number of speakers if known.
        min_speakers: Minimum expected number of speakers.
        max_speakers: Maximum expected number of speakers.
        skip_diarization: Skip speaker identification for faster processing.
        model: MLX model override (e.g., 'mlx-community/whisper-large-v3-turbo' for speed).
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    file_key = str(path.resolve())
    start_time = time.time()

    # Override MLX model if requested
    mlx_model = model or MLX_MODEL

    # Step 1: Load audio
    print(f"Loading audio: {path.name}")
    audio = whisperx.load_audio(str(path))
    duration_s = len(audio) / 16000
    print(f"Audio loaded: {duration_s:.0f}s ({duration_s/60:.1f} min)")

    # Step 2: Transcribe (MLX GPU or faster-whisper CPU)
    if USE_MLX:
        result = _transcribe_mlx(str(path), language=language, model_repo=mlx_model)
        detected_lang = result["language"]
    else:
        print("Transcribing (CPU)...")
        whisper_model = _get_model()
        result = whisper_model.transcribe(
            audio,
            batch_size=16 if DEVICE != "mps" else 4,
            language=language,
        )
        detected_lang = result.get("language", language or "unknown")
        print(f"Transcription complete. Language: {detected_lang}")

    # Step 3: Align (word-level timestamps) — needed for diarization merge
    if not skip_diarization:
        # When diarizing, run alignment and diarization in parallel
        def _align():
            print("Aligning timestamps...")
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_lang,
                device="cpu",
            )
            aligned = whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device="cpu",
                return_char_alignments=False,
            )
            print("Alignment complete.")
            return aligned

        def _diarize():
            print("Running speaker diarization...")
            diarize_pipeline = _get_diarize_pipeline()
            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs["num_speakers"] = num_speakers
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_result = diarize_pipeline(str(path), **diarize_kwargs)
            print("Diarization complete.")
            return diarize_result

        # Run alignment and diarization concurrently — both release the GIL
        with ThreadPoolExecutor(max_workers=2) as executor:
            align_future = executor.submit(_align)
            diarize_future = executor.submit(_diarize)
            aligned_result = align_future.result()
            diarize_segments = diarize_future.result()

        result = whisperx.assign_word_speakers(diarize_segments, aligned_result)

    elapsed = time.time() - start_time

    # Store result
    _transcriptions[file_key] = {
        "file": path.name,
        "path": file_key,
        "language": detected_lang,
        "duration_s": duration_s,
        "processing_time_s": elapsed,
        "segments": result["segments"],
        "speaker_names": {},
    }

    # Build summary
    speakers = set()
    for seg in result["segments"]:
        if "speaker" in seg:
            speakers.add(seg["speaker"])

    transcript_text = _format_transcript(result)

    backend = f"MLX ({mlx_model.split('/')[-1]})" if USE_MLX else "CPU"
    diarization_status = "skipped" if skip_diarization else f"{len(speakers)} speakers"

    summary = (
        f"Transcription complete: {path.name}\n"
        f"Duration: {duration_s/60:.1f} min | "
        f"Processing time: {elapsed:.0f}s ({duration_s/elapsed:.1f}x real-time) | "
        f"Backend: {backend}\n"
        f"Language: {detected_lang} | "
        f"Diarization: {diarization_status}"
    )
    if speakers:
        summary += f" ({', '.join(sorted(speakers))})"
    summary += (
        f"\nSegments: {len(result['segments'])}\n\n"
        f"--- Transcript ---\n\n{transcript_text}"
    )

    return summary


@mcp.tool()
def list_transcriptions() -> str:
    """List all completed transcriptions in this session."""
    if not _transcriptions:
        return "No transcriptions yet. Use transcribe_audio to transcribe a file."

    lines = ["Completed transcriptions:\n"]
    for key, t in _transcriptions.items():
        speakers = set()
        for seg in t["segments"]:
            if "speaker" in seg:
                speakers.add(seg["speaker"])
        names = t.get("speaker_names", {})
        speaker_info = []
        for s in sorted(speakers):
            name = names.get(s, s)
            speaker_info.append(f"{s}={name}" if s != name else s)

        lines.append(
            f"- {t['file']} ({t['duration_s']/60:.1f} min, "
            f"{t['language']}, {len(speakers)} speakers: {', '.join(speaker_info)})"
        )
    return "\n".join(lines)


@mcp.tool()
def get_transcription(
    file_path: str,
    include_timestamps: bool = True,
    speaker_filter: str | None = None,
) -> str:
    """Get the full transcription text for a previously transcribed file.

    Args:
        file_path: Path to the audio file (as used in transcribe_audio).
        include_timestamps: Whether to include timestamps in output.
        speaker_filter: Only show segments from this speaker (e.g., 'SPEAKER_00').
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}. Use transcribe_audio first."

    t = _transcriptions[key]
    result = {"segments": t["segments"]}
    names = t.get("speaker_names", {})

    # Apply speaker name substitutions
    for seg in result["segments"]:
        speaker_id = seg.get("speaker", "Unknown")
        if speaker_id in names:
            seg["speaker"] = names[speaker_id]

    # Filter by speaker if requested
    if speaker_filter:
        result["segments"] = [
            s for s in result["segments"]
            if s.get("speaker", "") == speaker_filter
        ]

    return _format_transcript(result, include_timestamps=include_timestamps)


@mcp.tool()
def set_speaker_name(file_path: str, speaker_id: str, name: str) -> str:
    """Assign a human-readable name to an identified speaker.

    Args:
        file_path: Path to the transcribed audio file.
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') from the diarization.
        name: The human-readable name to assign (e.g., 'Jaime').
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}"

    _transcriptions[key]["speaker_names"][speaker_id] = name
    return f"Speaker {speaker_id} renamed to '{name}' in {Path(file_path).name}"


@mcp.tool()
def export_transcription(
    file_path: str,
    output_path: str | None = None,
    format: str = "txt",
) -> str:
    """Export a transcription to a file.

    Args:
        file_path: Path to the transcribed audio file.
        output_path: Where to save the export. Defaults to same directory as audio file.
        format: Export format — 'txt', 'json', or 'srt'.
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}"

    t = _transcriptions[key]
    names = t.get("speaker_names", {})

    if output_path is None:
        output_path = str(path.with_suffix(f".{format}"))

    out = Path(output_path)

    if format == "json":
        export_data = {
            "file": t["file"],
            "language": t["language"],
            "duration_s": t["duration_s"],
            "speaker_names": names,
            "segments": t["segments"],
        }
        out.write_text(json.dumps(export_data, indent=2, ensure_ascii=False))

    elif format == "srt":
        lines = []
        for i, seg in enumerate(t["segments"], 1):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("speaker", "Unknown")
            speaker = names.get(speaker, speaker)
            text = seg.get("text", "").strip()
            lines.append(str(i))
            lines.append(f"{_srt_time(start)} --> {_srt_time(end)}")
            lines.append(f"[{speaker}] {text}")
            lines.append("")
        out.write_text("\n".join(lines))

    else:  # txt
        segments = t["segments"]
        for seg in segments:
            sid = seg.get("speaker", "Unknown")
            if sid in names:
                seg["speaker"] = names[sid]
        out.write_text(_format_transcript({"segments": segments}))

    return f"Exported to {out}"


def _srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
