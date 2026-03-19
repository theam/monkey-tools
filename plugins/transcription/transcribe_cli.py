#!/usr/bin/env python3
"""CLI entry point for transcription — used by Cowork skill via Bash.

Usage:
    uv run python transcribe_cli.py <audio_file> [options]

Options:
    --language LANG         Language code (e.g., 'en', 'es'). Auto-detected if omitted.
    --skip-diarization      Skip speaker identification for faster processing.
    --num-speakers N        Exact number of speakers if known.
    --min-speakers N        Minimum expected speakers.
    --max-speakers N        Maximum expected speakers.
    --model REPO            MLX model override (e.g., 'mlx-community/whisper-large-v3-turbo').
    --export FORMAT         Export format: txt (default), json, srt.
"""

import argparse
import json
import sys

from transcription import export_transcript, format_transcript, transcribe


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with speaker diarization",
    )
    parser.add_argument("audio_file", help="Path to the audio file to transcribe")
    parser.add_argument("--language", default=None, help="Language code (e.g., 'en', 'es')")
    parser.add_argument("--skip-diarization", action="store_true", help="Skip speaker identification")
    parser.add_argument("--num-speakers", type=int, default=None, help="Exact number of speakers")
    parser.add_argument("--min-speakers", type=int, default=None, help="Minimum expected speakers")
    parser.add_argument("--max-speakers", type=int, default=None, help="Maximum expected speakers")
    parser.add_argument("--model", default=None, help="MLX model repo override")
    parser.add_argument("--export", default="txt", choices=["txt", "json", "srt"], help="Output format")

    args = parser.parse_args()

    try:
        result = transcribe(
            file_path=args.audio_file,
            language=args.language,
            num_speakers=args.num_speakers,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            skip_diarization=args.skip_diarization,
            model=args.model,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    # Print summary to stderr (so stdout is clean transcript)
    diarization_status = "skipped" if result["skip_diarization"] else f"{len(result['speakers'])} speakers"
    summary = (
        f"Transcription complete: {result['file']}\n"
        f"Duration: {result['duration_s']/60:.1f} min | "
        f"Processing time: {result['processing_time_s']:.0f}s "
        f"({result['duration_s']/result['processing_time_s']:.1f}x real-time) | "
        f"Backend: {result['backend']}\n"
        f"Language: {result['language']} | Diarization: {diarization_status}"
    )
    if result["speakers"]:
        summary += f" ({', '.join(result['speakers'])})"
    print(summary, file=sys.stderr)

    # Output transcript to stdout
    if args.export == "json":
        print(json.dumps({
            "file": result["file"],
            "language": result["language"],
            "duration_s": result["duration_s"],
            "segments": result["segments"],
        }, indent=2, ensure_ascii=False))
    elif args.export == "srt":
        print(export_transcript(result["segments"], format="srt"))
    else:
        print(format_transcript({"segments": result["segments"]}))


if __name__ == "__main__":
    main()
