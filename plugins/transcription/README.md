# Transcription MCP Plugin

Local-first audio transcription with speaker diarization for Claude Code and Claude Desktop. Runs entirely on your machine — no data leaves your computer.

Uses **MLX** on Apple Silicon for ~5x real-time GPU-accelerated transcription.

## Prerequisites

- **Apple Silicon Mac** (M1/M2/M3/M4) — required for GPU acceleration
- **Python 3.11+**
- **uv** — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

That's it. No API keys, no accounts, no cloud services needed.

## Install in Claude Code

```bash
claude --plugin-dir /path/to/plugins/transcription
```

Or for persistent use, add the plugin directory in Claude Code settings.

## Install in Claude Desktop

Add this to your Claude Desktop config file at `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "transcription": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/plugins/transcription", "python", "server.py"]
    }
  }
}
```

Replace `/absolute/path/to/plugins/transcription` with the actual path to this directory.

Restart Claude Desktop after adding the config.

## First Run

The first transcription will take a few extra minutes to:
1. Install Python dependencies (torch, whisperx, mlx-whisper) — ~2GB download, cached after first install
2. Download the Whisper large-v3 MLX model — ~3GB download, cached in `~/.cache/huggingface/`

Subsequent runs start immediately.

## Usage

Once installed, you can ask Claude to:

- **"List audio files in ~/Downloads"** — finds .m4a, .mp3, .wav, .flac, etc.
- **"Transcribe /path/to/recording.m4a"** — transcribes with speaker identification
- **"Transcribe recording.m4a with skip_diarization"** — faster, no speaker labels
- **"Export the transcription as SRT"** — exports to .txt, .json, or .srt
- **"Rename SPEAKER_00 to Jaime"** — assign names to identified speakers

## Tools

| Tool | Description |
|------|-------------|
| `list_audios` | List audio files in a directory |
| `transcribe_audio` | Transcribe with optional diarization |
| `list_transcriptions` | Show completed transcriptions |
| `get_transcription` | View a transcription |
| `set_speaker_name` | Name a speaker |
| `export_transcription` | Export to txt/json/srt |

## Performance

Tested on M3 Max (36GB):

| Mode | Speed | Notes |
|------|-------|-------|
| Transcription only (`skip_diarization=true`) | ~5.4x real-time | 75 min audio in ~14 min |
| With diarization | ~1.1x real-time | 75 min audio in ~67 min |

## Models

Default: `mlx-community/whisper-large-v3-mlx` (most accurate, 6.4% WER)

For faster processing at slightly lower accuracy, pass the model parameter:
`mlx-community/whisper-large-v3-turbo` (~7.75% WER, ~4.5x faster)

## Bundled Models

Speaker diarization models (pyannote) are bundled in `models/` — no HuggingFace account or token needed. These models are MIT licensed (see `LICENSES/` directory).
