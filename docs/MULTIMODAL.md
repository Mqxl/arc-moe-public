# Images, speech and sampled video

These are three different paths. Qwen3.8-Flash-Next's BF16 `mmproj` is a **vision** encoder, not an audio encoder. Do not advertise native audio/video request support in an OpenAI-compatible client: the current Strata API accepts text and images.

## Images

`./arc setup --vision cpu --context 131072 --gguf-dir /path/to/existing/GGUF` adopts existing model shards and downloads/builds the image encoder. CPU encoding leaves both GPUs available for the experts. Choose the context for the available memory; 128K is optional, not a global hardware assumption. The model configuration must contain the installer-generated `vision` block, and the engine must receive `--vision`. Verify with an actual image before setting OpenCode's model `attachment: true` and `modalities: {"input": ["text", "image"], "output": ["text"]}`.

## Speech and video bridge

`tools/media_mcp.py` is an optional local MCP server and command-line client. It uses ffmpeg/ffprobe, CPU int8 faster-whisper for speech, and the same authenticated Strata vision endpoint for questions. It runs on the **client computer**, so it can read that computer's media files. It does not start another chat model or use a GPU for speech recognition.

Install ffmpeg in PATH, create a Python virtual environment and run:

```sh
python -m pip install -r tools/media-requirements.txt
export ARC_BASE_URL=http://your-server:8798/v1
export ARC_API_KEY=your-existing-api-key
python tools/media_mcp.py /path/to/clip.mp4 --frames 6 --prompt "Describe the sequence and the speech."
```

Keep the API key in the environment, never in tracked files. Windows uses the same Python entry point and environment variables. `ARC_MODEL` optionally selects the API model ID. `ARC_STT_MODEL` selects a faster-whisper model, default `small`; its first use downloads the CPU speech weights.

For OpenCode, add a local MCP entry pointing to that environment's Python:

```json
{
  "mcp": {
    "arc-media": {
      "type": "local",
      "command": ["python", "/path/to/repository/tools/media_mcp.py", "--stdio"],
      "environment": {"ARC_BASE_URL": "http://your-server:8798/v1"},
      "enabled": true,
      "timeout": 900000
    }
  }
}
```

The process inherits `ARC_API_KEY`. Ask the assistant to use `arc-media` on a local file path or HTTP(S) URL. This also works with other editors that support local MCP tools. Native audio/video attachment handling remains unsupported; use the tool instead.

Limits: at most 16 evenly spaced video frames, images scaled within 768 pixels, first 600 seconds of speech, HTTP download limit 512 MiB. Speech transcription can be wrong; music, noises and speaker identity are not classified. Frame sampling can miss short events. The answer includes sampled timestamps and the transcript so these limits are reviewable. Converted copies live in a temporary directory and are removed after the request; original files are preserved.
