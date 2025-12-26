# vectorme

A command-line tool and HTTP server for speaker identification and diarization using ECAPA-TDNN, with built-in vector database.

<img width="400" alt="screenshot" src="https://github.com/user-attachments/assets/3e5a2459-4afe-4b0d-ac87-93a568d78ad5" />

## Features

- **Speaker Embeddings**: Extract 192-dimensional speaker embeddings
- **Speaker Database**: Store and query known speakers
- **Diarization**: Process audio in chunks to identify who is speaking when
- **HTTP Server**: REST API with streaming support
- **Format Support**: WAV, M4A, MP3, AAC, FLAC, OGG, and more (via ffmpeg)
- **GPU Acceleration**: CUDA and Metal (MPS) support

## Installation

```bash
cd vectorme
pip install -e .
```

Requires `ffmpeg` for audio format conversion:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## Download the model

```bash
vectorme --download-only
```

## Usage

### Basic Embedding Extraction

**Pipe audio from stdin:**
```bash
cat audio.wav | vectorme
```

**Read from file (supports wav, m4a, mp3, etc.):**
```bash
vectorme --file audio.m4a
```

### Vector Database

vectorme includes a built-in vector database for storing and querying speaker embeddings. The database is stored at `~/.vectorme/speakers.npz` by default.

**Add a speaker to the database:**
```bash
vectorme --file audio.wav --name "Doug"
# Added 'Doug' to database (1 total)
```

**Query for the closest match:**
```bash
vectorme --file unknown.wav
# {"matches": [{"name": "Doug", "similarity": 0.95}, ...], "best": {"name": "Doug", "similarity": 0.95}}
```

**List all speakers in the database:**
```bash
vectorme --list
# Doug
# Alice
# Bob
```

**Remove a speaker from the database:**
```bash
vectorme --remove "Doug"
# Removed 'Doug' from database
```

**Use a custom database location:**
```bash
vectorme --file audio.wav --db /path/to/custom.npz --name "Doug"
```

### Diarization

Process audio in chunks to identify speakers over time:

```bash
vectorme --file conversation.m4a --diarize
```

Output (NDJSON - one JSON object per line):
```json
{"event": "segment", "start": 0.0, "end": 1.0, "speaker": null, "vad_confidence": 0.82}
{"event": "segment", "start": 1.0, "end": 3.0, "speaker": "Doug", "vad_confidence": 0.91}
{"event": "segment", "start": 3.0, "end": 4.5, "speaker": null, "vad_confidence": 0.78}
{"event": "segment", "start": 4.5, "end": 8.0, "speaker": "Ayush", "vad_confidence": 0.85}
```

The `vad_confidence` field indicates the average speech probability for each segment (0.0-1.0). Higher values indicate higher confidence that the segment contains speech. This field is only included when VAD is enabled (default).

**Voice Activity Detection (VAD):**

VAD is enabled by default to filter out non-speech segments (silence, music, noise). This improves speaker identification accuracy by only processing chunks with detected speech.

```bash
# Default behavior - VAD enabled
vectorme --file audio.m4a --diarize

# Disable VAD
vectorme --file audio.m4a --diarize --no-vad

# Adjust VAD sensitivity (default: 0.5, range 0.0-1.0)
vectorme --file audio.m4a --diarize --vad-threshold 0.3
```

**Tuning parameters:**
```bash
vectorme --file audio.m4a --diarize \
  --chunk-size 3.0 \      # Chunk duration in seconds (default: 3.0, matches training)
  --chunk-hop 0.5 \       # Hop between chunks (default: 0.5)
  --threshold 0.5 \       # Minimum similarity to identify speaker (default: 0.5)
  --change-threshold 0.7 \# Threshold for speaker change detection (default: 0.7)
  --vad-threshold 0.5     # VAD speech probability threshold (default: 0.5)
```

### GPU Acceleration

Use `--gpu` to enable CUDA or Metal (MPS on Mac):

```bash
vectorme --file audio.m4a --gpu
# Using MPS acceleration
```

### Output Formats

When the database is empty, vectorme outputs the raw embedding:

**JSON (default):**
```bash
vectorme --file audio.wav --format json
# {"embedding": [0.123, -0.456, ...], "dimensions": 192}
```

**Space-separated list:**
```bash
vectorme --file audio.wav --format list
# 0.123 -0.456 0.789 ...
```

**NumPy binary:**
```bash
vectorme --file audio.wav --format numpy > embedding.npy
```

## HTTP Server

Run vectorme as an HTTP server for integration with other applications:

```bash
vectorme --serve
# Starting server on 127.0.0.1:3120
```

**Options:**
```bash
vectorme --serve --host 0.0.0.0 --port 8080 --gpu
```

### API Endpoints

#### Health Check
```bash
curl http://localhost:3120/health
# {"status": "ok", "speakers": 2}
```

#### List Speakers
```bash
curl http://localhost:3120/v1/speakers
# {"speakers": ["Ayush", "Doug"], "count": 2}
```

#### Add Speaker
```bash
curl -X POST http://localhost:3120/v1/speakers \
  -F "file=@voice_sample.m4a" \
  -F "name=Alice"
# {"message": "Added 'Alice' to database", "count": 3}
```

#### Remove Speaker
```bash
curl -X DELETE http://localhost:3120/v1/speakers/Alice
# {"message": "Removed 'Alice'", "count": 2}
```

#### Identify Speaker (single embedding)
```bash
curl -X POST http://localhost:3120/v1/audio/transcriptions \
  -F "file=@audio.m4a"
# {"matches": [{"name": "Doug", "similarity": 0.95}], "best": {"name": "Doug", "similarity": 0.95}}
```

#### Diarization (batch)
```bash
curl -X POST http://localhost:3120/v1/audio/transcriptions \
  -F "file=@conversation.m4a" \
  -F "response_format=diarized_json"
# {"duration": 59.07, "segments": [...], "speakers": ["Ayush", "Doug"]}
```

#### Diarization (streaming)
```bash
curl -N -X POST http://localhost:3120/v1/audio/transcriptions \
  -F "file=@conversation.m4a" \
  -F "response_format=diarized_json" \
  -F "stream=true"
```

Real-time NDJSON output:
```
{"event": "start", "duration": 59.07, "speakers": ["Ayush", "Doug"], "vad": true}
{"event": "speaker_change", "time": 0.0, "speaker": null, "similarity": 0.41, "vad_confidence": 0.85}
{"event": "segment", "start": 0.0, "end": 1.0, "speaker": null, "vad_confidence": 0.82}
{"event": "speaker_change", "time": 1.0, "speaker": "Doug", "similarity": 0.54, "vad_confidence": 0.91}
{"event": "segment", "start": 1.0, "end": 3.0, "speaker": "Doug", "vad_confidence": 0.88}
...
{"event": "done"}
```

**Event fields:**
- `segment.vad_confidence` - Average VAD speech probability for the segment (0.0-1.0, only when VAD enabled)
- `speaker_change.vad_confidence` - VAD confidence at the speaker change point (only when VAD enabled)

**Additional parameters:**
- `chunk_size` - Chunk duration in seconds (default: 3.0, matches ECAPA-TDNN training)
- `chunk_hop` - Hop between chunks (default: 0.5)
- `threshold` - Minimum similarity to identify speaker (default: 0.5)
- `change_threshold` - Speaker change detection threshold (default: 0.7)
- `filter_unknown=true` - Hide segments with unknown speakers
- `vad=false` - Disable Voice Activity Detection (enabled by default)
- `vad_threshold` - VAD speech probability threshold (default: 0.5)

## About ECAPA-TDNN

ECAPA-TDNN produces 192-dimensional speaker embeddings that can be used for:
- Speaker verification (is this the same person?)
- Speaker identification (who is speaking?)
- Speaker clustering (group audio by speaker)

The model is trained on VoxCeleb dataset and downloaded from SpeechBrain's model hub.

### Training Segment Size

The pretrained `speechbrain/spkrec-ecapa-voxceleb` model was trained on **fixed-length 3.0 second audio crops** at 16kHz (48,000 samples). This has important implications:

- **Chunk size**: We default to 3.0s chunks to match training conditions
- **Shorter segments**: Segments much shorter than 3s produce noisier, less stable embeddings
- **Longer segments**: The model supports variable-length input via attentive statistical pooling, but if multiple speakers are present in one segment, the embedding will be a "blend" of all speakers
- **Best practice for enrollment**: Use 3-10 seconds of single-speaker audio when adding speakers to the database

References:
- [SpeechBrain ECAPA recipe](https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml)
- [Model card](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
