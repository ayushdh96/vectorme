# vectorme

A command-line tool and HTTP server for speaker identification and diarization using ECAPA-TDNN, with built-in vector database.

<img width="400" alt="screenshot" src="https://github.com/user-attachments/assets/3e5a2459-4afe-4b0d-ac87-93a568d78ad5" />

Watch a video: https://youtu.be/-kvUzvcfD6o


## Features

- **Speaker Embeddings**: Extract 192-dimensional speaker embeddings using ECAPA-TDNN
- **Speaker Database**: Store and query known speakers with vector similarity search
- **Real-time Diarization**: Live streaming speaker detection with confidence scores
- **TS-VAD Refinement**: NeMo MSDD-based batch diarization with precise speaker boundaries
- **Web UI**: React-based voice recorder with live waveform visualization and speaker timeline
- **HTTP Server**: REST API with OpenAI-compatible endpoints and streaming support
- **Format Support**: WAV, M4A, MP3, AAC, FLAC, OGG, WebM, and more (via ffmpeg)
- **GPU Acceleration**: CUDA and Metal (MPS) support
- **CPU Compatibility**: Automatic tensor layout fixes for CPU-based NeMo MSDD processing

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

Run vectorme as an HTTP server with web UI for voice recording and speaker analysis:

```bash
vectorme --serve
# Starting server on 127.0.0.1:3120
# Open http://localhost:3120 in your browser
```

**Options:**
```bash
vectorme --serve --host 0.0.0.0 --port 8080 --gpu
```

### Web UI

The web interface provides:
- **Real-time recording** with live waveform visualization
- **Streaming diarization** with instant speaker detection
- **Interactive audio timeline** showing speaker segments
- **Drag-to-select** audio regions for focused analysis
- **Speaker labeling** interface for naming unknown speakers
- **Saved recordings** library with shareable links
- **Multi-scale embeddings** with confidence scores

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
- `segment.vad_confidence` - A (NeMo MSDD)

TS-VAD uses NeMo's Multi-Scale Diarization Decoder (MSDD) for advanced batch diarization with precise speaker boundaries and intelligent unknown speaker clustering
**Additional parameters:**
- `chunk_size` - Chunk duration in seconds (default: 3.0, matches ECAPA-TDNN training)
- `chunk_hop` - Hop between chunks (default: 0.5)
- `threshold` - Minimum similarity to identify speaker (default: 0.5)
- `change_threshold` - Speaker change detection threshold (default: 0.7)
- `filter_unknown=true` - Hide segments with unknown speakers
- `vad=false` - Disable Voice Activity Detection (enabled by default)
- `vad_threshold` - VAD speech probability threshold (default: 0.5)

### TS-VAD Refined Diarization (NeMo MSDD)

TS-VAD uses NeMo's NeuralDiarizer with Multi-Scale Diarization Decoder (MSDD) for advanced batch diarization with precise speaker boundaries and intelligent unknown speaker clustering.

**NeMo Integration:**
- **NeuralDiarizer**: NVIDIA NeMo's end-to-end diarization pipeline
- **MSDD Model**: Multi-scale temporal modeling with `diar_msdd_telephonic` weights
- **VAD**: MarbleNet voice activity detection (`vad_marblenet`)
- **Embeddings**: TitaNet/ECAPA-TDNN speaker embeddings with multi-scale windows
- **CPU Compatibility**: Automatic tensor layout fixes via `.reshape()` fallback for CPU execution

**Key differences from streaming diarization:**
- **Batch processing**: Analyzes the entire audio file instead of streaming chunks
- **Multi-scale embeddings**: 5 temporal scales (0.5s to 1.5s windows) for robust speaker modeling
- **Unknown speaker clustering**: Groups unidentified segments into `unknown_1`, `unknown_2`, etc. based on voice similarity
- **Precise timestamps**: MSDD refinement provides more accurate speaker change boundaries

**Enable TS-VAD mode:**
```bash
curl -X POST http://localhost:3120/v1/audio/transcriptions \
  -F "file=@conversation.m4a" \
  -F "response_format=diarized_json" \
  -F "diarization_mode=ts_vad"
```

**TS-VAD-specific parameters:**
- `diarization_mode=ts_vad` - Enable TS-VAD refinement (default: `streaming`)
- `window_size` - Analysis window duration in seconds (default: 2.0)
- `window_hop` - Hop between windows in seconds (default: 0.5)
- `unknown_assign_threshold` - Similarity threshold for grouping unknown speakers (default: 0.60)
- `min_segment_duration` - Minimum segment length in seconds (default: 0.5)

**TS-VAD Response:**
```json
{
  "mode": "ts_vad",
  "duration": 59.07,
  "segments": [
    {"start": 0.0, "end": 3.2, "speaker": "Ayush", "similarity": 0.78, "vad_confidence": 0.91},
    {"start": 3.2, "end": 7.5, "speaker": "unknown_1", "similarity": 0.42, "vad_confidence": 0.85},
    {"start": 7.5, "end": 12.1, "speaker": "Doug", "similarity": 0.82, "vad_confidence": 0.89}
  ],
  "known_speakers": ["Ayush", "Doug"],
  "unknown_speakers": ["unknown_1"],
  "total_segments": 3
}
```

**Unknown speaker handling:**
- Segments with similarity below `threshold` (0.5) are marked as unknown
- Unknown segments are clustered by voice similarity
- Each cluster gets a unique ID: `unknown_1`, `unknown_2`, etc.
- Similar-sounding unknowns are grouped together (controlled by `unknown_assign_threshold`)

**Web UI features:**
- Real-time speaker detection during recording with live waveform visualization
- Confidence scores for each speaker identification
- Color-coded similarity indicators (green ≥70%, yellow 50-69%, red <50%)
- Speaker labeling interface for unknown speakers
- Drag-to-select audio regions for focused analysis

**CLI example with all parameters:**
```bash
curl -X POST http://localhost:3120/v1/audio/transcriptions \
  -F "file=@meeting.m4a" \
  -F "response_format=diarized_json" \
  -F "diarization_mode=ts_vad" \
  -F "window_size=2.0" \
  -F "window_hop=0.5" \
  -F "unknown_assign_threshold=0.60" \
  -F "vad=true" \
  -F "threshold=0.5"
```

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
## Frontend Architecture

The web UI is built with a modular, no-build-step architecture for rapid development:

### File Structure
- **`static/index.html`** (26 lines) - Minimal HTML shell with CDN imports
- **`static/styles.css`** (692 lines) - Complete styling with responsive design and SASS-ready structure
- **`static/app.js`** (1767 lines) - React component with JSX transpiled in-browser via Babel

### Technology Stack
- **React 18** - Loaded via unpkg CDN, no build step required
- **WaveSurfer.js 7** - Audio waveform visualization and region selection
- **Babel Standalone** - Client-side JSX transformation for rapid iteration
- **MediaRecorder API** - WebM/Opus recording with 1-second chunks for streaming
- **Fetch Streaming** - NDJSON event stream processing for real-time updates

### Key Features
- **Zero build complexity**: Edit app.js and refresh - changes appear instantly
- **Semantic CSS**: All elements use meaningful class names (`.segment-speaker`, `.timeline-info`)
- **Modular separation**: HTML structure, styling, and logic cleanly separated
- **ARIA accessibility**: Full screen reader support with proper labeling
- **Responsive design**: Mobile-optimized with breakpoints at 768px and 480px

### Development Workflow
1. Edit `static/app.js` for logic changes
2. Edit `static/styles.css` for styling
3. Refresh browser - no build required
4. Changes persist immediately with Flask auto-reload

This architecture enables rapid prototyping while maintaining production-ready code organization.