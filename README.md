# vectorme

A simple command-line tool to extract speaker embeddings using ECAPA-TDNN.

## Installation

```bash
cd vectorme
pip install -e .
```

## Download the model

```bash
vectorme --download-only
```

## Usage

### Pipe audio from stdin:
```bash
cat audio.wav | vectorme
```

### Read from file:
```bash
vectorme --file audio.wav
```

### Output formats:

**JSON (default):**
```bash
cat audio.wav | vectorme --format json
# {"embedding": [0.123, -0.456, ...], "dimensions": 192}
```

**Space-separated list:**
```bash
cat audio.wav | vectorme --format list
# 0.123 -0.456 0.789 ...
```

**NumPy binary:**
```bash
cat audio.wav | vectorme --format numpy > embedding.npy
```

## About ECAPA-TDNN

ECAPA-TDNN produces 192-dimensional speaker embeddings that can be used for:
- Speaker verification (is this the same person?)
- Speaker identification (who is speaking?)
- Speaker clustering (group audio by speaker)

The model is trained on VoxCeleb dataset and downloaded from SpeechBrain's model hub.
