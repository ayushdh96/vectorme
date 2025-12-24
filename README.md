# vectorme

A simple command-line tool to extract speaker embeddings using ECAPA-TDNN, with built-in vector database for speaker identification.

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

### Basic Embedding Extraction

**Pipe audio from stdin:**
```bash
cat audio.wav | vectorme
```

**Read from file:**
```bash
vectorme --file audio.wav
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
cat unknown.wav | vectorme
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

**Control number of matches returned:**
```bash
cat audio.wav | vectorme --top 3
```

### Output Formats

When the database is empty, vectorme outputs the raw embedding:

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
