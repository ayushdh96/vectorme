#!/usr/bin/env python3
"""
vectorme - Extract speaker embeddings using ECAPA-TDNN

Usage:
    cat audio.wav | vectorme                    # Output embedding JSON
    vectorme --file audio.wav                   # Output embedding JSON
    vectorme --file audio.wav --name "Doug"     # Add to vector DB
    cat audio.wav | vectorme                    # Query DB for closest match
    vectorme --list                             # List all names in DB
    vectorme --remove "Doug"                    # Remove from DB
"""

import sys
import io
import argparse
import json
import warnings
import os
from pathlib import Path
import numpy as np

# Suppress noisy warnings from dependencies
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Default database location
DEFAULT_DB_PATH = Path.home() / ".vectorme" / "speakers.npz"


class VectorDB:
    """Minimal vector database for speaker embeddings."""
    
    def __init__(self, db_path=None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.names = []
        self.embeddings = None
        self._load()
    
    def _load(self):
        """Load database from disk."""
        if self.db_path.exists():
            data = np.load(self.db_path, allow_pickle=True)
            self.names = data["names"].tolist()
            self.embeddings = data["embeddings"]
        else:
            self.names = []
            self.embeddings = None
    
    def _save(self):
        """Save database to disk."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.savez(self.db_path, names=np.array(self.names), embeddings=self.embeddings)
    
    def add(self, name, embedding):
        """Add or update a speaker embedding."""
        embedding = np.array(embedding).reshape(1, -1)
        
        if name in self.names:
            # Update existing
            idx = self.names.index(name)
            self.embeddings[idx] = embedding
        else:
            # Add new
            self.names.append(name)
            if self.embeddings is None:
                self.embeddings = embedding
            else:
                self.embeddings = np.vstack([self.embeddings, embedding])
        
        self._save()
    
    def remove(self, name):
        """Remove a speaker from the database."""
        if name not in self.names:
            return False
        
        idx = self.names.index(name)
        self.names.pop(idx)
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        
        if len(self.names) == 0:
            self.embeddings = None
        
        self._save()
        return True
    
    def query(self, embedding, top_k=5):
        """Find closest matches using cosine similarity."""
        if self.embeddings is None or len(self.names) == 0:
            return []
        
        embedding = np.array(embedding).reshape(1, -1)
        
        # Cosine similarity
        norm_query = embedding / np.linalg.norm(embedding)
        norm_db = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = np.dot(norm_db, norm_query.T).flatten()
        
        # Sort by similarity (descending)
        indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in indices:
            results.append({
                "name": self.names[idx],
                "similarity": float(similarities[idx])
            })
        
        return results
    
    def list_names(self):
        """List all names in the database."""
        return sorted(self.names)
    
    def __len__(self):
        return len(self.names)


def get_embedding(args, classifier, torchaudio, torch):
    """Extract embedding from audio file or stdin."""
    if args.file:
        waveform, sample_rate = torchaudio.load(args.file)
    else:
        if sys.stdin.isatty():
            return None
        audio_bytes = sys.stdin.buffer.read()
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract embedding
    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
        embedding = embedding.squeeze().cpu().numpy()
    
    return embedding


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embedding vector using ECAPA-TDNN"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to audio file (alternative to stdin)"
    )
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Name to associate with this voice (adds to DB)"
    )
    parser.add_argument(
        "--format", "-o",
        choices=["json", "numpy", "list"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--top", "-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)"
    )
    parser.add_argument(
        "--db",
        type=str,
        help="Path to vector database file"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all names in the database"
    )
    parser.add_argument(
        "--remove", "-r",
        type=str,
        help="Remove a name from the database"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model, don't process audio"
    )
    args = parser.parse_args()

    # Initialize database
    db = VectorDB(args.db)

    # Handle --list
    if args.list:
        names = db.list_names()
        if names:
            for name in names:
                print(name)
        else:
            print("Database is empty", file=sys.stderr)
        return

    # Handle --remove
    if args.remove:
        if db.remove(args.remove):
            print(f"Removed '{args.remove}' from database", file=sys.stderr)
        else:
            print(f"'{args.remove}' not found in database", file=sys.stderr)
            sys.exit(1)
        return

    # Import heavy libraries only when needed
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    # Load the ECAPA-TDNN model (downloads on first use)
    model_dir = "~/.cache/speechbrain/spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": "cpu"}
    )

    if args.download_only:
        print("Model downloaded successfully!", file=sys.stderr)
        return

    # Get embedding from audio
    embedding = get_embedding(args, classifier, torchaudio, torch)
    
    if embedding is None:
        print("Error: No audio input. Pipe audio or use --file", file=sys.stderr)
        print("Usage: cat audio.wav | vectorme", file=sys.stderr)
        sys.exit(1)

    # If --name provided, add to database
    if args.name:
        db.add(args.name, embedding)
        print(f"Added '{args.name}' to database ({len(db)} total)", file=sys.stderr)
        return

    # If database has entries, query for matches
    if len(db) > 0:
        results = db.query(embedding, top_k=args.top)
        output = {
            "matches": results,
            "best": results[0] if results else None
        }
        print(json.dumps(output))
    else:
        # No database, just output the embedding
        if args.format == "json":
            output = {
                "embedding": embedding.tolist(),
                "dimensions": len(embedding)
            }
            print(json.dumps(output))
        elif args.format == "list":
            print(" ".join(map(str, embedding)))
        elif args.format == "numpy":
            np.save(sys.stdout.buffer, embedding)


if __name__ == "__main__":
    main()
