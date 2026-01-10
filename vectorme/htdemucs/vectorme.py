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
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import yaml

# Suppress noisy warnings from dependencies
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Default database location
DEFAULT_DB_PATH = Path.home() / ".vectorme" / "speakers.npz"

# Default recordings directory
DEFAULT_RECORDINGS_PATH = Path.home() / ".vectorme" / "recordings"

# Default LR model location (trained on stored embeddings)
DEFAULT_LR_MODEL_PATH = Path.home() / ".vectorme" / "speaker_lr.npz"


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=np.float32)
    x = x - np.max(x)
    ex = np.exp(x)
    s = np.sum(ex)
    return ex / s if s > 0 else ex


class SpeakerLRModel:
    """
    Lightweight multinomial logistic regression on top of ECAPA embeddings.
    Pure NumPy implementation (no sklearn dependency).
    """

    def __init__(self, classes=None, W=None, b=None):
        self.classes = classes or []
        self.W = W
        self.b = b

    @property
    def is_trained(self) -> bool:
        return self.W is not None and self.b is not None and len(self.classes) > 0

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, classes=np.array(self.classes, dtype=object), W=self.W, b=self.b)

    @classmethod
    def load(cls, path: Path) -> "SpeakerLRModel":
        path = Path(path)
        if not path.exists():
            return cls()
        data = np.load(path, allow_pickle=True)
        return cls(
            classes=data["classes"].tolist(),
            W=data["W"].astype(np.float32),
            b=data["b"].astype(np.float32),
        )

    @staticmethod
    def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float32)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.where(n == 0, 1.0, n)
        return X / n

    @staticmethod
    def _l2_normalize_vec(x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float32).flatten()
        n = np.linalg.norm(x)
        return x / n if n > 0 else x

    def train_from_db(
        self,
        db: "VectorDB",
        min_samples_per_class: int = 5,
        lr: float = 0.5,
        epochs: int = 400,
        reg: float = 1e-4,
        seed: int = 1337,
    ) -> Dict[str, Any]:
        if db.embeddings is None or len(db.names) == 0:
            raise ValueError("VectorDB is empty; add speakers before training.")

        counts = db.count_by_name()
        eligible = sorted([n for n, c in counts.items() if c >= min_samples_per_class])
        if len(eligible) < 2:
            raise ValueError(
                f"Need at least 2 speakers with ≥{min_samples_per_class} samples each. Eligible: {eligible}"
            )

        X_list, y_list = [], []
        for i, name in enumerate(db.names):
            if name not in eligible:
                continue
            X_list.append(db.embeddings[i])
            y_list.append(eligible.index(name))

        X = np.vstack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.int64)

        # Normalize like cosine world
        X = self._l2_normalize_rows(X)

        N, D = X.shape
        K = len(eligible)

        rng = np.random.default_rng(seed)
        W = (0.01 * rng.standard_normal((K, D))).astype(np.float32)
        b = np.zeros((K,), dtype=np.float32)

        # one-hot
        Y = np.zeros((N, K), dtype=np.float32)
        Y[np.arange(N), y] = 1.0

        for _ in range(int(epochs)):
            logits = (X @ W.T) + b
            logits = logits - np.max(logits, axis=1, keepdims=True)
            exp = np.exp(logits)
            P = exp / np.sum(exp, axis=1, keepdims=True)

            dZ = (P - Y) / float(N)
            dW = dZ.T @ X + reg * W
            dbias = np.sum(dZ, axis=0)

            W -= lr * dW
            b -= lr * dbias

        self.classes = eligible
        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)

        # quick sanity train accuracy
        logits = (X @ self.W.T) + self.b
        preds = np.argmax(logits, axis=1)
        acc = float(np.mean(preds == y))

        return {
            "num_samples": int(N),
            "num_classes": int(K),
            "classes": eligible,
            "train_accuracy": round(acc, 4),
        }

    def predict(self, embedding: np.ndarray) -> Dict[str, Any]:
        if not self.is_trained:
            return {"name": None, "top_prob": 0.0, "margin": 0.0, "probs": {}}

        x = self._l2_normalize_vec(embedding)
        logits = (self.W @ x) + self.b
        probs = _softmax(logits)

        order = np.argsort(probs)[::-1]
        top_i = int(order[0])
        top2_i = int(order[1]) if len(order) > 1 else top_i

        top_prob = float(probs[top_i])
        top2_prob = float(probs[top2_i]) if len(order) > 1 else 0.0
        margin = float(top_prob - top2_prob)

        probs_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        return {"name": self.classes[top_i], "top_prob": top_prob, "margin": margin, "probs": probs_dict}


def load_config() -> dict:
    """Load configuration from config.yaml. Raises FileNotFoundError if missing."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "The config.yaml file is required for VectorMe to run."
        )
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration at module level
CONFIG = load_config()


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
        """Add a speaker embedding (appends if name exists, creating multiple embeddings per speaker)."""
        embedding = np.array(embedding).reshape(1, -1)
        
        # Always append - allows multiple embeddings per speaker for better matching
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
        """List all unique names in the database."""
        return sorted(set(self.names))
    
    def count_by_name(self):
        """Return dict of name -> count of embeddings."""
        from collections import Counter
        return dict(Counter(self.names))
    
    def get_embeddings_by_name(self, name):
        """Get all embeddings for a given name."""
        indices = [i for i, n in enumerate(self.names) if n == name]
        if not indices or self.embeddings is None:
            return []
        return [self.embeddings[i] for i in indices]
    
    def __len__(self):
        return len(self.names)


# Audio formats that need ffmpeg conversion
FFMPEG_FORMATS = {'.m4a', '.mp3', '.aac', '.ogg', '.wma', '.flac', '.opus', '.webm', '.mp4'}


def convert_to_wav(input_path):
    """Convert audio file to 16kHz mono WAV using ffmpeg. Returns temp file path."""
    suffix = Path(input_path).suffix.lower()
    if suffix not in FFMPEG_FORMATS:
        return None  # No conversion needed
    
    # Create temp file for WAV output
    temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(temp_fd)
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', '-y', temp_path],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            os.unlink(temp_path)
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        return temp_path
    except FileNotFoundError:
        os.unlink(temp_path)
        raise RuntimeError("ffmpeg not found. Install ffmpeg to process m4a/mp3 files.")


# ------------------ Demucs helper (TS-VAD only, backend) ------------------

def maybe_extract_vocals_with_demucs(wav_path: str, enable: bool, device: str = "cpu") -> Tuple[str, Optional[str]]:
    """Best-effort Demucs vocal separation.

    Returns (wav_path_to_use, demucs_temp_dir).
    - If enable is False -> (wav_path, None)
    - If Demucs fails / not installed -> (wav_path, None)
    - If successful -> (<temp>/htdemucs/<track>/vocals.wav, <temp_dir>)

    The caller should delete demucs_temp_dir when done.
    """
    if not enable:
        return wav_path, None

    try:
        out_dir = tempfile.mkdtemp(prefix="demucs_outputs_")
        track_name = Path(wav_path).stem

        cmd = [
            sys.executable,
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems=vocals",
            wav_path,
            "-o",
            out_dir,
        ]

        # demucs expects --device cpu|cuda
        if device and str(device).lower().startswith("cuda"):
            cmd += ["--device", "cuda"]
        else:
            cmd += ["--device", "cpu"]

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            try:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            return wav_path, None

        vocals_path = Path(out_dir) / "htdemucs" / track_name / "vocals.wav"
        if not vocals_path.exists():
            try:
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            return wav_path, None

        return str(vocals_path), out_dir
    except Exception:
        return wav_path, None


def get_embedding(args, classifier, torchaudio, torch):
    """Extract embedding from audio file or stdin."""
    temp_wav = None
    try:
        if args.file:
            # Check if conversion is needed
            temp_wav = convert_to_wav(args.file)
            audio_path = temp_wav if temp_wav else args.file
            waveform, sample_rate = torchaudio.load(audio_path)
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
    finally:
        # Clean up temp file
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SileroVAD:
    """Silero Voice Activity Detection wrapper."""
    
    def __init__(self, torch, threshold=0.5):
        self.torch = torch
        self.threshold = threshold
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.get_speech_timestamps = self.utils[0]
    
    def has_speech(self, waveform, sample_rate=16000):
        """Check if waveform contains speech above threshold."""
        # Ensure correct shape (1D tensor)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            waveform,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.threshold
        )
        
        return len(speech_timestamps) > 0
    
    def get_speech_ratio(self, waveform, sample_rate=16000):
        """Get ratio of speech to total audio duration."""
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        speech_timestamps = self.get_speech_timestamps(
            waveform,
            self.model,
            sampling_rate=sample_rate,
            threshold=self.threshold
        )
        
        if not speech_timestamps:
            return 0.0
        
        total_speech = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
        return total_speech / len(waveform)
    
    def get_speech_confidence(self, waveform, sample_rate=16000):
        """Get speech confidence (probability) for the waveform.
        
        Returns tuple: (has_speech: bool, confidence: float, speech_ratio: float)
        - has_speech: True if speech detected above threshold
        - confidence: Average speech probability across the waveform (0.0-1.0)
        - speech_ratio: Ratio of speech frames to total frames (0.0-1.0)
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)
        
        # Reset model state for clean inference
        self.model.reset_states()
        
        # Get frame-level probabilities using windowed approach
        # Silero VAD works on 512 sample windows at 16kHz (32ms)
        window_size = 512
        probabilities = []
        
        for i in range(0, len(waveform) - window_size + 1, window_size):
            chunk = waveform[i:i + window_size]
            prob = self.model(chunk, sample_rate).item()
            probabilities.append(prob)
        
        if not probabilities:
            return (False, 0.0, 0.0)
        
        # Calculate metrics
        avg_confidence = sum(probabilities) / len(probabilities)
        speech_frames = sum(1 for p in probabilities if p >= self.threshold)
        speech_ratio = speech_frames / len(probabilities)
        has_speech = speech_ratio > 0.1  # At least 10% of frames have speech
        
        return (has_speech, avg_confidence, speech_ratio)


def process_chunks(args, classifier, torchaudio, torch, db, vad=None):
    """Process audio in chunks and emit speaker change events."""
    temp_wav = None
    try:
        if args.file:
            temp_wav = convert_to_wav(args.file)
            audio_path = temp_wav if temp_wav else args.file
            waveform, sample_rate = torchaudio.load(audio_path)
        else:
            if sys.stdin.isatty():
                return
            audio_bytes = sys.stdin.buffer.read()
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Chunk parameters
        chunk_duration = args.chunk_size  # seconds
        chunk_samples = int(chunk_duration * sample_rate)
        hop_duration = args.chunk_hop  # seconds
        hop_samples = int(hop_duration * sample_rate)
        
        total_samples = waveform.shape[1]
        
        prev_embedding = None
        prev_speaker = None
        prev_vad_confidence = None
        segment_start = 0.0
        
        position = 0
        while position < total_samples:
            end_pos = min(position + chunk_samples, total_samples)
            chunk = waveform[:, position:end_pos]
            
            # Skip chunks that are too short
            if chunk.shape[1] < sample_rate * 0.5:  # min 0.5 seconds
                break
            
            current_time = position / sample_rate
            
            # VAD check - get speech confidence
            vad_confidence = None
            if vad is not None:
                has_speech, vad_confidence, vad_speech_ratio = vad.get_speech_confidence(chunk, sample_rate)
                if not has_speech:
                    # No speech detected - treat as silence
                    if prev_speaker is not None:
                        # Emit previous segment before silence
                        if segment_start is not None and segment_start < current_time:
                            event = {
                                "event": "segment",
                                "start": round(segment_start, 2),
                                "end": round(current_time, 2),
                                "speaker": prev_speaker
                            }
                            if prev_vad_confidence is not None:
                                event["vad_confidence"] = round(prev_vad_confidence, 3)
                            print(json.dumps(event))
                        prev_speaker = None
                        prev_embedding = None
                        prev_vad_confidence = None
                    # Mark that we're in silence
                    segment_start = None
                    position += hop_samples
                    continue
            
            # If we were in silence and speech resumed, update segment_start
            if segment_start is None:
                segment_start = current_time
                prev_vad_confidence = vad_confidence
            
            # Extract embedding for this chunk
            with torch.no_grad():
                embedding = classifier.encode_batch(chunk)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Query database for speaker
            if len(db) > 0:
                results = db.query(embedding, top_k=1)
                current_speaker = results[0]["name"] if results and results[0]["similarity"] > args.threshold else None
                current_similarity = results[0]["similarity"] if results else 0.0
            else:
                current_speaker = None
                current_similarity = 0.0
            
            # Check for speaker change
            speaker_changed = False
            if prev_embedding is not None:
                similarity = cosine_similarity(prev_embedding, embedding)
                # Speaker changed if embedding similarity drops below threshold
                # or if the identified speaker is different
                if similarity < args.change_threshold or current_speaker != prev_speaker:
                    speaker_changed = True
            else:
                speaker_changed = True  # First chunk
            
            if speaker_changed:
                # Emit previous segment
                if segment_start is not None and segment_start < current_time:
                    event = {
                        "event": "segment",
                        "start": round(segment_start, 2),
                        "end": round(current_time, 2),
                        "speaker": prev_speaker
                    }
                    if prev_vad_confidence is not None:
                        event["vad_confidence"] = round(prev_vad_confidence, 3)
                    print(json.dumps(event))
                
                segment_start = current_time
                prev_speaker = current_speaker
                prev_vad_confidence = vad_confidence
                
                # Emit speaker change event
                event = {
                    "event": "speaker_change",
                    "time": round(current_time, 2),
                    "speaker": current_speaker,
                    "similarity": round(current_similarity, 3)
                }
                if vad_confidence is not None:
                    event["vad_confidence"] = round(vad_confidence, 3)
                print(json.dumps(event))
            else:
                # Track running average of VAD confidence for the segment
                if vad_confidence is not None and prev_vad_confidence is not None:
                    prev_vad_confidence = (prev_vad_confidence + vad_confidence) / 2
                elif vad_confidence is not None:
                    prev_vad_confidence = vad_confidence
            
            prev_embedding = embedding
            position += hop_samples
        
        # Emit final segment
        final_time = total_samples / sample_rate
        if segment_start is not None and segment_start < final_time:
            event = {
                "event": "segment",
                "start": round(segment_start, 2),
                "end": round(final_time, 2),
                "speaker": prev_speaker
            }
            if prev_vad_confidence is not None:
                event["vad_confidence"] = round(prev_vad_confidence, 3)
            print(json.dumps(event))
        
    finally:
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)


def diarize_audio_bytes(audio_bytes, classifier, torchaudio, torch, db,
                        chunk_size=3.0, chunk_hop=0.5, threshold=0.5, change_threshold=0.7,
                        filter_unknown=False):
    """Process audio bytes and return diarization results."""
    temp_input = None
    temp_wav = None
    events = []
    
    try:
        # Write bytes to temp file
        temp_fd, temp_input = tempfile.mkstemp()
        os.write(temp_fd, audio_bytes)
        os.close(temp_fd)
        
        # Always convert through ffmpeg since we don't know the format
        temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
        os.close(temp_fd2)
        
        result = subprocess.run(
            ['ffmpeg', '-i', temp_input, '-ar', '16000', '-ac', '1', '-y', temp_wav],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        
        waveform, sample_rate = torchaudio.load(temp_wav)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        chunk_samples = int(chunk_size * sample_rate)
        hop_samples = int(chunk_hop * sample_rate)
        total_samples = waveform.shape[1]
        duration = total_samples / sample_rate
        
        prev_embedding = None
        prev_speaker = None
        segment_start = 0.0
        
        position = 0
        while position < total_samples:
            end_pos = min(position + chunk_samples, total_samples)
            chunk = waveform[:, position:end_pos]
            
            if chunk.shape[1] < sample_rate * 0.5:
                break
            
            with torch.no_grad():
                embedding = classifier.encode_batch(chunk)
                embedding = embedding.squeeze().cpu().numpy()
            
            current_time = position / sample_rate
            
            if len(db) > 0:
                results = db.query(embedding, top_k=1)
                current_speaker = results[0]["name"] if results and results[0]["similarity"] > threshold else None
                current_similarity = results[0]["similarity"] if results else 0.0
            else:
                current_speaker = None
                current_similarity = 0.0
            
            speaker_changed = False
            if prev_embedding is not None:
                similarity = cosine_similarity(prev_embedding, embedding)
                if similarity < change_threshold or current_speaker != prev_speaker:
                    speaker_changed = True
            else:
                speaker_changed = True
            
            if speaker_changed:
                # Emit previous segment
                if segment_start < current_time:
                    if not filter_unknown or prev_speaker is not None:
                        events.append({
                            "type": "segment",
                            "start": round(segment_start, 2),
                            "end": round(current_time, 2),
                            "speaker": prev_speaker
                        })
                segment_start = current_time
                prev_speaker = current_speaker
            
            prev_embedding = embedding
            position += hop_samples
        
        # Final segment
        if segment_start < duration:
            if not filter_unknown or prev_speaker is not None:
                events.append({
                    "type": "segment",
                    "start": round(segment_start, 2),
                    "end": round(duration, 2),
                "speaker": prev_speaker
            })
        
        return {
            "duration": round(duration, 2),
            "segments": events,
            "speakers": db.list_names()
        }
        
    finally:
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
        if temp_wav and os.path.exists(temp_wav):
            os.unlink(temp_wav)





# ------------------- NeMo MSDD diarization + known-speaker naming -------------------

def _nemo_create_msdd_config(
    work_dir: str,
    wav_path: str,
    duration: float,
    device: str,
    domain_type: str = "telephonic",
) -> "OmegaConf":
    """Create a NeMo diarization config from the local YAML file and override runtime fields.

    This mirrors the approach used in your `diarize.py` (NeuralDiarizer(cfg=create_config(...)).to(device)).
    We keep it offline diarization only (no transcription).
    """
    from omegaconf import OmegaConf

    # Local YAML should live next to this file under nemo_msdd_configs/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, "nemo_msdd_configs", f"diar_infer_{domain_type}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"NeMo diarization config not found at {cfg_path}. "
            "Create it under vectorme/nemo_msdd_configs/diar_infer_telephonic.yaml"
        )

    cfg = OmegaConf.load(cfg_path)

    # Some NeMo diarization configs are structured; if the YAML is missing a root `device` key,
    # NeMo may crash when it tries to access cfg.device. Ensure it exists.
    try:
        has_device_key = ("device" in cfg)
    except Exception:
        has_device_key = hasattr(cfg, "device")

    if not has_device_key:
        # Temporarily disable struct to allow inserting missing keys
        try:
            OmegaConf.set_struct(cfg, False)
        except Exception:
            pass
        cfg.device = None
        try:
            OmegaConf.set_struct(cfg, True)
        except Exception:
            pass

    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # NeMo diarization expects a manifest JSONL
    manifest_path = os.path.join(data_dir, "input_manifest.json")
    meta = {
        "audio_filepath": wav_path,
        "offset": 0,
        "duration": float(duration) if duration else None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(meta) + "\n")

    # Override runtime fields
    cfg.num_workers = 0
    cfg.diarizer.manifest_filepath = manifest_path
    cfg.diarizer.out_dir = work_dir

    # Set device only if the config supports it (we ensured the key exists above).
    # NeuralDiarizer will also be moved to the device via `.to(device)`.
    try:
        cfg.device = str(device) if device else None
    except Exception:
        pass

    # Ensure we don't accidentally run ASR
    if "asr" in cfg.diarizer:
        cfg.diarizer.asr.model_path = None

    return cfg

def _parse_rttm_file(rttm_path: str) -> List[Dict[str, Any]]:
    """Parse an RTTM file into a list of segments.

    RTTM format: SPEAKER <file-id> 1 <start> <dur> <...> <speaker-id> <...>
    """
    segments: List[Dict[str, Any]] = []
    try:
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 9:
                    continue
                if parts[0].upper() != "SPEAKER":
                    continue
                start = float(parts[3])
                dur = float(parts[4])
                spk = parts[7]
                end = start + dur
                segments.append({
                    "start": start,
                    "end": end,
                    "speaker": spk,
                    "raw_speaker": spk,
                    "cause": "ts_vad",
                    "confidence": None,
                    "similarity": None,
                })
    except FileNotFoundError:
        return []

    # Sort and merge consecutive segments with same label (simple cleanup)
    segments.sort(key=lambda x: (x["start"], x["end"]))
    merged: List[Dict[str, Any]] = []
    for s in segments:
        if not merged:
            merged.append(s)
            continue
        prev = merged[-1]
        if prev["speaker"] == s["speaker"] and s["start"] <= prev["end"] + 1e-3:
            prev["end"] = max(prev["end"], s["end"])
            # preserve raw_speaker
            prev["raw_speaker"] = prev["raw_speaker"]
        else:
            merged.append(s)

    # Round for API consistency
    for s in merged:
        s["start"] = round(float(s["start"]), 2)
        s["end"] = round(float(s["end"]), 2)
        # Do NOT round or remove raw_speaker
    return merged


# --- Helper functions for NeMo MSDD + known-speaker naming ---
def _normalize_vec(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float32).flatten()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _concat_waveform_for_speaker(
    waveform,
    sample_rate: int,
    segments: List[Dict[str, Any]],
    raw_speaker: str,
    max_total_seconds: float = None,
) -> Tuple[Optional[Any], float]:
    import torch
    import sys

    if waveform is None:
        return None, 0.0

    # Ensure waveform is always 2D: [1, T]
    if hasattr(waveform, "dim") and waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if max_total_seconds is None:
        max_total_seconds = CONFIG['ts_vad']['max_concat_audio']

    pieces = []
    total = 0.0

    for s in segments:
        if s.get("raw_speaker") != raw_speaker:
            continue

        start = float(s["start"])
        end = float(s["end"])
        dur = max(0.0, end - start)
        if dur <= 0:
            continue
        if total >= max_total_seconds:
            break

        take = min(dur, max_total_seconds - total)
        a0 = int(start * sample_rate)
        a1 = int((start + take) * sample_rate)
        if a1 <= a0:
            continue

        piece = waveform[:, a0:a1]
        if hasattr(piece, "dim") and piece.dim() == 1:
            piece = piece.unsqueeze(0)

        pieces.append(piece)
        total += take

    if not pieces:
        return None, 0.0

    # Normalize before concat + debug
    norm_pieces = []
    for p in pieces:
        if hasattr(p, "dim") and p.dim() == 1:
            p = p.unsqueeze(0)
        norm_pieces.append(p)

    try:
        shapes = [tuple(p.shape) for p in norm_pieces]
        print(f"[TS-VAD][concat] raw_speaker={raw_speaker} num_pieces={len(norm_pieces)} shapes={shapes}", file=sys.stderr)
    except Exception:
        pass

    return torch.cat(norm_pieces, dim=1), total


def _match_cluster_to_known(
    classifier,
    torch,
    cluster_wave,
    db: 'VectorDB',
    identify_threshold: float,
    lr_model: Optional[SpeakerLRModel] = None,
    prob_threshold: float = 0.70,
    margin_threshold: float = 0.05,
) -> Tuple[Optional[str], float]:
    """Match cluster embedding to known speakers.

    If LR model is available: return (name, top_prob) when confident.
    Otherwise fall back to cosine DB query and return (name, cosine_sim).
    """
    with torch.no_grad():
        emb = classifier.encode_batch(cluster_wave)
        emb = _normalize_vec(emb.squeeze().cpu().numpy())

    # 1) LR model (preferred)
    if lr_model is not None and getattr(lr_model, "is_trained", False):
        pred = lr_model.predict(emb)
        name = pred.get("name")
        top_prob = float(pred.get("top_prob", 0.0) or 0.0)
        margin = float(pred.get("margin", 0.0) or 0.0)

        if name and top_prob >= float(prob_threshold) and margin >= float(margin_threshold):
            return name, top_prob

        # Not confident enough from LR; fall through to cosine as a backstop.
        # This prevents strong cosine matches (e.g., known speakers) from being labeled as unknown
        # just because LR thresholds are strict or the class is missing/weak.
        lr_score = top_prob

    # 2) Cosine fallback
    if len(db) == 0:
        # If LR was attempted but not confident, return LR score so callers can log it.
        return None, float(locals().get("lr_score", 0.0))

    res = db.query(emb, top_k=1)
    if res and res[0]["similarity"] >= identify_threshold:
        return res[0]["name"], float(res[0]["similarity"])

    # No confident cosine match either.
    # Return the best available score (cosine if present, else LR top_prob) for debugging/telemetry.
    cosine_score = float(res[0]["similarity"]) if res else 0.0
    return None, max(cosine_score, float(locals().get("lr_score", 0.0)))


from contextlib import contextmanager

@contextmanager
def _safe_tensor_view(torch_module):
    """Temporarily patch torch.Tensor.view to fall back to reshape on CPU.
    
    NeMo operations may call .view() on tensors with non-contiguous memory layout,
    which fails on CPU. This context manager scopes the patch to prevent global side effects.
    """
    _original_view = torch_module.Tensor.view
    def _safe_view(self, *args):
        try:
            return _original_view(self, *args)
        except RuntimeError as e:
            if "view size is not compatible" in str(e):
                return self.reshape(*args)
            raise
    try:
        torch_module.Tensor.view = _safe_view
        yield
    finally:
        torch_module.Tensor.view = _original_view


def nemo_ts_vad_refine(
    wav_path: str,
    duration: float,
    known_speakers: List[str],
    classifier,
    torchaudio,
    torch,
    db: 'VectorDB',
    identify_threshold: float = None,
    min_segment_duration: float = None,
    min_cluster_audio_seconds: float = None,
    device: str = "cpu",
    lr_model: Optional[SpeakerLRModel] = None,
    prob_threshold: Optional[float] = None,
    margin_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run NeMo MSDD diarization (TS-VAD-style refinement) and then map diarized speakers to known names.

    - Stage 1: NeMo diarization -> RTTM segments with raw speaker ids.
    - Stage 2: For each raw speaker id, concatenate that speaker's audio and compute an ECAPA embedding.
               Match embedding against VectorDB; if match >= identify_threshold, rename to known speaker.
               Otherwise assign stable unknown_1, unknown_2, ...

    Returns API schema compatible with the frontend: {mode, duration, segments, unknown_speakers, known_speakers, backend}
    """
    # Load default values from config
    if identify_threshold is None:
        identify_threshold = CONFIG['ts_vad']['identify_threshold']
    if min_segment_duration is None:
        min_segment_duration = CONFIG['ts_vad']['min_segment_duration']
    if min_cluster_audio_seconds is None:
        min_cluster_audio_seconds = CONFIG['ts_vad']['min_cluster_audio']

    # LR identification thresholds (optional)
    if prob_threshold is None:
        prob_threshold = float(CONFIG.get('ts_vad', {}).get('lr_prob_threshold', 0.70))
    if margin_threshold is None:
        margin_threshold = float(CONFIG.get('ts_vad', {}).get('lr_margin_threshold', 0.05))
    
    from omegaconf import OmegaConf

    try:
        from nemo.collections.asr.models import NeuralDiarizer
    except Exception as e:
        raise ImportError("NeMo diarization (NeuralDiarizer) not available") from e

    work_dir = tempfile.mkdtemp(prefix="nemo_msdd_")
    try:
        # Build NeMo config from local YAML and override runtime paths
        diarizer_cfg = _nemo_create_msdd_config(
            work_dir=work_dir,
            wav_path=wav_path,
            duration=float(duration),
            device=str(device),
            domain_type="telephonic",
        )

        # Run diarization (MSDD enabled via YAML)
        msdd_model = NeuralDiarizer(cfg=diarizer_cfg).to(str(device))
        # Some NeMo versions keep a nested _cfg; ensure paths are consistent
        try:
            msdd_model._cfg.diarizer.manifest_filepath = diarizer_cfg.diarizer.manifest_filepath
            msdd_model._cfg.diarizer.out_dir = diarizer_cfg.diarizer.out_dir
        except Exception:
            pass

        # Wrap diarization in context manager to safely handle tensor view operations on CPU
        with _safe_tensor_view(torch):
            msdd_model.diarize()

        # NeMo usually writes RTTM under <out_dir>/pred_rttms/mono_file.rttm or <uniq_id>.rttm.
        # We don't control uniq_id here, so search pred_rttms for the first RTTM.
        rttm_dir = os.path.join(work_dir, "pred_rttms")
        rttm_path = None
        if os.path.isdir(rttm_dir):
            for fn in os.listdir(rttm_dir):
                if fn.lower().endswith(".rttm"):
                    rttm_path = os.path.join(rttm_dir, fn)
                    break

        if not rttm_path or not os.path.exists(rttm_path):
            raise RuntimeError("NeMo diarization did not produce an RTTM file in pred_rttms")

        segments = _parse_rttm_file(rttm_path)

        # Filter short segments
        filtered: List[Dict[str, Any]] = []
        for s in segments:
            if (float(s["end"]) - float(s["start"])) >= float(min_segment_duration):
                filtered.append(s)

        # Load waveform once for cluster-audio concatenation
        waveform, sr = torchaudio.load(wav_path)
        if hasattr(waveform, "dim") and waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        print(f"[TS-VAD][refine] wav_path={wav_path} initial_waveform_shape={tuple(waveform.shape)} sr={sr}", file=sys.stderr)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        # Ensure contiguous memory layout for CPU compatibility with NeMo operations
        waveform = waveform.contiguous()

        # VAD hard gate: filter out silence/noise segments
        vad = SileroVAD(torch, threshold=CONFIG['vad']['confidence_threshold'])
        vad_filtered: List[Dict[str, Any]] = []
        for s in filtered:
            start_sample = int(float(s['start']) * sr)
            end_sample = int(float(s['end']) * sr)
            segment_wave = waveform[:, start_sample:end_sample]
            
            if segment_wave.shape[1] == 0:
                continue  # Skip empty segments
            
            has_speech, avg_confidence, speech_ratio = vad.get_speech_confidence(segment_wave, sr)
            
            # Drop segment if speech_ratio < 0.15 OR avg_confidence < 0.25
            if speech_ratio < 0.15 or avg_confidence < 0.25:
                continue  # Drop this segment (silence/noise)
            
            # Keep segment with VAD metrics
            s['vad_confidence'] = round(float(avg_confidence), 3)
            s['vad_speech_ratio'] = round(float(speech_ratio), 3)
            vad_filtered.append(s)
        
        filtered = vad_filtered

        # Collect raw speakers
        raw_speakers: List[str] = []
        for s in filtered:
            rs = s.get("raw_speaker")
            if rs and rs not in raw_speakers:
                raw_speakers.append(rs)

        # Map raw speaker -> final label (known name or unknown_i)
        speaker_map: Dict[str, Dict[str, Any]] = {}
        unknown_idx = 0
        for rs in raw_speakers:
            cluster_wave, total_sec = _concat_waveform_for_speaker(
                waveform, sr, filtered, rs, max_total_seconds=12.0
            )
            if cluster_wave is None or total_sec < float(min_cluster_audio_seconds):
                # Not enough audio to identify reliably
                unknown_idx += 1
                speaker_map[rs] = {"label": f"unknown_{unknown_idx}", "similarity": 0.0}
                continue

            name, sim = _match_cluster_to_known(
                classifier=classifier,
                torch=torch,
                cluster_wave=cluster_wave,
                db=db,
                identify_threshold=float(identify_threshold),
                lr_model=lr_model,
                prob_threshold=prob_threshold,
                margin_threshold=margin_threshold,
            )
            if name is not None:
                speaker_map[rs] = {"label": name, "similarity": float(sim)}
            else:
                unknown_idx += 1
                speaker_map[rs] = {"label": f"unknown_{unknown_idx}", "similarity": float(sim)}

        # Apply speaker labels to segments for merging
        labeled_segments: List[Dict[str, Any]] = []
        for s in filtered:
            rs = s.get("raw_speaker")
            mapped = speaker_map.get(rs, {"label": None, "similarity": 0.0})
            s_copy = s.copy()
            s_copy["final_speaker"] = mapped.get("label")
            s_copy["similarity"] = mapped.get("similarity", 0.0)
            labeled_segments.append(s_copy)
        
        # Sort by start time
        labeled_segments.sort(key=lambda x: float(x["start"]))
        
        # Merge segments with same speaker if gap < 0.5s
        # Remove short unknown segments (<0.6s) sandwiched between same speakers
        merged: List[Dict[str, Any]] = []
        for s in labeled_segments:
            if not merged:
                merged.append(s)
                continue
            
            prev = merged[-1]
            gap = float(s["start"]) - float(prev["end"])
            
            # Merge if same speaker and gap < 0.5s
            if prev["final_speaker"] == s["final_speaker"] and gap < 0.5:
                prev["end"] = s["end"]
                # Keep higher similarity
                prev["similarity"] = max(float(prev.get("similarity", 0.0)), float(s.get("similarity", 0.0)))
                continue
            
            # Check if previous segment is short unknown sandwiched between same speakers
            if (
                len(merged) >= 2 and
                prev["final_speaker"] and
                prev["final_speaker"].startswith("unknown_") and
                (float(prev["end"]) - float(prev["start"])) < 0.6
            ):
                # Look ahead to see if current segment has same speaker as segment before prev
                before_prev = merged[-2]
                if before_prev["final_speaker"] == s["final_speaker"]:
                    # Remove the short unknown segment and merge surrounding segments
                    merged.pop()  # Remove prev (short unknown)
                    before_prev["end"] = s["end"]
                    before_prev["similarity"] = max(
                        float(before_prev.get("similarity", 0.0)),
                        float(s.get("similarity", 0.0))
                    )
                    continue
            
            merged.append(s)
        
        filtered = merged

        # Apply mapping to segments
        out_segments: List[Dict[str, Any]] = []
        unknown_speakers: List[str] = []
        for s in filtered:
            label = s.get("final_speaker")
            sim = s.get("similarity", 0.0)
            if isinstance(label, str) and label.startswith("unknown_") and label not in unknown_speakers:
                unknown_speakers.append(label)

            out_segments.append({
                "start": s["start"],
                "end": s["end"],
                "speaker": label,
                "cause": "ts_vad",
                "confidence": None,
                "similarity": round(float(sim), 3) if sim is not None else None,
            })

        return {
            "mode": "ts_vad",
            "duration": round(float(duration), 2),
            "segments": out_segments,
            "unknown_speakers": unknown_speakers,
            "known_speakers": known_speakers,
            "backend": "nemo_msdd",
        }

    finally:
        try:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


def run_server(host, port, db_path, device="cpu", recordings_path=None):
    """Run HTTP server for audio diarization."""
    from flask import Flask, request, jsonify, Response, send_from_directory, send_file
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier
    import uuid
    from datetime import datetime
    
    # Get the directory where this file is located
    import os.path as osp
    static_folder = osp.join(osp.dirname(osp.abspath(__file__)), 'static')
    
    # Setup recordings directory
    recordings_dir = Path(recordings_path) if recordings_path else DEFAULT_RECORDINGS_PATH
    recordings_dir.mkdir(parents=True, exist_ok=True)
    
    app = Flask(__name__, static_folder=static_folder)
    
    # Load model once at startup
    print(f"Loading ECAPA-TDNN model on {device}...", file=sys.stderr)
    model_dir = "~/.cache/speechbrain/spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": device}
    )
    
    # Initialize database
    db = VectorDB(db_path)
    print(f"Loaded {len(db)} speakers from database", file=sys.stderr)
    
    # Load LR speaker classifier if available (optional)
    try:
        lr_model = SpeakerLRModel.load(DEFAULT_LR_MODEL_PATH)
        if not lr_model.is_trained:
            lr_model = None
        else:
            print(f"Loaded LR model with {len(lr_model.classes)} classes", file=sys.stderr)
            print(f"[LR] classes: {lr_model.classes}", file=sys.stderr)
    except Exception as e:
        print(f"Warning: failed to load LR model: {e}", file=sys.stderr)
        lr_model = None
    
    # VAD will be loaded on first use if requested
    vad_instance = None
    
    def get_vad(vad_threshold):
        nonlocal vad_instance
        if vad_instance is None:
            print("Loading Silero VAD...", file=sys.stderr)
            vad_instance = SileroVAD(torch, threshold=vad_threshold)
        return vad_instance
    
    def stream_diarization(audio_bytes, chunk_size, chunk_hop, threshold, change_threshold, filter_unknown, use_vad=False, vad_threshold=0.5):
        """Generator that yields diarization events as NDJSON."""
        temp_input = None
        temp_wav = None
        
        # Dynamic threshold based on VAD confidence
        # High VAD = confident speech, so we can trust lower similarity matches
        # When VAD is disabled (segment_vad is None), use a low default threshold
        def get_effective_threshold(segment_vad, base_threshold):
            if segment_vad is None:
                # VAD disabled - use a lower threshold to be more permissive
                return max(0.40, base_threshold - 0.10)
            if segment_vad >= 0.7:
                # High VAD confidence - speech is clear, lower threshold
                return max(0.35, base_threshold - 0.15)
            elif segment_vad >= 0.5:
                # Medium VAD - moderate adjustment
                return max(0.40, base_threshold - 0.10)
            else:
                # Lower VAD - use base threshold
                return base_threshold
        
        try:
            # Write bytes to temp file
            temp_fd, temp_input = tempfile.mkstemp()
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Convert through ffmpeg
            temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd2)
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_input, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                yield json.dumps({"error": f"ffmpeg conversion failed: {result.stderr}"}) + "\n"
                return
            
            waveform, sample_rate = torchaudio.load(temp_wav)
            
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            chunk_samples = int(chunk_size * sample_rate)
            hop_samples = int(chunk_hop * sample_rate)
            total_samples = waveform.shape[1]
            duration = total_samples / sample_rate
            
            # Emit metadata first
            yield json.dumps({"event": "start", "duration": round(duration, 2), "speakers": db.list_names(), "vad": use_vad}) + "\n"
            
            # Get VAD if enabled
            vad = get_vad(vad_threshold) if use_vad else None
            
            # Minimum VAD confidence required to attempt speaker identification
            # Segments below this are likely noise/silence and shouldn't be identified
            min_vad_for_identification = 0.1
            
            # Accumulated embedding for current segment (more accurate than single chunk)
            # Weight embeddings by VAD confidence so noise contributes less
            accumulated_embedding = None
            accumulated_weight = 0.0
            # Also track the best embedding (highest VAD) for speaker identification
            best_embedding = None
            best_embedding_vad = 0.0
            prev_speaker = None
            prev_vad_confidence = None
            segment_start = 0.0
            # Track consecutive silent chunks - only close segment after sustained silence
            consecutive_silent_chunks = 0
            min_silent_chunks_to_close = 2  # Require ~1 second of silence (2 chunks at 0.5s hop)
            
            position = 0
            while position < total_samples:
                end_pos = min(position + chunk_samples, total_samples)
                chunk = waveform[:, position:end_pos]
                
                if chunk.shape[1] < sample_rate * 0.5:
                    break
                
                current_time = position / sample_rate
                
                # VAD check - get speech confidence
                vad_confidence = None
                vad_speech_ratio = None
                if vad is not None:
                    has_speech, vad_confidence, vad_speech_ratio = vad.get_speech_confidence(chunk, sample_rate)
                    if not has_speech:
                        consecutive_silent_chunks += 1
                        # Only close segment after sustained silence (prevents gaps from breaking segments)
                        if consecutive_silent_chunks >= min_silent_chunks_to_close:
                            # Sustained silence - emit segment if we had one
                            if segment_start is not None and segment_start < current_time:
                                # Extract embedding from the full segment audio for accurate identification
                                segment_start_sample = int(segment_start * sample_rate)
                                segment_end_sample = int(current_time * sample_rate)
                                segment_waveform = waveform[:, segment_start_sample:segment_end_sample]
                                
                                final_speaker = None
                                # Compute VAD for the full segment (more accurate than running average)
                                if segment_waveform.shape[1] >= sample_rate * 0.5:
                                    _, segment_vad, _ = vad.get_speech_confidence(segment_waveform, sample_rate)
                                else:
                                    segment_vad = prev_vad_confidence if prev_vad_confidence is not None else 0.0
                                
                                # Skip segment entirely if VAD is too low (just noise/silence, not speech)
                                # Only apply VAD filtering if VAD is actually enabled
                                if vad is not None and segment_vad < min_vad_for_identification:
                                    # Don't emit segment - it's just noise
                                    pass
                                else:
                                    # Try to identify speaker
                                    final_similarity = 0.0
                                    if segment_waveform.shape[1] >= sample_rate * 0.5 and len(db) > 0:
                                        with torch.no_grad():
                                            full_segment_embedding = classifier.encode_batch(segment_waveform)
                                            full_segment_embedding = full_segment_embedding.squeeze().cpu().numpy()
                                        results = db.query(full_segment_embedding, top_k=1)
                                        # Use dynamic threshold - high VAD = confident speech = lower threshold
                                        effective_threshold = get_effective_threshold(segment_vad, threshold)
                                        final_speaker = results[0]["name"] if results and results[0]["similarity"] > effective_threshold else None
                                        final_similarity = results[0]["similarity"] if results else 0.0
                                    
                                    if not filter_unknown or final_speaker is not None:
                                        segment_data = {
                                            "event": "segment",
                                            "start": round(segment_start, 2),
                                            "end": round(current_time, 2),
                                            "speaker": final_speaker,
                                            "similarity": round(final_similarity, 3) if final_similarity else None
                                        }
                                        if vad is not None:
                                            segment_data["vad_confidence"] = round(segment_vad, 3)
                                        yield json.dumps(segment_data) + "\n"
                                accumulated_embedding = None
                                accumulated_weight = 0.0
                                best_embedding = None
                                best_embedding_vad = 0.0
                                prev_speaker = None
                                prev_vad_confidence = None
                            segment_start = None  # Mark that we're in silence
                        position += hop_samples
                        continue
                
                # Speech detected - reset silent chunk counter
                consecutive_silent_chunks = 0
                
                # If we were in silence and speech resumed, update segment_start
                if segment_start is None:
                    segment_start = current_time
                    prev_vad_confidence = vad_confidence
                
                with torch.no_grad():
                    embedding = classifier.encode_batch(chunk)
                    embedding = embedding.squeeze().cpu().numpy()
                
                # Check if this embedding is similar to accumulated (same speaker continuing)
                # or radically different (speaker change)
                # Only consider speaker change if VAD confidence is high enough (clear speech)
                speaker_changed = False
                min_vad_for_change = 0.5  # Minimum VAD to consider a speaker change
                if accumulated_embedding is not None:
                    similarity_to_accumulated = cosine_similarity(accumulated_embedding, embedding)
                    # Only trigger speaker change if:
                    # 1. Embedding is radically different AND
                    # 2. Current chunk has high enough VAD (not noise)
                    current_vad = vad_confidence if vad_confidence is not None else 0.5
                    if similarity_to_accumulated < change_threshold and current_vad >= min_vad_for_change:
                        # Radically different with clear speech - speaker change
                        speaker_changed = True
                else:
                    # First chunk of a new segment
                    speaker_changed = True
                
                if speaker_changed:
                    # Emit previous segment - extract embedding from FULL segment audio for accurate identification
                    if segment_start is not None and segment_start < current_time:
                        # Extract embedding from the full segment (not just best chunk)
                        segment_start_sample = int(segment_start * sample_rate)
                        segment_end_sample = int(current_time * sample_rate)
                        segment_waveform = waveform[:, segment_start_sample:segment_end_sample]
                        
                        final_speaker = None
                        final_similarity = 0.0
                        # Compute VAD for the full segment (more accurate than running average)
                        if vad is not None and segment_waveform.shape[1] >= sample_rate * 0.5:
                            _, segment_vad, _ = vad.get_speech_confidence(segment_waveform, sample_rate)
                        else:
                            segment_vad = None  # None means VAD is disabled
                        
                        # Only try to identify speaker if VAD is high enough (or VAD is disabled)
                        # When VAD is disabled (segment_vad is None), always try to identify
                        vad_ok = segment_vad is None or segment_vad >= min_vad_for_identification
                        if vad_ok and segment_waveform.shape[1] >= sample_rate * 0.5 and len(db) > 0:
                            with torch.no_grad():
                                full_segment_embedding = classifier.encode_batch(segment_waveform)
                                full_segment_embedding = full_segment_embedding.squeeze().cpu().numpy()
                            results = db.query(full_segment_embedding, top_k=1)
                            # Use dynamic threshold - high VAD = confident speech = lower threshold
                            effective_threshold = get_effective_threshold(segment_vad, threshold)
                            final_speaker = results[0]["name"] if results and results[0]["similarity"] > effective_threshold else None
                            final_similarity = results[0]["similarity"] if results else 0.0
                        
                        # Skip segment entirely if VAD is too low (just noise/silence)
                        # Only apply VAD filtering if VAD is actually enabled
                        if vad is None or segment_vad >= min_vad_for_identification:
                            if not filter_unknown or final_speaker is not None:
                                segment_data = {
                                    "event": "segment",
                                    "start": round(segment_start, 2),
                                    "end": round(current_time, 2),
                                    "speaker": final_speaker,
                                    "similarity": round(final_similarity, 3) if final_similarity else None
                                }
                                if vad is not None and prev_vad_confidence is not None:
                                    segment_data["vad_confidence"] = round(prev_vad_confidence, 3)
                                yield json.dumps(segment_data) + "\n"
                    
                    # Start new segment with this embedding (weighted by VAD confidence)
                    segment_start = current_time
                    accumulated_embedding = embedding.copy()
                    accumulated_weight = vad_confidence if vad_confidence is not None else 0.5
                    # Reset best embedding tracking
                    current_vad_val = vad_confidence if vad_confidence is not None else 0.5
                    best_embedding = embedding.copy()
                    best_embedding_vad = current_vad_val
                    prev_vad_confidence = vad_confidence
                    
                    # Identify speaker for this new chunk
                    if len(db) > 0:
                        results = db.query(embedding, top_k=1)
                        current_speaker = results[0]["name"] if results and results[0]["similarity"] > threshold else None
                        current_similarity = results[0]["similarity"] if results else 0.0
                    else:
                        current_speaker = None
                        current_similarity = 0.0
                    
                    prev_speaker = current_speaker
                    
                    # Emit speaker change event
                    change_data = {
                        "event": "speaker_change",
                        "time": round(current_time, 2),
                        "speaker": current_speaker,
                        "similarity": round(current_similarity, 3)
                    }
                    if vad_confidence is not None:
                        change_data["vad_confidence"] = round(vad_confidence, 3)
                    yield json.dumps(change_data) + "\n"
                else:
                    # Similar to accumulated - merge this embedding weighted by VAD confidence
                    # High VAD = clear speech = more weight; Low VAD = noise = less weight
                    chunk_weight = vad_confidence if vad_confidence is not None else 0.5
                    accumulated_weight += chunk_weight
                    
                    # Weighted running average: weight by VAD confidence
                    if accumulated_weight > 0:
                        blend = chunk_weight / accumulated_weight
                        accumulated_embedding = (1 - blend) * accumulated_embedding + blend * embedding
                        # Normalize to unit length for cosine similarity
                        accumulated_embedding = accumulated_embedding / np.linalg.norm(accumulated_embedding)
                    
                    # Track the best embedding (highest VAD) for speaker identification
                    current_vad_val = vad_confidence if vad_confidence is not None else 0.5
                    if current_vad_val > best_embedding_vad:
                        best_embedding = embedding.copy()
                        best_embedding_vad = current_vad_val
                    
                    # Track running average of VAD confidence for the segment
                    if vad_confidence is not None and prev_vad_confidence is not None:
                        prev_vad_confidence = (prev_vad_confidence + vad_confidence) / 2
                    elif vad_confidence is not None:
                        prev_vad_confidence = vad_confidence
                
                position += hop_samples
            
            # Final segment - extract embedding from full segment audio for accurate identification
            if segment_start is not None and segment_start < duration:
                segment_start_sample = int(segment_start * sample_rate)
                segment_waveform = waveform[:, segment_start_sample:]
                
                final_speaker = None
                # Compute VAD for the full segment (more accurate than running average)
                if vad is not None and segment_waveform.shape[1] >= sample_rate * 0.5:
                    _, segment_vad, _ = vad.get_speech_confidence(segment_waveform, sample_rate)
                else:
                    segment_vad = None  # None means VAD is disabled
                
                # Skip segment entirely if VAD is too low (just noise/silence)
                # Only apply VAD filtering if VAD is actually enabled
                vad_ok = segment_vad is None or segment_vad >= min_vad_for_identification
                if vad_ok:
                    # Try to identify speaker
                    final_similarity = 0.0
                    if segment_waveform.shape[1] >= sample_rate * 0.5 and len(db) > 0:
                        with torch.no_grad():
                            full_segment_embedding = classifier.encode_batch(segment_waveform)
                            full_segment_embedding = full_segment_embedding.squeeze().cpu().numpy()
                        results = db.query(full_segment_embedding, top_k=1)
                        # Use dynamic threshold - high VAD = confident speech = lower threshold
                        effective_threshold = get_effective_threshold(segment_vad, threshold)
                        final_speaker = results[0]["name"] if results and results[0]["similarity"] > effective_threshold else None
                        final_similarity = results[0]["similarity"] if results else 0.0
                    
                    if not filter_unknown or final_speaker is not None:
                        segment_data = {
                            "event": "segment",
                            "start": round(segment_start, 2),
                            "end": round(duration, 2),
                            "speaker": final_speaker,
                            "similarity": round(final_similarity, 3) if final_similarity else None
                        }
                        if prev_vad_confidence is not None:
                            segment_data["vad_confidence"] = round(prev_vad_confidence, 3)
                        yield json.dumps(segment_data) + "\n"
            
            yield json.dumps({"event": "done"}) + "\n"
            
        finally:
            if temp_input and os.path.exists(temp_input):
                os.unlink(temp_input)
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)
    
    @app.route("/v1/audio/transcriptions", methods=["POST"])
    def transcribe():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        audio_bytes = file.read()
        
        response_format = request.form.get("response_format", "json")
        chunk_size = float(request.form.get("chunk_size", 3.0))
        chunk_hop = float(request.form.get("chunk_hop", 0.5))
        threshold = float(request.form.get("threshold", 0.5))
        identify_threshold = float(request.form.get("identify_threshold", threshold))
        change_threshold = float(request.form.get("change_threshold", 0.7))
        filter_unknown = request.form.get("filter_unknown", "false").lower() == "true"
        stream = request.form.get("stream", "false").lower() == "true"
        # VAD enabled by default for diarization, use vad=false to disable
        use_vad = request.form.get("vad", "true").lower() != "false"
        vad_threshold = float(request.form.get("vad_threshold", 0.5))
        # Demucs is backend-supported only for TS-VAD path for now (UI will be added later)
        use_demucs = request.form.get("demucs", "false").lower() == "true"
        
        # TS-VAD refinement mode parameters
        diarization_mode = request.form.get("diarization_mode", "coarse")
        min_segment_duration = float(request.form.get("min_segment_duration", 0.5))
        
        try:
            if response_format == "diarized_json":
                if stream:
                    # Streaming response - NDJSON with buffering disabled
                    response = Response(
                        stream_diarization(audio_bytes, chunk_size, chunk_hop, threshold, change_threshold, filter_unknown, use_vad=use_vad, vad_threshold=vad_threshold),
                        mimetype='application/x-ndjson'
                    )
                    response.headers['X-Accel-Buffering'] = 'no'
                    response.headers['Cache-Control'] = 'no-cache'
                    return response
                elif diarization_mode == "ts_vad":
                    # TS-VAD refinement mode - non-streaming batch.
                    # Only use NeMo MSDD diarization + known-speaker naming. No fallback.
                    temp_input = None
                    temp_wav = None
                    try:
                        # Write bytes to temp file
                        fd, temp_input = tempfile.mkstemp()
                        os.write(fd, audio_bytes)
                        os.close(fd)

                        # Convert through ffmpeg to 16kHz mono WAV
                        fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
                        os.close(fd2)
                        conv = subprocess.run(
                            ['ffmpeg', '-i', temp_input, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                            capture_output=True,
                            text=True
                        )
                        if conv.returncode != 0:
                            raise RuntimeError(f"ffmpeg conversion failed: {conv.stderr}")

                        # Optional Demucs preprocessing (best-effort) for TS-VAD only
                        demucs_dir = None
                        wav_after_demucs, demucs_dir = maybe_extract_vocals_with_demucs(
                            temp_wav, enable=use_demucs, device=device
                        )
                        print(f"[TS-VAD][demucs] enabled={use_demucs} input={temp_wav} demucs_out={wav_after_demucs}", file=sys.stderr)

                        # IMPORTANT: Demucs often outputs 44.1kHz stereo. NeMo diarization expects 16kHz mono.
                        # So we always normalize to 16k mono WAV before passing into NeMo.
                        fd3, temp_wav_16k = tempfile.mkstemp(suffix='.wav')
                        os.close(fd3)
                        conv2 = subprocess.run(
                            ['ffmpeg', '-i', wav_after_demucs, '-ar', '16000', '-ac', '1', '-y', temp_wav_16k],
                            capture_output=True,
                            text=True
                        )
                        if conv2.returncode != 0:
                            raise RuntimeError(f"ffmpeg conversion (post-demucs) failed: {conv2.stderr}")

                        wav_for_processing = temp_wav_16k

                        # Get duration (from normalized 16k mono)
                        wf, sr = torchaudio.load(wav_for_processing)
                        print(f"[TS-VAD][audio] normalized_shape={tuple(wf.shape)} sr={sr} path={wav_for_processing}", file=sys.stderr)
                        if hasattr(wf, 'dim') and wf.dim() == 1:
                            wf = wf.unsqueeze(0)
                        if wf.shape[0] > 1:
                            wf = torch.mean(wf, dim=0, keepdim=True)
                        dur = float(wf.shape[1] / sr)

                        # Debug logging
                        audio_min, audio_max = float(wf.min()), float(wf.max())
                        audio_mean = float(wf.abs().mean())
                        print(f"[TS-VAD] Duration: {dur:.2f}s | Shape: {wf.shape} | SR: {sr}", file=sys.stderr)
                        print(f"[TS-VAD] Amplitude - min: {audio_min:.4f}, max: {audio_max:.4f}, mean: {audio_mean:.4f}", file=sys.stderr)

                        if dur < 2.0:
                            return jsonify({"error": f"Recording too short for TS-VAD (got {dur:.1f}s, need ≥2s)"}), 400

                        try:
                            # Run NeMo MSDD diarization + known-speaker naming
                            result = nemo_ts_vad_refine(
                                wav_path=wav_for_processing,
                                duration=dur,
                                known_speakers=db.list_names(),
                                classifier=classifier,
                                torchaudio=torchaudio,
                                torch=torch,
                                db=db,
                                identify_threshold=identify_threshold,
                                min_segment_duration=min_segment_duration,
                                device=device,
                                lr_model=lr_model,
                            )
                            return jsonify(result)
                        except ImportError as ie:
                            return jsonify({
                                "error": "NeMo diarization is not installed/enabled on this server",
                                "details": str(ie),
                            }), 400
                    finally:
                        # Clean up optional demucs outputs (TS-VAD path only)
                        try:
                            if 'demucs_dir' in locals() and demucs_dir:
                                import shutil
                                shutil.rmtree(demucs_dir, ignore_errors=True)
                        except Exception:
                            pass

                        if temp_input and os.path.exists(temp_input):
                            os.unlink(temp_input)
                        if temp_wav and os.path.exists(temp_wav):
                            os.unlink(temp_wav)
                        if 'temp_wav_16k' in locals() and temp_wav_16k and os.path.exists(temp_wav_16k):
                            os.unlink(temp_wav_16k)
                else:
                    # Batch response (coarse mode)
                    result = diarize_audio_bytes(
                        audio_bytes, classifier, torchaudio, torch, db,
                        chunk_size=chunk_size,
                        chunk_hop=chunk_hop,
                        threshold=threshold,
                        change_threshold=change_threshold,
                        filter_unknown=filter_unknown
                    )
                    return jsonify(result)
            else:
                # Default: return embedding for whole file
                temp_fd, temp_path = tempfile.mkstemp()
                temp_wav = None
                try:
                    os.write(temp_fd, audio_bytes)
                    os.close(temp_fd)
                    
                    # Always convert through ffmpeg
                    temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
                    os.close(temp_fd2)
                    
                    result = subprocess.run(
                        ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode != 0:
                        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
                    
                    waveform, sample_rate = torchaudio.load(temp_wav)
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    
                    with torch.no_grad():
                        embedding = classifier.encode_batch(waveform)
                        embedding = embedding.squeeze().cpu().numpy()
                    
                    if len(db) > 0:
                        results = db.query(embedding, top_k=5)
                        return jsonify({
                            "matches": results,
                            "best": results[0] if results else None
                        })
                    else:
                        return jsonify({
                            "embedding": embedding.tolist(),
                            "dimensions": len(embedding)
                        })
                finally:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    if temp_wav and os.path.exists(temp_wav):
                        os.unlink(temp_wav)
                        
        except Exception as e:
            import traceback
            print("[ERROR] /v1/audio/transcriptions failed:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return jsonify({"error": str(e)}), 500
    
    @app.route("/v1/speakers", methods=["GET"])
    def list_speakers():
        return jsonify({"speakers": db.list_names(), "count": len(db)})
    
    @app.route("/v1/speakers", methods=["POST"])
    def add_speaker():
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        if "name" not in request.form:
            return jsonify({"error": "No name provided"}), 400
        
        file = request.files["file"]
        name = request.form["name"]
        audio_bytes = file.read()
        
        temp_path = None
        temp_wav = None
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Always convert through ffmpeg
            temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd2)
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            waveform, sample_rate = torchaudio.load(temp_wav)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            with torch.no_grad():
                embedding = classifier.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            db.add(name, embedding)
            
            return jsonify({
                "message": f"Added '{name}' to database",
                "count": len(db)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)
    
    @app.route("/v1/speakers/<name>", methods=["DELETE"])
    def remove_speaker(name):
        if db.remove(name):
            return jsonify({"message": f"Removed '{name}'", "count": len(db)})
        else:
            return jsonify({"error": f"'{name}' not found"}), 404
    
    @app.route("/v1/speakers/from-segment", methods=["POST"])
    def add_speaker_from_segment():
        """Add a speaker from a segment of audio with start/end times."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        if "name" not in request.form:
            return jsonify({"error": "No name provided"}), 400
        if "start" not in request.form or "end" not in request.form:
            return jsonify({"error": "start and end times required"}), 400
        
        file = request.files["file"]
        name = request.form["name"]
        start = float(request.form["start"])
        end = float(request.form["end"])
        audio_bytes = file.read()
        
        temp_path = None
        temp_wav = None
        temp_segment = None
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Convert to WAV
            temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd2)
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            # Extract segment using ffmpeg
            temp_fd3, temp_segment = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd3)
            
            duration = end - start
            result = subprocess.run(
                ['ffmpeg', '-i', temp_wav, '-ss', str(start), '-t', str(duration), '-y', temp_segment],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg segment extraction failed: {result.stderr}")
            
            waveform, sample_rate = torchaudio.load(temp_segment)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            with torch.no_grad():
                embedding = classifier.encode_batch(waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            db.add(name, embedding)
            
            return jsonify({
                "message": f"Added '{name}' from segment {start:.1f}s-{end:.1f}s",
                "name": name,
                "count": len(db)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)
            if temp_segment and os.path.exists(temp_segment):
                os.unlink(temp_segment)
    
    @app.route("/v1/speakers/<name>/compare", methods=["POST"])
    def compare_to_speaker(name):
        """Compare audio segments against a specific speaker's embedding."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        # Get speaker's embeddings
        if name not in db.list_names():
            return jsonify({"error": f"Speaker '{name}' not found"}), 404
        
        # Get all embeddings for this speaker
        speaker_indices = [i for i, n in enumerate(db.names) if n == name]
        speaker_embeddings = db.embeddings[speaker_indices]
        
        # Get segment times from request
        segments_json = request.form.get("segments", "[]")
        try:
            segments = json.loads(segments_json)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid segments JSON"}), 400
        
        if not segments:
            return jsonify({"error": "No segments provided"}), 400
        
        file = request.files["file"]
        audio_bytes = file.read()
        
        temp_path = None
        temp_wav = None
        
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Convert to WAV
            temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd2)
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            waveform, sample_rate = torchaudio.load(temp_wav)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            results = []
            for seg in segments:
                start = float(seg.get("start", 0))
                end = float(seg.get("end", 0))
                
                # Extract segment
                start_sample = int(start * sample_rate)
                end_sample = int(end * sample_rate)
                segment_waveform = waveform[:, start_sample:end_sample]
                
                if segment_waveform.shape[1] < sample_rate * 0.1:  # Skip very short segments
                    results.append({"start": start, "end": end, "similarity": None})
                    continue
                
                # Get embedding for this segment
                with torch.no_grad():
                    embedding = classifier.encode_batch(segment_waveform)
                    embedding = embedding.squeeze().cpu().numpy()
                
                # Compute similarity against all speaker embeddings, take max
                max_similarity = 0.0
                for speaker_emb in speaker_embeddings:
                    sim = cosine_similarity(embedding, speaker_emb)
                    max_similarity = max(max_similarity, sim)
                
                results.append({
                    "start": start,
                    "end": end,
                    "similarity": round(max_similarity, 3)
                })
            
            return jsonify({"speaker": name, "segments": results})
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)
    
    @app.route("/v1/speakers/identify-segment", methods=["POST"])
    def identify_segment():
        """Identify the speaker for a specific audio segment."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        start = float(request.form.get("start", 0))
        end = float(request.form.get("end", 0))
        
        if end <= start:
            return jsonify({"error": "Invalid segment times"}), 400
        
        file = request.files["file"]
        audio_bytes = file.read()
        
        temp_path = None
        temp_wav = None
        
        try:
            temp_fd, temp_path = tempfile.mkstemp()
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Convert to WAV
            temp_fd2, temp_wav = tempfile.mkstemp(suffix='.wav')
            os.close(temp_fd2)
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_path, '-ar', '16000', '-ac', '1', '-y', temp_wav],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
            
            waveform, sample_rate = torchaudio.load(temp_wav)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Extract segment
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]
            
            if segment_waveform.shape[1] < sample_rate * 0.1:
                return jsonify({"error": "Segment too short"}), 400
            
            # Get embedding for this segment
            with torch.no_grad():
                embedding = classifier.encode_batch(segment_waveform)
                embedding = embedding.squeeze().cpu().numpy()
            
            # Query database for matches
            matches = db.query(embedding, top_k=5)
            
            # Group by speaker name and get best match per speaker
            best_by_speaker = {}
            for match in matches:
                name = match["name"]
                if name not in best_by_speaker or match["similarity"] > best_by_speaker[name]["similarity"]:
                    best_by_speaker[name] = match
            
            # Sort by similarity
            ranked = sorted(best_by_speaker.values(), key=lambda x: x["similarity"], reverse=True)
            
            return jsonify({
                "start": start,
                "end": end,
                "matches": ranked,
                "top_match": ranked[0] if ranked else None
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            if temp_wav and os.path.exists(temp_wav):
                os.unlink(temp_wav)
    
    # ===== Recording Management Endpoints =====
    
    @app.route("/v1/recordings", methods=["GET"])
    def list_recordings():
        """List all saved recordings with metadata."""
        recordings = []
        for meta_file in recordings_dir.glob("*.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    # Check if the audio file exists
                    audio_path = recordings_dir / metadata.get("filename", "")
                    if audio_path.exists():
                        recordings.append(metadata)
            except (json.JSONDecodeError, IOError):
                continue
        # Sort by timestamp, newest first
        recordings.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return jsonify({"recordings": recordings, "count": len(recordings)})
    
    @app.route("/v1/recordings", methods=["POST"])
    def save_recording():
        """Save a recording with optional metadata."""
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files["file"]
        audio_bytes = file.read()
        
        if len(audio_bytes) == 0:
            return jsonify({"error": "Empty file"}), 400
        
        # Generate unique ID
        recording_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        # Get optional metadata from form
        name = request.form.get("name", f"Recording {recording_id}")
        duration = request.form.get("duration", 0)
        
        # Write raw upload to temp file for conversion
        temp_input = None
        temp_wav = None
        try:
            temp_fd, temp_input = tempfile.mkstemp(suffix='.webm')
            os.write(temp_fd, audio_bytes)
            os.close(temp_fd)
            
            # Convert to clean WAV using ffmpeg to fix WebM chunk concatenation issues
            # This normalizes the audio and removes any stuttering from timeslice recording
            audio_filename = f"{recording_id}.wav"
            audio_path = recordings_dir / audio_filename
            
            result = subprocess.run(
                ['ffmpeg', '-i', temp_input, '-ar', '16000', '-ac', '1', '-y', str(audio_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Fallback: save original file if conversion fails
                audio_filename = f"{recording_id}.webm"
                audio_path = recordings_dir / audio_filename
                with open(audio_path, 'wb') as f:
                    f.write(audio_bytes)
            
            # Get actual file size after conversion
            file_size = audio_path.stat().st_size
            
            # Save metadata
            metadata = {
                "id": recording_id,
                "name": name,
                "filename": audio_filename,
                "original_name": file.filename or "recording.webm",
                "timestamp": timestamp,
                "duration": float(duration) if duration else 0,
                "size": file_size
            }
            
            meta_path = recordings_dir / f"{recording_id}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return jsonify({
                "message": f"Recording saved as '{name}'",
                "recording": metadata
            })
        finally:
            if temp_input and os.path.exists(temp_input):
                os.unlink(temp_input)
    
    @app.route("/v1/recordings/<recording_id>", methods=["GET"])
    def get_recording(recording_id):
        """Download a saved recording."""
        meta_path = recordings_dir / f"{recording_id}.json"
        if not meta_path.exists():
            return jsonify({"error": "Recording not found"}), 404
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            return jsonify({"error": "Invalid recording metadata"}), 500
        
        audio_path = recordings_dir / metadata.get("filename", "")
        if not audio_path.exists():
            return jsonify({"error": "Audio file not found"}), 404
        
        # Determine mimetype
        ext = audio_path.suffix.lower()
        mimetypes = {
            ".webm": "audio/webm",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg"
        }
        mimetype = mimetypes.get(ext, "application/octet-stream")
        
        return send_file(
            audio_path,
            mimetype=mimetype,
            as_attachment=False,
            download_name=metadata.get("original_name", audio_path.name)
        )
    
    @app.route("/v1/recordings/<recording_id>/metadata", methods=["GET"])
    def get_recording_metadata(recording_id):
        """Get metadata for a recording."""
        meta_path = recordings_dir / f"{recording_id}.json"
        if not meta_path.exists():
            return jsonify({"error": "Recording not found"}), 404
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return jsonify(metadata)
        except (json.JSONDecodeError, IOError):
            return jsonify({"error": "Invalid recording metadata"}), 500
    
    @app.route("/v1/recordings/<recording_id>", methods=["DELETE"])
    def delete_recording(recording_id):
        """Delete a saved recording."""
        meta_path = recordings_dir / f"{recording_id}.json"
        if not meta_path.exists():
            return jsonify({"error": "Recording not found"}), 404
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            return jsonify({"error": "Invalid recording metadata"}), 500
        
        # Delete audio file
        audio_path = recordings_dir / metadata.get("filename", "")
        if audio_path.exists():
            os.unlink(audio_path)
        
        # Delete metadata file
        os.unlink(meta_path)
        
        return jsonify({
            "message": f"Recording '{metadata.get('name', recording_id)}' deleted",
            "id": recording_id
        })
    
    @app.route("/v1/recordings/<recording_id>", methods=["PATCH"])
    def update_recording(recording_id):
        """Update recording metadata (e.g., rename)."""
        meta_path = recordings_dir / f"{recording_id}.json"
        if not meta_path.exists():
            return jsonify({"error": "Recording not found"}), 404
        
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, IOError):
            return jsonify({"error": "Invalid recording metadata"}), 500
        
        # Update allowed fields
        data = request.get_json() or {}
        if "name" in data:
            metadata["name"] = data["name"]
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            "message": "Recording updated",
            "recording": metadata
        })
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "speakers": len(db)})
    
    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(static_folder, "index.html")
    
    @app.route("/recording/<recording_id>", methods=["GET"])
    def recording_page(recording_id):
        """Serve index.html for deep linking to recordings. Frontend handles routing."""
        return send_from_directory(static_folder, "index.html")
    
    print(f"Starting server on {host}:{port}", file=sys.stderr)
    app.run(host=host, port=port, threaded=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract speaker embedding vector using ECAPA-TDNN"
    )
    # ---- LR speaker model training (optional) ----
    parser.add_argument("--train-lr", action="store_true",
                        help="Train LR speaker model from current VectorDB embeddings")
    parser.add_argument("--lr-model-path", type=str, default=str(DEFAULT_LR_MODEL_PATH),
                        help="Path to save LR speaker model (.npz)")
    parser.add_argument("--lr-min-samples", type=int, default=5,
                        help="Minimum embeddings per speaker to include in LR training")
    parser.add_argument("--lr-epochs", type=int, default=400,
                        help="Training epochs for LR model")
    parser.add_argument("--lr-step", type=float, default=0.5,
                        help="Learning rate (step size) for LR training")
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
        "--distance",
        action="store_true",
        help="With --list, show cosine distances between embeddings for each speaker"
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
    parser.add_argument(
        "--diarize", "-d",
        action="store_true",
        help="Process audio in chunks and emit speaker change events"
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=3.0,
        help="Chunk duration in seconds for diarization (default: 3.0, matches ECAPA-TDNN training)"
    )
    parser.add_argument(
        "--chunk-hop",
        type=float,
        default=0.5,
        help="Hop between chunks in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum similarity to identify a speaker (default: 0.5)"
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.7,
        help="Similarity threshold below which speaker is considered changed (default: 0.7)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (CUDA or MPS/Metal on Mac)"
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable Voice Activity Detection (VAD is enabled by default)"
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD speech probability threshold (default: 0.5)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as HTTP server daemon"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=3120,
        help="Server port (default: 3120)"
    )
    args = parser.parse_args()
    # ---- Train LR model and exit ----
    if getattr(args, "train_lr", False):
        # Reuse whichever DB flag already exists in this CLI
        # (supports --db-path, --db, etc. without forcing a rename)
        db_path_arg = (
            getattr(args, "db_path", None)
            or getattr(args, "db", None)
            or getattr(args, "dbfile", None)
            or None
        )

        db = VectorDB(db_path_arg)
        model_path = Path(getattr(args, "lr_model_path"))

        lr_model = SpeakerLRModel()
        summary = lr_model.train_from_db(
            db,
            min_samples_per_class=int(getattr(args, "lr_min_samples", 5)),
            lr=float(getattr(args, "lr_step", 0.5)),
            epochs=int(getattr(args, "lr_epochs", 400)),
        )
        lr_model.save(model_path)

        print(json.dumps({"message": "trained_lr_model", "model_path": str(model_path), **summary}, indent=2))
        sys.exit(0)

    # Handle --serve mode first (before loading model)
    if args.serve:
        # Determine device
        import torch
        if args.gpu:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                print("Warning: No GPU available, falling back to CPU", file=sys.stderr)
                device = "cpu"
        else:
            device = "cpu"
        
        run_server(args.host, args.port, args.db, device=device)
        return

    # Initialize database
    db = VectorDB(args.db)

    # Handle --list
    if args.list:
        counts = db.count_by_name()
        if counts:
            for name in sorted(counts.keys()):
                count = counts[name]
                print(f"{name}: {count} embedding(s)")
                
                # Show distances if requested and there are multiple embeddings
                if args.distance and count > 1:
                    embeddings = db.get_embeddings_by_name(name)
                    # Calculate pairwise cosine similarities
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            e1 = embeddings[i].reshape(1, -1)
                            e2 = embeddings[j].reshape(1, -1)
                            norm1 = e1 / np.linalg.norm(e1)
                            norm2 = e2 / np.linalg.norm(e2)
                            sim = float(np.dot(norm1, norm2.T))
                            dist = 1 - sim
                            print(f"  [{i+1}] <-> [{j+1}]: similarity={sim:.3f}, distance={dist:.3f}")
            print(f"\nTotal: {len(db)} embeddings, {len(counts)} speakers", file=sys.stderr)
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

    # Determine device
    if args.gpu:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            print("Warning: No GPU available, falling back to CPU", file=sys.stderr)
            device = "cpu"
    else:
        device = "cpu"

    # Load the ECAPA-TDNN model (downloads on first use)
    model_dir = "~/.cache/speechbrain/spkrec-ecapa-voxceleb"
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=model_dir,
        run_opts={"device": device}
    )
    
    if args.gpu and device != "cpu":
        print(f"Using {device.upper()} acceleration", file=sys.stderr)

    if args.download_only:
        print("Model downloaded successfully!", file=sys.stderr)
        return

    # Initialize VAD (enabled by default for diarization)
    vad = None
    if args.diarize and not args.no_vad:
        print("Loading Silero VAD...", file=sys.stderr)
        vad = SileroVAD(torch, threshold=args.vad_threshold)

    # Handle --diarize mode
    if args.diarize:
        process_chunks(args, classifier, torchaudio, torch, db, vad=vad)
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