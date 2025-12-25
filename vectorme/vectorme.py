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

# Suppress noisy warnings from dependencies
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# Default database location
DEFAULT_DB_PATH = Path.home() / ".vectorme" / "speakers.npz"

# Default recordings directory
DEFAULT_RECORDINGS_PATH = Path.home() / ".vectorme" / "recordings"


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
            
            prev_embedding = None
            prev_speaker = None
            prev_vad_confidence = None
            segment_start = 0.0
            
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
                        # No speech detected
                        if prev_speaker is not None:
                            # Emit previous segment before silence
                            if segment_start < current_time:
                                if not filter_unknown or prev_speaker is not None:
                                    yield json.dumps({
                                        "event": "segment",
                                        "start": round(segment_start, 2),
                                        "end": round(current_time, 2),
                                        "speaker": prev_speaker,
                                        "vad_confidence": round(prev_vad_confidence, 3) if prev_vad_confidence else None
                                    }) + "\n"
                            prev_speaker = None
                            prev_embedding = None
                            prev_vad_confidence = None
                        # Don't update segment_start here - we'll set it when speech resumes
                        segment_start = None  # Mark that we're in silence
                        position += hop_samples
                        continue
                
                # If we were in silence and speech resumed, update segment_start
                if segment_start is None:
                    segment_start = current_time
                    prev_vad_confidence = vad_confidence
                
                with torch.no_grad():
                    embedding = classifier.encode_batch(chunk)
                    embedding = embedding.squeeze().cpu().numpy()
                
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
                    if segment_start is not None and segment_start < current_time:
                        if not filter_unknown or prev_speaker is not None:
                            segment_data = {
                                "event": "segment",
                                "start": round(segment_start, 2),
                                "end": round(current_time, 2),
                                "speaker": prev_speaker
                            }
                            if prev_vad_confidence is not None:
                                segment_data["vad_confidence"] = round(prev_vad_confidence, 3)
                            yield json.dumps(segment_data) + "\n"
                    
                    segment_start = current_time
                    prev_speaker = current_speaker
                    prev_vad_confidence = vad_confidence
                    
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
                    # Track running average of VAD confidence for the segment
                    if vad_confidence is not None and prev_vad_confidence is not None:
                        prev_vad_confidence = (prev_vad_confidence + vad_confidence) / 2
                    elif vad_confidence is not None:
                        prev_vad_confidence = vad_confidence
                
                prev_embedding = embedding
                position += hop_samples
            
            # Final segment
            if segment_start is not None and segment_start < duration:
                if not filter_unknown or prev_speaker is not None:
                    segment_data = {
                        "event": "segment",
                        "start": round(segment_start, 2),
                        "end": round(duration, 2),
                        "speaker": prev_speaker
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
        change_threshold = float(request.form.get("change_threshold", 0.7))
        filter_unknown = request.form.get("filter_unknown", "false").lower() == "true"
        stream = request.form.get("stream", "false").lower() == "true"
        # VAD enabled by default for diarization, use vad=false to disable
        use_vad = request.form.get("vad", "true").lower() != "false"
        vad_threshold = float(request.form.get("vad_threshold", 0.5))
        
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
                else:
                    # Batch response
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
