#!/usr/bin/env python3
"""
vectorme - Extract speaker embeddings using ECAPA-TDNN

Usage:
    cat audio.wav | vectorme
    vectorme < audio.wav
    vectorme --file audio.wav
"""

import sys
import io
import argparse
import json
import warnings
import os
import numpy as np

# Suppress noisy warnings from dependencies
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

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
        "--format", "-o",
        choices=["json", "numpy", "list"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download the model, don't process audio"
    )
    args = parser.parse_args()

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

    # Read audio data
    if args.file:
        # Read from file
        waveform, sample_rate = torchaudio.load(args.file)
    else:
        # Read from stdin
        if sys.stdin.isatty():
            print("Error: No audio input. Pipe audio or use --file", file=sys.stderr)
            print("Usage: cat audio.wav | vectorme", file=sys.stderr)
            sys.exit(1)
        
        audio_bytes = sys.stdin.buffer.read()
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = torchaudio.load(audio_buffer)

    # Resample to 16kHz if needed (model expects 16kHz)
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

    # Output the embedding
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
