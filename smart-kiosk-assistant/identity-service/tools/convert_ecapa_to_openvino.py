#!/usr/bin/env python3
"""Convert SpeechBrain ECAPA-TDNN to an OpenVINO IR (full-pipeline, raw waveform).

This is a **setup-time** tool (run from setup_models.sh inside .setup-venv); it
is never imported by the running identity-service.  It exports a model that
takes a mono 16 kHz waveform tensor ``1xT`` and returns a ``192``-d speaker
embedding, so the runtime engine needs no feature engineering.

Source: ``speechbrain/spkrec-ecapa-voxceleb`` (VoxCeleb, 192-d).

Usage:
    python convert_ecapa_to_openvino.py --output-dir <dir> [--model-name ecapa-tdnn-voice]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_wrapper(source: str):
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=source,
        savedir=str(Path.home() / ".cache" / "speechbrain" / source.replace("/", "_")),
        run_opts={"device": "cpu"},
    )
    classifier.eval()

    class EcapaEmbedder(torch.nn.Module):
        """Wraps feature extraction + encoder into a single waveform->embedding graph."""

        def __init__(self, mods):
            super().__init__()
            self.compute_features = mods.compute_features
            self.mean_var_norm = mods.mean_var_norm
            self.embedding_model = mods.embedding_model

        def forward(self, wav):  # wav: [B, T]
            feats = self.compute_features(wav)
            lengths = torch.ones(wav.shape[0], device=wav.device)
            feats = self.mean_var_norm(feats, lengths)
            embedding = self.embedding_model(feats)  # [B, 1, 192]
            return embedding.squeeze(1)              # [B, 192]

    return EcapaEmbedder(classifier.mods)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="Target dir for the IR.")
    parser.add_argument("--model-name", default="ecapa-tdnn-voice", help="IR base filename.")
    parser.add_argument(
        "--source", default="speechbrain/spkrec-ecapa-voxceleb", help="HF model id."
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000, help="Example waveform rate (Hz)."
    )
    args = parser.parse_args()

    import openvino as ov
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{args.model_name}.xml"

    if xml_path.exists() and (out_dir / f"{args.model_name}.bin").exists():
        print(f"  ✓ Voice IR already present at {xml_path} — skipping")
        return 0

    print(f"  Loading SpeechBrain model: {args.source}")
    wrapper = _build_wrapper(args.source).eval()

    # 3 seconds of audio as the tracing example; time axis is made dynamic below.
    example = torch.randn(1, args.sample_rate * 3, dtype=torch.float32)
    with torch.no_grad():
        out = wrapper(example)
    print(f"  Traced embedding shape: {tuple(out.shape)}")

    print("  Converting to OpenVINO IR (dynamic time axis)...")
    ov_model = ov.convert_model(
        wrapper,
        example_input=example,
        input=[[1, -1]],
    )
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    print(f"  ✓ Saved {xml_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
