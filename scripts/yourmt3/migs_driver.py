"""migs-tab driver for YourMT3+.

Runs YourMT3+ transcription on a single audio file and writes a MIDI to a
caller-specified path. Bypasses the Gradio/yt-dlp paths in app.py and uses
MPS when available (CUDA hardcoded paths in upstream `transcribe()` are
sidestepped here).

Invoke as a subprocess from the migs-tab main env — keeps its pinned deps
(numpy 1.26.4, transformers 4.45.1, lightning) out of our main env.

Usage:
    python migs_driver.py \\
        --audio /abs/path/to/input.wav \\
        --output /abs/path/to/output.mid \\
        --variant YMT3+ \\
        --batch-size 2
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# YourMT3 expects amt/src on sys.path before any of its imports resolve.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "amt" / "src"))


VARIANT_ARGS: dict[str, list[str]] = {
    "YMT3+": [
        "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt",
        "-p",
        "2024",
    ],
    "YPTF+Single (noPS)": [
        "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt",
        "-p",
        "2024",
        "-enc",
        "perceiver-tf",
        "-ac",
        "spec",
        "-hop",
        "300",
        "-atc",
        "1",
    ],
    "YPTF+Multi (PS)": [
        "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt",
        "-p",
        "2024",
        "-tk",
        "mc13_full_plus_256",
        "-dec",
        "multi-t5",
        "-nl",
        "26",
        "-enc",
        "perceiver-tf",
        "-ac",
        "spec",
        "-hop",
        "300",
        "-atc",
        "1",
    ],
    "YPTF.MoE+Multi (noPS)": [
        "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
        "-p",
        "2024",
        "-tk",
        "mc13_full_plus_256",
        "-dec",
        "multi-t5",
        "-nl",
        "26",
        "-enc",
        "perceiver-tf",
        "-sqr",
        "1",
        "-ff",
        "moe",
        "-wf",
        "4",
        "-nmoe",
        "8",
        "-kmoe",
        "2",
        "-act",
        "silu",
        "-epe",
        "rope",
        "-rp",
        "1",
        "-ac",
        "spec",
        "-hop",
        "300",
        "-atc",
        "1",
    ],
}


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _run(audio_path: Path, output_midi: Path, variant: str, batch_size: int) -> None:
    import torch
    import torchaudio

    # Defer YourMT3+ imports until sys.path has been adjusted.
    from model_helper import load_model_checkpoint  # noqa: PLC0415
    from utils.audio import slice_padded_array  # noqa: PLC0415
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes  # noqa: PLC0415
    from utils.note2event import mix_notes  # noqa: PLC0415
    from utils.utils import write_model_output_as_midi  # noqa: PLC0415

    if variant not in VARIANT_ARGS:
        raise SystemExit(f"unknown variant {variant!r}; pick one of {list(VARIANT_ARGS)}")

    device_str = _select_device()
    # Use fp32 on MPS — bf16/fp16 paths in upstream still hit cuda-only kernels.
    precision = "32" if device_str != "cuda" else "16"
    args = [*VARIANT_ARGS[variant], "-pr", precision]
    print(f"[migs] device={device_str} precision={precision} variant={variant}", flush=True)

    model = load_model_checkpoint(args=args, device=device_str)
    model.to(device_str)
    model.eval()

    audio, sr = torchaudio.load(uri=str(audio_path))
    audio = torch.mean(audio, dim=0).unsqueeze(0)
    target_sr = model.audio_cfg["sample_rate"]
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    frames = model.audio_cfg["input_frames"]
    audio_segments = slice_padded_array(audio, frames, frames)
    audio_segments = torch.from_numpy(audio_segments.astype("float32")).to(device_str).unsqueeze(1)
    print(f"[migs] segments={audio_segments.shape[0]} frames_per_seg={frames}", flush=True)

    with torch.no_grad():
        pred_token_arr, _ = model.inference_file(bsz=batch_size, audio_segments=audio_segments)

    num_channels = model.task_manager.num_decoding_channels
    n_items = audio_segments.shape[0]
    start_secs = [frames * i / target_sr for i in range(n_items)]
    pred_notes_in_file = []
    n_err = Counter()
    for ch in range(num_channels):
        pred_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped, _list_events, ch_err = model.task_manager.detokenize_list_batches(
            pred_arr_ch, start_secs, return_events=True
        )
        notes_ch, n_err_ch = merge_zipped_note_events_and_ties_to_notes(zipped)
        pred_notes_in_file.append(notes_ch)
        n_err += n_err_ch
    pred_notes = mix_notes(pred_notes_in_file)

    # write_model_output_as_midi expects a directory and a "track_name" stem.
    # We let it write into a temp dir then move into place.
    out_dir = output_midi.parent / "_mt3_tmp"
    out_dir.mkdir(parents=True, exist_ok=True)
    track_stem = "migs_mt3"
    write_model_output_as_midi(
        pred_notes,
        str(out_dir) + os.sep,
        track_stem,
        model.midi_output_inverse_vocab,
    )
    produced = out_dir / "model_output" / f"{track_stem}.mid"
    if not produced.exists():
        raise SystemExit(f"[migs] MT3 finished but no MIDI at {produced}")
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), str(output_midi))
    shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[migs] wrote {output_midi}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="migs-tab YourMT3+ driver")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variant",
        default="YMT3+",
        choices=list(VARIANT_ARGS),
        help="Model variant (default: YMT3+ — smallest, ~540 MB)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    ns = parser.parse_args()
    _run(ns.audio.resolve(), ns.output.resolve(), ns.variant, ns.batch_size)


if __name__ == "__main__":
    main()
