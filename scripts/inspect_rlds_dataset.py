#!/usr/bin/env python3
"""
Inspect a TFDS/RLDS dataset stored on disk (data-only TFDS builder directory).

Example:
  python scripts/inspect_rlds_dataset.py \
    --builder_dir data/libero/libero_object_no_noops/1.0.0 \
    --split train \
    --max_episodes 50 \
    --save_samples_dir /tmp/rlds_samples
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image


def _maybe_decode_bytes(x: Any) -> str:
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, np.bytes_):
        return x.tobytes().decode("utf-8", errors="replace")
    return str(x)


def _safe_int_from_cardinality(card: tf.Tensor) -> Optional[int]:
    try:
        v = int(card.numpy())
    except Exception:
        return None
    # tf.data.UNKNOWN_CARDINALITY = -2, tf.data.INFINITE_CARDINALITY = -1
    if v < 0:
        return None
    return v


def _percentiles(x: np.ndarray, ps: Tuple[float, ...]) -> Dict[str, float]:
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    q = np.percentile(x, ps)
    return {f"p{int(p)}": float(v) for p, v in zip(ps, q)}


def _find_dataset_statistics_json(builder_dir: Path) -> Optional[Path]:
    candidates = sorted(builder_dir.glob("dataset_statistics_*.json"))
    if not candidates:
        return None
    # If multiple exist, pick the newest (mtime) as a reasonable default.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _print_feature_schema(features: tfds.features.FeaturesDict) -> None:
    print("Features (schema):")
    print(features)
    try:
        steps = features["steps"]
        if hasattr(steps, "feature"):
            step_features = steps.feature
        else:
            step_features = None
    except Exception:
        step_features = None

    if step_features is not None:
        print("\nStep keys:", list(step_features.keys()))
        if "observation" in step_features:
            obs = step_features["observation"]
        else:
            obs = None
        if obs is not None and hasattr(obs, "keys"):
            print("Observation keys:", list(obs.keys()))


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def inspect_dataset(
    builder_dir: Path,
    split: str,
    max_episodes: Optional[int],
    max_steps_per_episode: Optional[int],
    save_samples_dir: Optional[Path],
    sample_every_n_episodes: int,
    near_zero_eps: float,
) -> None:
    tf.get_logger().setLevel("ERROR")

    builder = tfds.builder_from_directory(str(builder_dir))
    builder.download_and_prepare(download_dir=None)  # no-op for builder_from_directory; keeps behavior consistent

    print(f"Builder dir: {builder_dir}")
    print(f"TFDS name: {builder.info.name}")
    print(f"Version: {builder.info.version}")
    print(f"Splits: {builder.info.splits}")
    _print_feature_schema(builder.info.features)

    stats_path = _find_dataset_statistics_json(builder_dir)
    if stats_path is not None:
        print(f"\nFound dataset statistics: {stats_path.name}")
        try:
            with stats_path.open("r") as f:
                stats = json.load(f)
            num_transitions = stats.get("num_transitions", None)
            num_trajectories = stats.get("num_trajectories", None)
            print(f"  num_trajectories: {num_trajectories}")
            print(f"  num_transitions: {num_transitions}")
            if "action" in stats:
                print("  action: keys =", list(stats["action"].keys()))
            if "proprio" in stats:
                print("  proprio: keys =", list(stats["proprio"].keys()))
        except Exception as e:
            print(f"  (warning) failed to read dataset statistics: {type(e).__name__}: {e}")

    ds = builder.as_dataset(split=split, shuffle_files=False)

    episode_lengths: list[int] = []
    instruction_counter: Counter[str] = Counter()
    total_steps = 0

    action_dim_values: list[np.ndarray] = []
    proprio_dim_values: list[np.ndarray] = []
    near_zero_count = 0
    gripper_open_count = 0
    gripper_close_count = 0

    for ep_idx, episode in enumerate(tfds.as_numpy(ds)):
        if max_episodes is not None and ep_idx >= max_episodes:
            break

        steps = episode["steps"]
        # When using tfds.as_numpy, nested tf.data.Dataset becomes a python iterable of dicts.
        ep_len = 0

        for st_idx, step in enumerate(steps):
            if max_steps_per_episode is not None and st_idx >= max_steps_per_episode:
                break

            ep_len += 1
            total_steps += 1

            if st_idx == 0 and "language_instruction" in step:
                instruction_counter[_maybe_decode_bytes(step["language_instruction"]).strip()] += 1

            if "action" in step:
                a = np.asarray(step["action"], dtype=np.float32)
                action_dim_values.append(a)
                if a.shape[0] >= 7:
                    # First 6 dims: motion; last dim: gripper (dataset-dependent encoding)
                    if float(np.linalg.norm(a[:6])) <= near_zero_eps:
                        near_zero_count += 1
                    if a[6] >= 0.5:
                        gripper_open_count += 1
                    if a[6] <= -0.5:
                        gripper_close_count += 1

            obs = step.get("observation", {})
            if "state" in obs:
                proprio_dim_values.append(np.asarray(obs["state"], dtype=np.float32))

            if (
                save_samples_dir is not None
                and sample_every_n_episodes > 0
                and (ep_idx % sample_every_n_episodes == 0)
                and st_idx == 0
            ):
                # Save episode-idx step-0 images
                if "image" in obs:
                    _save_image(obs["image"], save_samples_dir / f"ep{ep_idx:04d}_step0_image.png")
                if "wrist_image" in obs:
                    _save_image(obs["wrist_image"], save_samples_dir / f"ep{ep_idx:04d}_step0_wrist.png")

        episode_lengths.append(ep_len)

    print("\n=== Summary ===")
    print(f"Episodes scanned: {len(episode_lengths)}")
    print(f"Total steps scanned: {total_steps}")

    if episode_lengths:
        lens = np.asarray(episode_lengths, dtype=np.int64)
        print(
            "Episode length: "
            f"min={lens.min()} mean={lens.mean():.1f} median={np.median(lens):.1f} max={lens.max()}"
        )

    print("\nTop language instructions (by episode):")
    for text, c in instruction_counter.most_common(20):
        print(f"  {c:5d}  {text}")

    if action_dim_values:
        actions = np.stack(action_dim_values, axis=0)  # [N, 7]
        print("\nAction stats (raw, over scanned steps):")
        for d in range(actions.shape[1]):
            vals = actions[:, d]
            ps = _percentiles(vals, (1, 50, 99))
            print(
                f"  dim{d}: mean={vals.mean(): .4f} std={vals.std(): .4f} "
                f"min={vals.min(): .4f} {ps['p1']=:.4f} {ps['p50']=:.4f} {ps['p99']=:.4f} max={vals.max(): .4f}"
            )

        near_zero_frac = near_zero_count / max(1, total_steps)
        print(f"\nNear-zero motion fraction (||action[:6]|| <= {near_zero_eps}): {near_zero_frac:.4%}")
        if gripper_open_count + gripper_close_count > 0:
            print(
                "Gripper approx counts (thresholded): "
                f"open(a[6]>=0.5)={gripper_open_count}, close(a[6]<=-0.5)={gripper_close_count}"
            )

    if proprio_dim_values:
        proprio = np.stack(proprio_dim_values, axis=0)  # [N, 8]
        print("\nProprio/state stats (observation.state, raw):")
        for d in range(proprio.shape[1]):
            vals = proprio[:, d]
            ps = _percentiles(vals, (1, 50, 99))
            print(
                f"  dim{d}: mean={vals.mean(): .4f} std={vals.std(): .4f} "
                f"min={vals.min(): .4f} {ps['p1']=:.4f} {ps['p50']=:.4f} {ps['p99']=:.4f} max={vals.max(): .4f}"
            )

    if save_samples_dir is not None:
        print(f"\nSaved sample images to: {save_samples_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--builder_dir",
        type=Path,
        default=Path("data/libero/libero_object_no_noops/1.0.0"),
        help="Path to TFDS builder directory (contains dataset_info.json, features.json, and *.tfrecord shards).",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--save_samples_dir", type=Path, default=None)
    parser.add_argument("--sample_every_n_episodes", type=int, default=25)
    parser.add_argument("--near_zero_eps", type=float, default=1e-3)
    args = parser.parse_args()

    if not args.builder_dir.exists():
        raise SystemExit(f"builder_dir does not exist: {args.builder_dir}")

    inspect_dataset(
        builder_dir=args.builder_dir,
        split=args.split,
        max_episodes=None if args.max_episodes <= 0 else int(args.max_episodes),
        max_steps_per_episode=args.max_steps_per_episode,
        save_samples_dir=args.save_samples_dir,
        sample_every_n_episodes=int(args.sample_every_n_episodes),
        near_zero_eps=float(args.near_zero_eps),
    )


if __name__ == "__main__":
    main()
