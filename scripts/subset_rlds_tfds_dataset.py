#!/usr/bin/env python3
"""
Subset an existing TFDS RLDS dataset (stored on disk) into a new TFDS dataset directory.

Use case:
  - Take `.../libero_object_no_noops/1.0.0` and create a new dataset dir with
    exactly N trajectories (episodes), selecting at most K trajectories per "task".

For LIBERO datasets in this repo, "task" is defined by the per-episode
`language_instruction` (we use the first step's instruction as the episode key).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.core import dataset_info as dataset_info_lib
from tensorflow_datasets.core import example_serializer
from tensorflow_datasets.core import naming
from tensorflow_datasets.core import splits as splits_lib
from tensorflow_datasets.core import writer as writer_lib


def _disable_tf_gpu() -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        # Some TF builds might have already initialized devices; best-effort only.
        pass


def _task_key_from_episode(episode: dict) -> str:
    steps = episode["steps"]
    first_step = next(iter(steps.take(1)))
    language_instruction = first_step["language_instruction"].numpy()
    if isinstance(language_instruction, (bytes, bytearray)):
        language_instruction = language_instruction.decode("utf-8")
    return str(language_instruction)


def subset_dataset(
    *,
    src_version_dir: Path,
    dst_dataset_dir: Path,
    split: str,
    max_traj_per_task: int,
    num_tasks: int,
    disable_shuffling: bool,
    overwrite: bool,
) -> Path:
    if not src_version_dir.exists():
        raise FileNotFoundError(f"Missing source TFDS version directory: {src_version_dir}")
    if not (src_version_dir / "dataset_info.json").exists():
        raise FileNotFoundError(f"Not a TFDS dataset version directory (missing dataset_info.json): {src_version_dir}")

    src_builder = tfds.builder_from_directory(str(src_version_dir))
    version = str(src_builder.info.version)
    dst_version_dir = dst_dataset_dir / version

    if dst_version_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Destination already exists: {dst_version_dir} (pass --overwrite to replace)")
        shutil.rmtree(dst_version_dir)
    dst_version_dir.mkdir(parents=True, exist_ok=True)

    filename_template = naming.ShardedFileTemplate(
        dst_version_dir,
        dataset_name=src_builder.info.name,
        split=split,
        template="{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
        filetype_suffix=src_builder.info.file_format.value,
    )

    features = src_builder.info.features
    serialized_info = features.get_serialized_info()
    writer = writer_lib.Writer(
        serializer=example_serializer.ExampleSerializer(serialized_info),
        filename_template=filename_template,
        hash_salt=split,
        disable_shuffling=bool(disable_shuffling),
        file_format=src_builder.info.file_format,
    )

    seen_tasks: dict[str, int] = {}
    selected = 0

    ds = src_builder.as_dataset(split=split, shuffle_files=False)
    for idx, episode in enumerate(ds):
        task_key = _task_key_from_episode(episode)
        if seen_tasks.get(task_key, 0) >= max_traj_per_task:
            continue

        # Materialize only the selected episode (steps are not decoded unless we encode).
        episode_np = tfds.as_numpy(episode)
        encoded = features.encode_example(episode_np)
        writer.write(idx, encoded)

        seen_tasks[task_key] = seen_tasks.get(task_key, 0) + 1
        selected += 1
        print(f"[subset] selected {selected}/{num_tasks}: task={task_key}")

        if selected >= num_tasks:
            break

    if selected < num_tasks:
        raise RuntimeError(
            f"Only selected {selected} episodes, expected {num_tasks}. "
            f"Found tasks={len(seen_tasks)} (max_traj_per_task={max_traj_per_task})."
        )

    shard_lengths, num_bytes = writer.finalize()

    split_info = splits_lib.SplitInfo(
        name=split,
        shard_lengths=[int(x) for x in shard_lengths],
        num_bytes=int(num_bytes),
        filename_template=filename_template,
    )
    split_dict = splits_lib.SplitDict([split_info], dataset_name=src_builder.info.name)

    identity = dataset_info_lib.DatasetIdentity(
        name=src_builder.info.name,
        version=src_builder.info.version,
        data_dir=str(dst_version_dir),
        module_name=src_builder.info.module_name,
    )
    # Some DatasetInfo attributes differ across TFDS versions; use best-effort extraction.
    description = getattr(src_builder.info, "description", None)
    homepage = getattr(src_builder.info, "homepage", None)
    citation = getattr(src_builder.info, "citation", None)
    license_str = getattr(src_builder.info, "license", None)
    if license_str is None:
        license_str = getattr(src_builder.info.as_proto, "license", None) or None
    info = dataset_info_lib.DatasetInfo(
        builder=identity,
        features=features,
        split_dict=split_dict,
        # IMPORTANT:
        # - If this is True, TFDS treats the dataset as "ordered" and enforces ordering guards (e.g., forbids
        #   interleave_cycle_length != 1 when reading with shuffling disabled).
        # - For training, we generally don't need a globally-ordered dataset, so default is False to avoid
        #   TFDS errors when parallel-reading.
        disable_shuffling=bool(disable_shuffling),
        description=description,
        homepage=homepage,
        citation=citation,
        license=license_str,
    )
    info.as_proto.file_format = src_builder.info.as_proto.file_format
    info.write_to_directory(dst_version_dir)

    return dst_version_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_version_dir",
        type=Path,
        required=True,
        help="Source TFDS dataset version directory (e.g. data/libero/libero_object_no_noops/1.0.0).",
    )
    parser.add_argument(
        "--dst_dataset_dir",
        type=Path,
        required=True,
        help="Destination TFDS dataset directory (will create <dst_dataset_dir>/<version>/...).",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_traj_per_task", type=int, default=1)
    parser.add_argument("--num_tasks", type=int, default=10)
    parser.add_argument(
        "--disable_shuffling",
        action="store_true",
        help="Mark the destination TFDS dataset as ordered (disableShuffling=true). Default is false.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _disable_tf_gpu()

    out = subset_dataset(
        src_version_dir=args.src_version_dir,
        dst_dataset_dir=args.dst_dataset_dir,
        split=args.split,
        max_traj_per_task=int(args.max_traj_per_task),
        num_tasks=int(args.num_tasks),
        disable_shuffling=bool(args.disable_shuffling),
        overwrite=bool(args.overwrite),
    )
    print(f"[subset] done: {out}")


if __name__ == "__main__":
    main()
