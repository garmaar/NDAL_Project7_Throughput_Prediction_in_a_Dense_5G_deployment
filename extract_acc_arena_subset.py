#!/usr/bin/env python3
"""
extract_acc_arena_subset.py
============================

STAGE 1 of the Project #7 pipeline: turn the raw "ACC Arena" 5G dataset
into a single, compact, flat CSV file that the analysis notebook can load
directly with `pandas.read_csv`.

--------------------------------------------------------------------------
WHERE THIS FITS IN THE PROJECT
--------------------------------------------------------------------------
    raw ACC Arena dataset (many wide per-metric CSV shards, one row per
    timestamp, one column per user)
                    |
                    |   <-- this script
                    v
    a single flat CSV: one row per (user, timestamp), with all metrics
    as columns  ->  acc_arena_subset.csv
                    |
                    v
    the analysis notebook (feature engineering, Random Forest / MLP,
    the X-sweep experiment, Federated Learning)

This script does NOT do any machine learning. Its only job is data
extraction and reshaping: picking a manageable subset of users out of
the full ~12,000-user ACC Arena venue, and reshaping the raw "wide"
files (one column per user) into the "long"/tidy format the notebook
expects (one row per user per timestamp).

--------------------------------------------------------------------------
EXPECTED INPUT LAYOUT (--dataset-root)
--------------------------------------------------------------------------
    <dataset-root>/
        Throughput_Acc_Arena/Throughput_UE_Id_<start>_<end>.csv
        BLER_Acc_Arena/BLER_UE_Id_<start>_<end>.csv
        PRB_Acc_Arena/PRBS_UE_Id_<start>_<end>.csv
        SINR_Acc_Arena/SINRDL_UE_Id_<start>_<end>.csv
        SINR_Acc_Arena/SINRUL_UE_Id_<start>_<end>.csv
        RU_Association_Acc_Arena/RU_UE_Id_<start>_<end>.csv
        Positions_Acc_Arena/Positions_Salt_Tar_UE_Id_<start>_<end>.csv

Each of these "shard" files covers a contiguous range of user IDs
(encoded in the filename as `<start>_<end>`), because the original
dataset is split into chunks that are too large to ship as one file.

--------------------------------------------------------------------------
OUTPUT (--output-path)
--------------------------------------------------------------------------
A single CSV with one row per (user_id, timestamp) and these columns,
in this order:

    venue, timestamp, user_id, traffic_type, ru_id,
    sinr_dl, sinr_ul, throughput_mbps, prb, bler, x, y, z

--------------------------------------------------------------------------
EXAMPLE USAGE
--------------------------------------------------------------------------
    python extract_acc_arena_subset.py \\
        --dataset-root "ACC Arena" \\
        --output-path data/raw/acc_arena_subset.csv \\
        --measurement-device-count 20 \\
        --target-user-start 20 \\
        --target-user-count 200 \\
        --time-stride 20

This is exactly the configuration used to produce the subset analyzed
in the project notebook: 20 operator measurement devices (user IDs
0-19) plus 200 target users (user IDs 20-219), sampled every 20th
aligned timestamp.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Matches a trailing "_<start>_<end>.csv" in a shard filename, e.g.
# "Throughput_UE_Id_0_999.csv" -> start=0, end=999. Every metric file for
# a given shard shares the same "<start>_<end>" suffix, which is how we
# line up the throughput/BLER/PRB/SINR/RU/position files for one shard.
_USER_RANGE_RE = re.compile(r"_(\d+)_(\d+)\.csv$")


def extract_range_bounds(path: Path) -> tuple[int, int]:
    """Parse the (start, end) user-ID range encoded in a shard filename."""
    match = _USER_RANGE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Could not extract UE ID range from file name: {path.name}")
    return int(match.group(1)), int(match.group(2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a compact, flat ACC Arena subset CSV for the Project 7 notebook.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ACC Arena"),
        help="Path to the original ACC Arena dataset directory (contains "
        "Throughput_Acc_Arena/, BLER_Acc_Arena/, etc.).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/raw/acc_arena_subset.csv"),
        help="Where to write the compact flat CSV.",
    )
    parser.add_argument(
        "--measurement-device-count",
        type=int,
        default=5,
        help="Number of operator measurement devices, assumed to be user IDs [0, X-1]. "
        "Their throughput is used as a feature for other users; they are never "
        "part of the prediction target set.",
    )
    parser.add_argument(
        "--target-user-start",
        type=int,
        default=5,
        help="First target user ID to extract (the users whose throughput is predicted).",
    )
    parser.add_argument(
        "--target-user-count",
        type=int,
        default=100,
        help="Number of target users to extract.",
    )
    parser.add_argument(
        "--time-stride",
        type=int,
        default=20,
        help="Keep one sample every N aligned timestamps (reduces row count).",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Fail early with a clear message instead of a confusing pandas error later."""
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {args.dataset_root}")
    if args.measurement_device_count < 0:
        raise ValueError("--measurement-device-count must be >= 0")
    if args.target_user_count <= 0:
        raise ValueError("--target-user-count must be > 0")
    if args.time_stride <= 0:
        raise ValueError("--time-stride must be >= 1")


def _user_ids_for_shard(selected_user_ids: list[int], shard_start: int, shard_end: int) -> list[int]:
    """Which of our selected user IDs fall inside this shard's [start, end] range."""
    return [user_id for user_id in selected_user_ids if shard_start <= user_id <= shard_end]


def _simple_metric_columns(user_ids: list[int]) -> list[str]:
    """Column names to read from a "simple" metric file (throughput, BLER, PRB, SINR, RU).

    These files have a named column per user (e.g. "entityStats id 0"), so we
    can select columns by name -- no positional bookkeeping needed here.
    """
    return ["time"] + [f"entityStats id {user_id}" for user_id in user_ids]


def _positions_usecols(user_ids: list[int], shard_start: int) -> list[int]:
    """Column *indices* to read from the positions file for this shard.

    Unlike the simple metric files, the positions file has no per-user column
    names -- each user occupies 5 anonymous columns (user_id, x, y, z,
    traffic_type) in a fixed block, in ascending user-ID order. This computes
    each selected user's block position in the *original, full* shard file
    (indices are always relative to shard_start here, which is correct because
    the raw file itself is contiguous from shard_start to shard_end).
    """
    usecols = [0]  # the "time" column
    for user_id in user_ids:
        offset = user_id - shard_start
        base = 1 + 5 * offset
        usecols.extend([base, base + 1, base + 2, base + 3, base + 4])
    return usecols


def _build_rows_for_user(
    *,
    user_id: int,
    shard_user_ids: list[int],
    timestamps: list[float],
    throughput_frame: pd.DataFrame,
    bler_frame: pd.DataFrame,
    prb_frame: pd.DataFrame,
    sinr_dl_frame: pd.DataFrame,
    sinr_ul_frame: pd.DataFrame,
    ru_frame: pd.DataFrame,
    positions_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Build the long-format rows (one per timestamp) for a single user."""
    throughput_column = f"entityStats id {user_id}"
    timestamp_set = set(timestamps)

    def selected_metric(frame: pd.DataFrame, column_name: str) -> pd.Series:
        selected = frame.loc[
            pd.to_numeric(frame["time"], errors="coerce").isin(timestamp_set),
            ["time", column_name],
        ].copy()
        selected["time"] = pd.to_numeric(selected["time"], errors="coerce")
        return selected.drop_duplicates(subset=["time"]).set_index("time")[column_name]

    throughput_series = selected_metric(throughput_frame, throughput_column)
    bler_series = selected_metric(bler_frame, throughput_column)
    prb_series = selected_metric(prb_frame, throughput_column)
    sinr_dl_series = selected_metric(sinr_dl_frame, throughput_column)
    sinr_ul_series = selected_metric(sinr_ul_frame, throughput_column)
    ru_series = selected_metric(ru_frame, throughput_column)

    # IMPORTANT: `positions_frame` was already read with only the selected
    # users' columns (see `_positions_usecols`), so it is a *reduced* frame.
    # pandas.read_csv keeps columns in their original file order regardless
    # of the order given to `usecols`, and `shard_user_ids` is sorted
    # ascending -- so a user's block position *within this reduced frame*
    # is simply how many selected users come before them, not
    # `user_id - shard_start` (that offset is only valid against the
    # original, full shard file, and silently breaks if the selected user
    # IDs are non-contiguous). We use the reduced-frame-local position here
    # to stay correct regardless of which users were selected.
    local_index = shard_user_ids.index(user_id)
    block_start = 1 + 5 * local_index
    position_columns = positions_frame.columns[block_start : block_start + 5]

    positions_selected = positions_frame.loc[
        pd.to_numeric(positions_frame.iloc[:, 0], errors="coerce").isin(timestamp_set),
        [positions_frame.columns[0], *position_columns],
    ].copy()
    positions_selected.columns = ["timestamp", "user_id_raw", "x", "y", "z", "traffic_type"]
    positions_selected["timestamp"] = pd.to_numeric(positions_selected["timestamp"], errors="coerce")
    positions_selected = positions_selected.drop_duplicates(subset=["timestamp"]).set_index("timestamp")

    aligned_index = pd.Index(sorted(timestamp_set), name="timestamp")
    user_frame = pd.DataFrame(index=aligned_index)
    user_frame["throughput_mbps"] = throughput_series.reindex(aligned_index)
    user_frame["bler"] = bler_series.reindex(aligned_index)
    user_frame["prb"] = prb_series.reindex(aligned_index)
    user_frame["sinr_dl"] = sinr_dl_series.reindex(aligned_index)
    user_frame["sinr_ul"] = sinr_ul_series.reindex(aligned_index)
    user_frame["ru_id"] = ru_series.reindex(aligned_index)
    user_frame["x"] = positions_selected["x"].reindex(aligned_index)
    user_frame["y"] = positions_selected["y"].reindex(aligned_index)
    user_frame["z"] = positions_selected["z"].reindex(aligned_index)
    user_frame["traffic_type"] = positions_selected["traffic_type"].reindex(aligned_index)

    # Drop timestamps where any metric was missing for this user, so every row in the final CSV is fully populated.
    user_frame = user_frame.dropna().reset_index()
    user_frame["user_id"] = user_id
    user_frame["venue"] = "ACC Arena"
    return user_frame.loc[
        :,
        [
            "venue",
            "timestamp",
            "user_id",
            "traffic_type",
            "ru_id",
            "sinr_dl",
            "sinr_ul",
            "throughput_mbps",
            "prb",
            "bler",
            "x",
            "y",
            "z",
        ],
    ].copy()


def _selected_timestamps(metric_frames: list[pd.DataFrame], positions_frame: pd.DataFrame, time_stride: int) -> list[float]:
    """Timestamps present in *every* file for this shard, then downsampled by stride."""
    common_timestamps: set[float] | None = None
    for frame in metric_frames + [positions_frame]:
        current_times = set(pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().tolist())
        common_timestamps = current_times if common_timestamps is None else common_timestamps & current_times
    if not common_timestamps:
        raise ValueError("No overlapping timestamps found across the selected files.")
    return sorted(common_timestamps)[::time_stride]


def main() -> None:
    args = parse_args()
    _validate_args(args)

    # --- Step 1: decide which user IDs we want ---------------------------
    measurement_user_ids = list(range(args.measurement_device_count))
    target_user_ids = list(range(args.target_user_start, args.target_user_start + args.target_user_count))
    selected_user_ids = sorted(set(measurement_user_ids + target_user_ids))

    print(f"Operator measurement devices: {args.measurement_device_count} "
          f"(user IDs 0-{args.measurement_device_count - 1})" if args.measurement_device_count else
          "Operator measurement devices: none")
    print(f"Target users: {args.target_user_count} "
          f"(user IDs {args.target_user_start}-{args.target_user_start + args.target_user_count - 1})")
    print(f"Total distinct users selected: {len(selected_user_ids)}")

    # --- Step 2: find which shard files exist -----------------------------
    throughput_dir = args.dataset_root / "Throughput_Acc_Arena"
    throughput_files = sorted(throughput_dir.glob("Throughput_UE_Id_*.csv"))
    if not throughput_files:
        raise ValueError(f"No throughput files found under {throughput_dir}")

    # --- Step 3: process each shard that overlaps our selected users ------
    extracted_frames: list[pd.DataFrame] = []
    for throughput_file in throughput_files:
        shard_start, shard_end = extract_range_bounds(throughput_file)
        shard_user_ids = _user_ids_for_shard(selected_user_ids, shard_start, shard_end)
        if not shard_user_ids:
            continue  # this shard doesn't contain any of our selected users

        print(f"Reading shard {shard_start}-{shard_end}: {len(shard_user_ids)} user(s) selected")

        suffix = f"{shard_start}_{shard_end}"
        bler_path = args.dataset_root / "BLER_Acc_Arena" / f"BLER_UE_Id_{suffix}.csv"
        prb_path = args.dataset_root / "PRB_Acc_Arena" / f"PRBS_UE_Id_{suffix}.csv"
        sinr_dl_path = args.dataset_root / "SINR_Acc_Arena" / f"SINRDL_UE_Id_{suffix}.csv"
        sinr_ul_path = args.dataset_root / "SINR_Acc_Arena" / f"SINRUL_UE_Id_{suffix}.csv"
        ru_path = args.dataset_root / "RU_Association_Acc_Arena" / f"RU_UE_Id_{suffix}.csv"
        positions_path = args.dataset_root / "Positions_Acc_Arena" / f"Positions_Salt_Tar_UE_Id_{suffix}.csv"

        metric_columns = _simple_metric_columns(shard_user_ids)
        throughput_frame = pd.read_csv(throughput_file, usecols=metric_columns)
        bler_frame = pd.read_csv(bler_path, usecols=metric_columns)
        prb_frame = pd.read_csv(prb_path, usecols=metric_columns)
        sinr_dl_frame = pd.read_csv(sinr_dl_path, usecols=metric_columns)
        sinr_ul_frame = pd.read_csv(sinr_ul_path, usecols=metric_columns)
        ru_frame = pd.read_csv(ru_path, usecols=metric_columns)
        positions_frame = pd.read_csv(positions_path, usecols=_positions_usecols(shard_user_ids, shard_start))

        timestamps = _selected_timestamps(
            [throughput_frame, bler_frame, prb_frame, sinr_dl_frame, sinr_ul_frame, ru_frame],
            positions_frame,
            args.time_stride,
        )

        for user_id in shard_user_ids:
            extracted_frames.append(
                _build_rows_for_user(
                    user_id=user_id,
                    shard_user_ids=shard_user_ids,
                    timestamps=timestamps,
                    throughput_frame=throughput_frame,
                    bler_frame=bler_frame,
                    prb_frame=prb_frame,
                    sinr_dl_frame=sinr_dl_frame,
                    sinr_ul_frame=sinr_ul_frame,
                    ru_frame=ru_frame,
                    positions_frame=positions_frame,
                )
            )

    if not extracted_frames:
        raise ValueError("No rows were extracted. Check the selected user range and dataset path.")

    # --- Step 4: combine, clean up dtypes, and write the flat CSV ---------
    output_frame = pd.concat(extracted_frames, ignore_index=True)
    output_frame["traffic_type"] = pd.to_numeric(output_frame["traffic_type"], errors="coerce").fillna(-1).astype(int)
    output_frame["ru_id"] = pd.to_numeric(output_frame["ru_id"], errors="coerce").fillna(-1).astype(int)
    output_frame = output_frame.sort_values(["timestamp", "user_id"]).reset_index(drop=True)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(args.output_path, index=False)

    print(
        f"\nDone. Wrote {len(output_frame)} rows for {output_frame['user_id'].nunique()} users "
        f"to {args.output_path}"
    )


if __name__ == "__main__":
    main()
