"""Shared provenance, bootstrap, and checkpoint helpers for revision experiments.

The checkpoint format is intentionally simple: the first JSONL record contains the
run metadata and each subsequent record is one completed job.  A resumed run refuses
to mix records produced under different metadata or duplicate job keys.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from multiprocessing import Pool

import numpy as np


N_BOOT = 20_000


def bootstrap_ci(values, seed=0, n_boot=N_BOOT):
    """Percentile CI for a mean, resampling the supplied instance-level values."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("bootstrap_ci expects a nonempty one-dimensional sample")
    draws = np.random.default_rng(seed).choice(a, size=(n_boot, len(a))).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def git_commit(repo_dir):
    """Return the exact repository commit recorded in an output artifact."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True
    ).strip()


def canonical_key(value):
    """Stable textual representation used for checkpoint job identities."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)


def _read_checkpoint(path, metadata, row_key):
    rows = {}
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            raise RuntimeError(f"empty checkpoint: {path}")
        header = json.loads(first)
        # normalize through JSON before comparing: int dict keys (e.g. ITERS={20: ...})
        # become strings in the file, and a resume must not fail on that artifact
        if header != json.loads(json.dumps({"_meta": metadata}, sort_keys=True,
                                           allow_nan=True)):
            raise RuntimeError(
                f"checkpoint metadata mismatch; move the stale file before resuming: {path}"
            )
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            row = json.loads(line)
            key = canonical_key(row_key(row))
            if key in rows:
                raise RuntimeError(f"duplicate checkpoint key at {path}:{line_number}: {key}")
            rows[key] = row
    return rows


def run_checkpointed(jobs, worker, job_key, row_key, checkpoint_path, metadata,
                     processes=36):
    """Run missing jobs and append each completed result to a resumable JSONL file.

    Results are returned in the same deterministic order as ``jobs`` even though
    workers complete out of order.
    """
    # Normalize tuples and integer dictionary keys exactly as JSON will store them so
    # a resumed process compares like with like.
    metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=True))
    expected = [canonical_key(job_key(item)) for item in jobs]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("input job keys are not unique")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    rows = _read_checkpoint(checkpoint_path, metadata, row_key)
    unexpected = set(rows).difference(expected_set)
    if unexpected:
        raise RuntimeError(f"checkpoint contains {len(unexpected)} unexpected jobs")
    remaining = [item for item, key in zip(jobs, expected) if key not in rows]

    mode = "a" if os.path.exists(checkpoint_path) else "w"
    with open(checkpoint_path, mode, encoding="utf-8") as handle:
        # One writer per checkpoint file. A second process appending concurrently
        # interleaves lines and duplicates jobs (observed 2026-09-02 when another agent
        # re-launched a running driver); fail fast instead.
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                f"checkpoint is being written by another process: {checkpoint_path}"
            ) from error
        if mode == "w":
            handle.write(json.dumps({"_meta": metadata}, sort_keys=True) + "\n")
            handle.flush()
        if remaining:
            with Pool(processes) as pool:
                for row in pool.imap_unordered(worker, remaining, chunksize=1):
                    key = canonical_key(row_key(row))
                    if key in rows:
                        raise RuntimeError(f"worker returned duplicate job key: {key}")
                    if key not in expected_set:
                        raise RuntimeError(f"worker returned unexpected job key: {key}")
                    handle.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
                    handle.flush()
                    rows[key] = row

    missing = expected_set.difference(rows)
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} jobs after execution")
    return [rows[key] for key in expected]
