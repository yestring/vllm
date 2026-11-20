# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in request timing for the single-node NIXL Pull benchmark."""

import json
import logging
import os

KV_TIMELINE_PATH = os.environ.get("PD_KV_TIMELINE_PATH")
logger = logging.getLogger(__name__)


def trace_kv_event(stage: str, key: str, ts_ns: int, **fields: str | int) -> None:
    if not KV_TIMELINE_PATH:
        return
    event = {"stage": stage, "key": key, "ts_ns": ts_ns, **fields}
    # The local benchmark shares an append-only file. Each event is one write.
    try:
        with open(KV_TIMELINE_PATH, "ab", buffering=0) as output:
            output.write((json.dumps(event, separators=(",", ":")) + "\n").encode())
    except OSError:
        logger.warning("Could not append to the KV request timeline", exc_info=True)
