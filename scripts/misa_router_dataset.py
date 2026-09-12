"""Compatibility shim. Prefer scripts/legacy for M8/pair-router experiments."""

from legacy.misa_router_dataset import *  # noqa: F401,F403
from legacy.misa_router_dataset import (  # noqa: F401
    RouterBatch,
    RouterDataset,
    _aggregate_context,
    load_router_dataset,
    load_sample_metadata,
)
