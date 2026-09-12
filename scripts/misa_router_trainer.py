"""Compatibility shim. Prefer scripts/legacy for M8 assignment-router training."""

from legacy.misa_router_trainer import *  # noqa: F401,F403
from legacy.misa_router_trainer import (  # noqa: F401
    TrainConfig,
    _static_assignment,
    assignment_loss,
    build_model,
    evaluate_router,
    save_checkpoint,
    train_router,
)
