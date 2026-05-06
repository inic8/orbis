from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List
import os


@dataclass
class PipelineState:
    model_path: str
    output_dir: str
    config_path: str | None = None
    current_run_dir: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, step_name: str, status: str, details: Dict[str, Any]) -> None:
        self.history.append(
            {
                "step": step_name,
                "status": status,
                "details": details,
            }
        )


class BaseComponent(ABC):
    def __init__(self, name: str, params: Dict[str, Any] | None = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute component and return updated pipeline state."""

    def ensure_output_dir(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)