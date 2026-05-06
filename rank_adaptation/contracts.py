# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


INTERFACE_VERSION = "1.2"


class ArtifactType(str, Enum):
    MODEL = "model"
    CONFIG = "config"
    METRICS = "metrics"
    LOG = "log"
    REPORT = "report"
    BINARY = "binary"
    DATASET = "dataset"
    OTHER = "other"


class ComponentStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ArtifactDescriptor:
    name: str
    type: ArtifactType
    path: str
    format: str
    producer: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentResult:
    component_name: str
    status: ComponentStatus
    message: str = ""
    output_artifacts: List[ArtifactDescriptor] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyImprovement:
    before_ms: Optional[float] = None
    after_ms: Optional[float] = None
    absolute_ms: Optional[float] = None
    factor: Optional[float] = None
    reduction_pct: Optional[float] = None


@dataclass
class MemoryImprovement:
    before_mb: Optional[float] = None
    after_mb: Optional[float] = None
    absolute_mb: Optional[float] = None
    reduction_pct: Optional[float] = None


@dataclass
class RankAdaptationMetrics:
    params_before: int = 0
    params_after: int = 0
    parameter_reduction_pct: float = 0.0
    linear_params_before: int = 0
    linear_params_after: int = 0
    linear_reduction_pct: float = 0.0
    checkpoint_size_reduction_pct: Optional[float] = None
    latency: Optional[LatencyImprovement] = None
    parameter_memory: Optional[MemoryImprovement] = None
    peak_memory: Optional[MemoryImprovement] = None


@dataclass
class RankAdaptationComponentResult(ComponentResult):
    rank_adaptation_metrics: Optional[RankAdaptationMetrics] = None


@dataclass
class PipelineState:
    run_id: str
    pipeline_name: str
    work_dir: str
    artifacts: List[ArtifactDescriptor] = field(default_factory=list)
    global_metadata: Dict[str, Any] = field(default_factory=dict)
    step_history: List[ComponentResult] = field(default_factory=list)

    def add_artifact(self, artifact: ArtifactDescriptor) -> None:
        self.artifacts.append(artifact)

    def add_result(self, result: ComponentResult) -> None:
        self.step_history.append(result)
        for artifact in result.output_artifacts:
            self.add_artifact(artifact)

    def get_latest_artifact(
        self,
        artifact_type: ArtifactType,
        name: Optional[str] = None,
    ) -> Optional[ArtifactDescriptor]:
        for artifact in reversed(self.artifacts):
            if artifact.type != artifact_type:
                continue
            if name is not None and artifact.name != name:
                continue
            return artifact
        return None


class ComponentInterface(ABC):
    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def validate_params(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_inputs(self, state: PipelineState) -> None:
        raise NotImplementedError

    @abstractmethod
    def run(self, state: PipelineState) -> ComponentResult:
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        raise NotImplementedError