from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOCAL_PRUNING_ROOT = Path(__file__).resolve().parent
LOCAL_REPO_ROOT = LOCAL_PRUNING_ROOT.parent


@dataclass(frozen=True)
class OrbisModules:
    instantiate_from_config: Callable[..., Any]
    StructuredPruningConfig: type
    apply_structured_pruning: Callable[..., Any]
    get_pruning_summary: Callable[..., Any]


def _normalize_orbis_checkout(candidate: str | Path | None) -> Path | None:
    if candidate is None:
        return None

    path = Path(candidate).expanduser().resolve()
    if (path / "orbis" / "util.py").exists():
        return path / "orbis"
    if (path / "util.py").exists():
        return path
    return None


def _candidate_checkouts(orbis_repo_path: str | Path | None, checkpoint_path: str | Path | None) -> list[Path]:
    candidates: list[Path] = []

    local_checkout = _normalize_orbis_checkout(LOCAL_REPO_ROOT)
    if local_checkout is not None:
        candidates.append(local_checkout)

    explicit_checkout = _normalize_orbis_checkout(orbis_repo_path)
    if explicit_checkout is not None and explicit_checkout not in candidates:
        candidates.append(explicit_checkout)

    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        for parent in [checkpoint.parent, *checkpoint.parents]:
            detected_checkout = _normalize_orbis_checkout(parent)
            if detected_checkout is not None and detected_checkout not in candidates:
                candidates.append(detected_checkout)

    env_checkout = _normalize_orbis_checkout(os.getenv("ORBIS_REPO_PATH"))
    if env_checkout is not None and env_checkout not in candidates:
        candidates.append(env_checkout)

    return candidates


def _ensure_preferred_sys_path() -> None:
    for search_path in [LOCAL_REPO_ROOT, LOCAL_PRUNING_ROOT]:
        search_path_str = str(search_path)
        if search_path_str in sys.path:
            sys.path.remove(search_path_str)
        sys.path.insert(0, search_path_str)


def _checkout_search_paths(checkout: Path) -> list[Path]:
    search_paths: list[Path] = []

    # Some upstream Orbis files import via `external.orbis...`, which requires
    # the workspace root to be importable when the checkout lives at `external/orbis`.
    if checkout.parent.name == "external":
        search_paths.append(checkout.parent.parent)

    search_paths.extend([checkout.parent, checkout])
    return search_paths


def _import_from_current_sys_path() -> OrbisModules:
    _ensure_preferred_sys_path()

    if (LOCAL_REPO_ROOT / "util.py").exists():
        util_module = importlib.import_module("util")
    else:
        try:
            util_module = importlib.import_module("orbis.util")
        except ModuleNotFoundError as error:
            if error.name not in {"orbis", "orbis.util"}:
                raise
            util_module = importlib.import_module("util")

    try:
        structured_config_module = importlib.import_module("orbis.pruning.structured_config")
        structured_pruning_module = importlib.import_module("orbis.pruning.structured_pruning")
    except ModuleNotFoundError as error:
        if error.name not in {"orbis.pruning", "orbis.pruning.structured_config", "orbis.pruning.structured_pruning"}:
            raise

        try:
            local_pruning_module = importlib.import_module("pruning.local_pruning")
        except ModuleNotFoundError as fallback_error:
            if fallback_error.name not in {"pruning", "pruning.local_pruning"}:
                raise
            local_pruning_module = importlib.import_module("local_pruning")
        return OrbisModules(
            instantiate_from_config=util_module.instantiate_from_config,
            StructuredPruningConfig=local_pruning_module.StructuredPruningConfig,
            apply_structured_pruning=local_pruning_module.apply_structured_pruning,
            get_pruning_summary=local_pruning_module.get_pruning_summary,
        )

    return OrbisModules(
        instantiate_from_config=util_module.instantiate_from_config,
        StructuredPruningConfig=structured_config_module.StructuredPruningConfig,
        apply_structured_pruning=structured_pruning_module.apply_structured_pruning,
        get_pruning_summary=structured_pruning_module.get_pruning_summary,
    )


def _import_orbis_modules(orbis_repo_path: str | Path | None, checkpoint_path: str | Path | None) -> OrbisModules:
    import_errors: list[ModuleNotFoundError] = []

    try:
        return _import_from_current_sys_path()
    except ModuleNotFoundError as error:
        import_errors.append(error)

    for checkout in _candidate_checkouts(orbis_repo_path, checkpoint_path):
        for search_path in _checkout_search_paths(checkout):
            search_path_str = str(search_path)
            if search_path_str not in sys.path:
                sys.path.insert(0, search_path_str)

        try:
            return _import_from_current_sys_path()
        except ModuleNotFoundError as error:
            import_errors.append(error)
            continue

    missing_modules = [error.name or "<unknown>" for error in import_errors]
    unique_missing_modules = ", ".join(dict.fromkeys(missing_modules))
    raise ModuleNotFoundError(
        "Unable to import the Orbis codebase. Provide the repo root via orbis_repo_path or ORBIS_REPO_PATH, "
        "or make the current checkout importable. "
        f"Missing module(s): {unique_missing_modules}"
    )


def resolve_orbis_modules(
    *,
    orbis_repo_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> OrbisModules:
    return _import_orbis_modules(orbis_repo_path, checkpoint_path)
