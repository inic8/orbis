# SPDX-License-Identifier: MIT
# Author: Arunachalam Thirunavukkarasu
# Contributor: Dr Shashank Pathak
# Email: arunachalam.thirunavukkarasu@dlr.de
# Funding: German Research Project NXTAIM
# See LICENSE for the full MIT license text.

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOCAL_RANK_ADAPTATION_ROOT = Path(__file__).resolve().parent
LOCAL_REPO_ROOT = LOCAL_RANK_ADAPTATION_ROOT.parent


@dataclass(frozen=True)
class OrbisModules:
    instantiate_from_config: Callable[..., Any]


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
    for search_path in [LOCAL_REPO_ROOT, LOCAL_RANK_ADAPTATION_ROOT]:
        search_path_str = str(search_path)
        if search_path_str in sys.path:
            sys.path.remove(search_path_str)
        sys.path.insert(0, search_path_str)


def _checkout_search_paths(checkout: Path) -> list[Path]:
    search_paths: list[Path] = []
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

    return OrbisModules(instantiate_from_config=util_module.instantiate_from_config)


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