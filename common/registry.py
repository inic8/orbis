from typing import Dict, Type

from .base import BaseComponent

_COMPONENTS: Dict[str, Type[BaseComponent]] = {}


def register_component(name: str):
    def decorator(cls: Type[BaseComponent]):
        _COMPONENTS[name] = cls
        return cls
    return decorator


def get_component(name: str) -> Type[BaseComponent]:
    if name not in _COMPONENTS:
        available = ", ".join(sorted(_COMPONENTS.keys()))
        raise KeyError(f"Component '{name}' not registered. Available: {available}")
    return _COMPONENTS[name]


def list_components() -> Dict[str, Type[BaseComponent]]:
    return dict(_COMPONENTS)