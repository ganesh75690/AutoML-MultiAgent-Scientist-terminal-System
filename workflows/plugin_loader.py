from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


class PluginRegistry:
    """Keep a list of dynamically loaded plugin contributions."""

    def __init__(self) -> None:
        self._plugins: list[dict[str, Any]] = []

    def register(self, plugin_name: str, payload: dict[str, Any]) -> None:
        self._plugins.append({"plugin_name": plugin_name, **payload})

    @property
    def plugins(self) -> list[dict[str, Any]]:
        return list(self._plugins)


class PluginLoader:
    """Load future agents or workflow extensions from the plugins folder."""

    def __init__(self, plugins_dir: Path) -> None:
        self.plugins_dir = plugins_dir

    def load(self) -> PluginRegistry:
        registry = PluginRegistry()
        if not self.plugins_dir.exists():
            return registry
        for plugin_file in self.plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            module_name = f"plugins.{plugin_file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            register = getattr(module, "register", None)
            if callable(register):
                register(registry)
        return registry
