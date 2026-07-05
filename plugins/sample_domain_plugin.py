from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    registry.register(
        "sample_domain_plugin",
        {
            "description": "Demonstrates the dynamic plugin architecture for future specialist agents.",
            "capabilities": ["domain_notes", "custom_routing"],
        },
    )
