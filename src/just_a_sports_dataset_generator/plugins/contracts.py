from __future__ import annotations

from dataclasses import dataclass

PLUGIN_API_VERSION = 1


class PluginCompatibilityError(ValueError):
    """Raised before incompatible plugin code is used."""


@dataclass(frozen=True, slots=True)
class PluginMetadata:
    plugin_id: str
    version: str
    plugin_api: int
    kind: str
    capabilities: tuple[str, ...]

    def validate(self) -> None:
        if not self.plugin_id or any(char.isspace() for char in self.plugin_id):
            raise PluginCompatibilityError("plugin_id must be non-empty and contain no whitespace")
        if self.plugin_api != PLUGIN_API_VERSION:
            raise PluginCompatibilityError(
                f"plugin {self.plugin_id!r} needs API {self.plugin_api}; "
                f"host provides API {PLUGIN_API_VERSION}"
            )
        if self.kind not in {"renderer", "sport", "exporter", "model"}:
            raise PluginCompatibilityError(f"unsupported plugin kind: {self.kind!r}")

