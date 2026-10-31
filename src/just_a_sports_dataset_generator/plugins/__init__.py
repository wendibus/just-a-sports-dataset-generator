"""Versioned plugin contracts and compatibility checks."""

from .contracts import PLUGIN_API_VERSION, PluginCompatibilityError, PluginMetadata

__all__ = ["PLUGIN_API_VERSION", "PluginCompatibilityError", "PluginMetadata"]
