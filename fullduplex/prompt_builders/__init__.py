"""Prompt builders for different generation styles."""

from .gpt4o_style import build_prompt as build_gpt4o_prompt
from .doubao_style import build_prompt as build_doubao_prompt

__all__ = ["build_gpt4o_prompt", "build_doubao_prompt"]
