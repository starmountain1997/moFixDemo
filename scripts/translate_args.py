#!/usr/bin/env python3
"""
Translate CLI argument help text from English to Chinese using an LLM via pydantic-ai.

Reads all args from the two CLI modules, synthesizes English descriptions from
arg metadata (dest, type, default, choices), translates to Chinese, and saves
src/translations_zh.json.

Usage:
    python scripts/translate_args.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is importable (must precede imports from src/)
_SRC_DIR = Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from pydantic import BaseModel  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.models.anthropic import AnthropicModel  # noqa: E402

from arg_reflector import CLIReflector, _DEST_CHOICES, _ENUM_CHOICES  # noqa: E402

# ── API configuration ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed")

os.environ["ANTHROPIC_API_KEY"] = API_KEY
os.environ["ANTHROPIC_BASE_URL"] = BASE_URL


# ── English description synthesis ───────────────────────────────────────────────


def _type_label(type_name: str | None, choices: list | None) -> str:
    """Return a human-readable type description."""
    if type_name in ("check_positive_integer", "int"):
        return "integer"
    if type_name in ("check_positive_float", "check_prefix_cache_hit_rate", "float"):
        return "float (0.0–1.0)"
    if type_name in _DEST_CHOICES or choices:
        return "string (one of options)"
    return "string"


def synthesize_en(arg: Any) -> str:
    """Generate an English description for an ArgDef."""
    dest = str(getattr(arg, "dest", ""))
    group = str(getattr(arg, "group", ""))
    type_name = getattr(arg, "type_name", None)
    default = getattr(arg, "default", None)
    choices = getattr(arg, "choices", None)
    required = getattr(arg, "required", False)
    action = getattr(arg, "action", None)
    nargs = getattr(arg, "nargs", None)

    # Resolve choices
    resolved: list[str] = []
    if choices:
        for c in choices:
            s = str(c)
            if s.startswith("@enum:"):
                name = s[6:].split(".")[-1]
                resolved.extend(_ENUM_CHOICES.get(name, []))
            elif not s.startswith("@"):
                resolved.append(s)

    # Resolve default
    default_str = str(default) if default is not None else None
    if isinstance(default, str) and default.startswith("@var:"):
        default_str = None

    parts = dest.replace("_", " ").replace("-", " ").title()
    if group:
        parts = f"[{group}] {parts}"

    fragments = [parts, f"(type: {_type_label(type_name, resolved)})"]
    if required:
        fragments.append("required")
    if default_str and default_str not in ("None", "[]"):
        fragments.append(f"default: {default_str}")
    if resolved:
        sample = resolved[:5]
        fragments.append(
            f"options: {', '.join(sample)}{'…' if len(resolved) > 5 else ''}"
        )
    if action == "store_true":
        fragments.append("boolean flag")
    if nargs in ("*", "+"):
        fragments.append(f"space-separated list (nargs={nargs})")

    return " | ".join(fragments)


# ── Translation result model ──────────────────────────────────────────────────


class TranslationItem(BaseModel):
    key: str
    en: str
    zh: str


# ── Main ───────────────────────────────────────────────────────────────────────

CLI_DIR = Path(__file__).parent.parent / "msmodeling" / "cli"
COMMON_FILE = CLI_DIR / "utils.py"

SIM_REFLECTOR = CLIReflector(
    CLI_DIR / "inference" / "text_generate.py",
    common_file=COMMON_FILE,
    cli_module="cli.inference.text_generate",
)
OPT_REFLECTOR = CLIReflector(
    CLI_DIR / "inference" / "throughput_optimizer.py",
    common_file=COMMON_FILE,
    cli_module="cli.inference.throughput_optimizer",
)

SYSTEM_PROMPT = (
    "You are a professional technical translator. Translate English CLI argument "
    "descriptions to Simplified Chinese. Keep argument names, flags, type names, "
    "option values, and code in English. Use Chinese punctuation （，。：；？！）."
)

USER_TEMPLATE = (
    "Translate the following CLI argument descriptions to Chinese.\n"
    "Return a JSON array of objects with fields: key, en, zh.\n"
    "key: the identifier (e.g. cli.inference.text_generate:model_id)\n"
    "en: the original English description\n"
    "zh: the Chinese translation\n\n"
    "{lines}"
)


def main() -> None:
    lines: list[tuple[str, str]] = []  # (key, en_description)
    seen: set[str] = set()

    for reflector in [SIM_REFLECTOR, OPT_REFLECTOR]:
        cli_mod = reflector.cli_module
        for group in reflector.groups:
            for arg in group.args:
                key = f"{cli_mod}:{arg.dest}"
                if key in seen:
                    continue
                seen.add(key)
                en = synthesize_en(arg)
                lines.append((key, en))

    # Build prompt lines
    prompt_lines = "\n".join(f'  "{k}": "{en}"' for k, en in lines)
    user_prompt = USER_TEMPLATE.format(lines=prompt_lines)

    print(f"Synthesized {len(lines)} English descriptions")

    agent = Agent(
        AnthropicModel(MODEL_NAME),
        output_type=list[TranslationItem],
        system_prompt=SYSTEM_PROMPT,
        model_settings={"max_tokens": 4096},
    )

    # Batch to stay within context window
    translations: dict[str, str] = {}
    batch_size = 12
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        prompt_lines = "\n".join(f'  "{k}": "{en}"' for k, en in batch)
        user_prompt = USER_TEMPLATE.format(lines=prompt_lines)
        print(f"  Translating batch {i // batch_size + 1} ({i+1}–{i+len(batch)})...")
        result = agent.run_sync(user_prompt)
        for item in result.output:
            translations[item.key] = item.zh

    missing = len(lines) - len(translations)
    if missing:
        print(f"WARNING: {missing}/{len(lines)} translations missing", file=sys.stderr)

    out_path = Path(__file__).parent.parent / "src" / "translations_zh.json"
    out_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2))
    print(f"Saved {len(translations)} translations to {out_path}")


if __name__ == "__main__":
    main()
