#!/usr/bin/env python3
"""
Translate CLI argument help text from English to Chinese using an LLM via async-openai.

Reads all args from the two CLI modules, synthesizes English descriptions from
arg metadata (dest, type, default, choices), translates to Chinese, and saves
src/translations_zh.json.

Usage:
    python scripts/translate_args.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

# Ensure src/ is importable (must precede imports from src/)
_SRC_DIR = Path(__file__).parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from openai import AsyncOpenAI  # noqa: E402

from arg_reflector import CLIReflector  # noqa: E402

# ── API configuration ───────────────────────────────────────────────────────────

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.minimaxi.com/v1")
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "MiniMax-M2.7-highspeed")


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
    "names to Simplified Chinese. Keep argument names in English. "
    "Use Chinese punctuation （，。：；？！）."
)

USER_TEMPLATE = (
    "Translate the following CLI argument names to Simplified Chinese.\n"
    "Return a JSON array of objects with fields: key, en, zh.\n"
    "key: the identifier (e.g. cli.inference.text_generate:model_id)\n"
    "en: the original English argument name\n"
    "zh: the Chinese translation (just the translated name)\n\n"
    "{lines}"
)


async def translate_batch(
    client: AsyncOpenAI, batch: list[tuple[str, str]]
) -> dict[str, str]:
    """Translate a batch of argument names."""
    prompt_lines = "\n".join(f'  "{k}": "{en}"' for k, en in batch)
    user_prompt = USER_TEMPLATE.format(lines=prompt_lines)

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4096,
    )
    content = response.choices[0].message.content
    if content is None:
        return {}
    start = content.find("[")
    end = content.rfind("]") + 1
    if start == -1 or end == 0:
        return {}
    json_str = content[start:end]
    try:
        items = json.loads(json_str)
    except json.JSONDecodeError:
        return {}
    return {item["key"]: item["zh"] for item in items}


async def main() -> None:
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
                en = str(arg.help_text or "")
                lines.append((key, en))

    print(f"Synthesized {len(lines)} English descriptions")

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    semaphore = asyncio.Semaphore(8)

    async def translate_batch_with_semaphore(
        batch: list[tuple[str, str]],
    ) -> dict[str, str]:
        async with semaphore:
            return await translate_batch(client, batch)

    batch_size = 12
    batches = [lines[i : i + batch_size] for i in range(0, len(lines), batch_size)]
    tasks = [translate_batch_with_semaphore(batch) for batch in batches]
    results = await asyncio.gather(*tasks)
    translations: dict[str, str] = {}
    for result in results:
        translations.update(result)

    missing = len(lines) - len(translations)
    if missing:
        print(f"WARNING: {missing}/{len(lines)} translations missing", file=sys.stderr)

    out_path = Path(__file__).parent.parent / "src" / "translations_zh.json"
    out_path.write_text(json.dumps(translations, ensure_ascii=False, indent=2))
    print(f"Saved {len(translations)} translations to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
