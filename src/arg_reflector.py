#!/usr/bin/env python3
"""
src/arg_reflector.py

Parse msmodeling CLI argparse definitions via AST and build Gradio UI components.
Works purely from source text — no need to import tensor_cast / serving_cast.

Library usage (inside a gr.Blocks() context):
    from src.arg_reflector import CLIReflector
    r = CLIReflector("msmodeling/cli/inference/text_generate.py",
                     common_file="msmodeling/cli/utils.py",
                     cli_module="cli.inference.text_generate")
    components = r.build_accordions(open_groups={"General Options", "LLM Options"})
    btn.click(r.make_handler(), inputs=list(components.values()), outputs=output)

CLI usage (inspect discovered args):
    python src/arg_reflector.py msmodeling/cli/inference/text_generate.py \\
                                msmodeling/cli/utils.py
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gradio as gr

# ── Enum/choices tables (avoids importing tensor_cast at parse time) ──────────

_ENUM_CHOICES: dict[str, list[str]] = {
    "QuantizeLinearAction": [
        "DISABLED",
        "W8A16_STATIC",
        "W8A8_STATIC",
        "W4A8_STATIC",
        "W8A16_DYNAMIC",
        "W8A8_DYNAMIC",
        "W4A8_DYNAMIC",
        "FP8",
        "MXFP4",
    ],
    "QuantizeAttentionAction": ["DISABLED", "INT8", "FP8"],
    "WordEmbeddingTPMode": ["col", "row"],
}

# Choices that can't be resolved from AST (e.g. dict.keys(), variable refs)
_DEST_CHOICES: dict[str, list[str]] = {
    "device": [
        "TEST_DEVICE",
        "ATLAS_800_A2_376T_64G",
        "ATLAS_800_A2_313T_64G",
        "ATLAS_800_A2_280T_64G",
        "ATLAS_800_A2_280T_64G_PCIE",
        "ATLAS_800_A2_280T_32G_PCIE",
        "ATLAS_800_A3_752T_128G_DIE",
        "ATLAS_800_A3_560T_128G_DIE",
    ],
    "log_level": ["debug", "info", "warning", "error", "critical"],
    "remote_source": ["huggingface", "modelscope"],
    "performance_model": ["analytic", "profiling"],
}

_INT_TYPES = frozenset({"int", "check_positive_integer"})
_FLOAT_TYPES = frozenset(
    {"float", "check_positive_float", "check_prefix_cache_hit_rate"}
)

_NONE_SENTINEL = "(不设置)"


# ── ArgDef data model ─────────────────────────────────────────────────────────


@dataclass
class ArgDef:
    flags: list[str]  # ["--num-queries"] or ["model_id"] for positional
    dest: str  # argparse dest attribute: "num_queries"
    group: str  # label from add_argument_group()
    type_name: str | None  # leaf type name: "int", "check_positive_integer", …
    default: Any  # literal default, or "@var:…" sentinel for enum attrs
    choices: list | None  # literal values or "@enum:…" sentinels
    required: bool
    help_text: str
    action: str | None  # "store_true", "append", …
    nargs: Any | None  # "*", "+", int, or None
    is_positional: bool = False

    # ── Choices resolution ────────────────────────────────────────────────────

    def resolved_choices(self) -> list[str] | None:
        """Return the final list of choice strings, resolving @enum: sentinels."""
        if self.dest in _DEST_CHOICES:
            return _DEST_CHOICES[self.dest]
        if not self.choices:
            return None
        out: list[str] = []
        for c in self.choices:
            s = str(c)
            if s.startswith("@enum:"):
                name = s[6:].split(".")[-1]
                out.extend(_ENUM_CHOICES.get(name, []))
            elif not s.startswith("@"):
                out.append(s)
        return out or None

    def resolved_default(self, choices: list[str] | None = None) -> Any:
        """Return the default value, resolving "@var:Enum.MEMBER" sentinels."""
        d = self.default
        if isinstance(d, str) and d.startswith("@var:"):
            bare = d[5:].split(".")[
                -1
            ]  # "QuantizeLinearAction.W8A8_DYNAMIC" → "W8A8_DYNAMIC"
            if choices and bare in choices:
                return bare
            return choices[0] if choices else None
        return d

    # ── CLI command fragment ──────────────────────────────────────────────────

    def to_cli_args(self, value: Any) -> list[str]:
        """Convert a UI value to the CLI argument fragment for this argument."""
        if self.is_positional:
            return [str(value)] if value else []

        flag = self.flags[0]

        if self.action == "store_true":
            return [flag] if value else []

        if value in (None, "", False, _NONE_SENTINEL):
            return []

        if self.action == "append":
            # Textbox with space-separated values → repeat flag for each
            parts = str(value).split() if isinstance(value, str) else list(value)
            result: list[str] = []
            for p in parts:
                result.extend([flag, p])
            return result

        if self.nargs in ("*", "+"):
            parts = str(value).split()
            return [flag] + parts if parts else []

        return [flag, str(value)]

    # ── Gradio component ──────────────────────────────────────────────────────

    def to_gradio(self, *, translation: str | None = None) -> gr.components.Component:
        """Build the matching Gradio component for this argument."""
        flag = self.flags[0] if self.flags else f"--{self.dest.replace('_', '-')}"
        label = flag + (" *" if self.required else "")
        info = (translation or self.help_text or "")[:500]

        choices = self.resolved_choices()
        default = self.resolved_default(choices)

        # Boolean flag ────────────────────────────────────────────────────────
        if self.action == "store_true":
            return gr.Checkbox(label=label, value=False, info=info)

        # Multi-select append ─────────────────────────────────────────────────
        if self.action == "append" and choices:
            return gr.Dropdown(
                choices=choices,
                label=label,
                value=None,
                multiselect=True,
                info=info,
            )

        # Dropdown (choices present) ──────────────────────────────────────────
        if choices:
            strs = choices
            nullable = default is None and not self.required
            if nullable:
                strs = [_NONE_SENTINEL] + strs
            val = (
                _NONE_SENTINEL
                if nullable
                else (str(default) if default is not None else strs[0])
            )
            if val not in strs:
                val = strs[0]
            return gr.Dropdown(choices=strs, label=label, value=val, info=info)

        # Multi-value text (nargs) ────────────────────────────────────────────
        if self.nargs in ("*", "+"):
            val_str = (
                " ".join(str(x) for x in default)
                if isinstance(default, list)
                else str(default or "")
            )
            return gr.Textbox(label=label, value=val_str, info=info)

        # Numeric ─────────────────────────────────────────────────────────────
        type_lo = (self.type_name or "").lower()
        if type_lo in _INT_TYPES:
            return gr.Number(
                label=label,
                value=default if isinstance(default, (int, float)) else 0,
                precision=0,
                info=info,
            )
        if type_lo in _FLOAT_TYPES:
            return gr.Number(
                label=label,
                value=default if isinstance(default, (int, float)) else 0.0,
                info=info,
            )

        # Fallback: text ──────────────────────────────────────────────────────
        return gr.Textbox(
            label=label,
            value=str(default) if default is not None else "",
            info=info,
        )


@dataclass
class GroupDef:
    name: str
    args: list[ArgDef] = field(default_factory=list)


# ── AST evaluation helpers ────────────────────────────────────────────────────


def _eval(node: ast.expr | None) -> Any:
    """Best-effort convert an AST expression to a Python value."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        # Uppercase first letter → class / enum / module-level constant
        return f"@var:{node.id}" if node.id[0].isupper() else node.id
    if isinstance(node, ast.Attribute):
        return f"{_eval(node.value)}.{node.attr}"
    if isinstance(node, ast.List):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return [_eval(e) for e in node.elts]
    if isinstance(node, ast.Call):
        func = _eval(node.func)
        if func == "list" and len(node.args) == 1:
            inner = _eval(node.args[0])
            if isinstance(inner, str) and inner.startswith("@var:"):
                return [f"@enum:{inner[5:]}"]  # list(SomeEnum) → enum sentinel
            if isinstance(inner, list):
                return inner
            return None  # e.g. list(dict.keys()) — use _DEST_CHOICES fallback
        if func == "range" and 1 <= len(node.args) <= 2:
            try:
                return list(range(*[n.value for n in node.args]))  # type: ignore[union-attr]
            except (AttributeError, TypeError):
                pass
        return f"@call:{func}"
    if isinstance(node, ast.ListComp):
        # [x.value for x in SomeEnum] → treat iter as enum
        iter_node = node.generators[0].iter
        if isinstance(iter_node, ast.Name):
            return [f"@enum:{iter_node.id}"]
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _eval(node.operand)
        return -v if isinstance(v, (int, float)) else None
    return None


def _dest_from_flags(flags: list[str]) -> str:
    long = next((f for f in flags if f.startswith("--")), flags[0])
    return long.lstrip("-").replace("-", "_")


def _parse_add_argument(call: ast.Call, group: str) -> ArgDef | None:
    """Extract an ArgDef from a single add_argument() Call node."""
    flags = [
        arg.value
        for arg in call.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    ]
    if not flags:
        return None

    is_positional = not flags[0].startswith("-")
    dest = _dest_from_flags(flags)
    if dest == "help":
        return None

    kw: dict[str, Any] = {k.arg: _eval(k.value) for k in call.keywords if k.arg}

    # type_name: strip @var:/@call: prefix, take leaf name
    raw_type = kw.get("type")
    type_name: str | None = None
    if isinstance(raw_type, str):
        type_name = raw_type.removeprefix("@var:").removeprefix("@call:").split(".")[-1]
    elif raw_type in ("int", "float", "str"):
        type_name = raw_type

    # action: only keep simple string actions
    action = kw.get("action")
    if not isinstance(action, str) or action.startswith("@"):
        action = None

    # choices: keep only literal lists or sentinel lists; discard unresolvable refs
    choices_raw = kw.get("choices")
    choices: list | None = None
    if isinstance(choices_raw, list):
        choices = [c for c in choices_raw if c is not None]

    return ArgDef(
        flags=flags,
        dest=dest,
        group=group,
        type_name=type_name,
        default=kw.get("default"),
        choices=choices,
        required=bool(kw.get("required", False)) or is_positional,
        help_text=str(kw.get("help") or ""),
        action=action,
        nargs=kw.get("nargs"),
        is_positional=is_positional,
    )


# ── File parser ───────────────────────────────────────────────────────────────


def parse_cli_file(path: Path) -> list[GroupDef]:
    """
    Parse a Python CLI source file and return argument groups in source order.
    The file must use argparse and call add_argument() on parser/group variables.
    """
    tree = ast.parse(path.read_text())

    # Process statements in source order
    stmts = sorted(
        [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.Expr))],
        key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
    )

    group_vars: dict[str, str] = {}  # variable name → group label
    groups: dict[str, GroupDef] = {}
    group_order: list[str] = []
    seen_dests: set[str] = set()

    def _ensure_group(name: str) -> GroupDef:
        if name not in groups:
            groups[name] = GroupDef(name)
            group_order.append(name)
        return groups[name]

    for node in stmts:
        # Track: foo = parser.add_argument_group("Label")
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument_group"
                and call.args
                and isinstance(call.args[0], ast.Constant)
            ):
                gname = str(call.args[0].value)
                _ensure_group(gname)
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        group_vars[t.id] = gname

        # Track: foo.add_argument(...)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "add_argument"
                and isinstance(call.func.value, ast.Name)
            ):
                var = call.func.value.id
                gname = group_vars.get(var, "General")
                arg = _parse_add_argument(call, gname)
                if arg and arg.dest not in seen_dests:
                    seen_dests.add(arg.dest)
                    _ensure_group(gname).args.append(arg)

    return [groups[g] for g in group_order if groups[g].args]


# ── CLIReflector ──────────────────────────────────────────────────────────────


class CLIReflector:
    """
    Parse msmodeling CLI source files and expose methods to build the Gradio UI
    and generate CLI commands from component values.

    Example::

        r = CLIReflector(
            "msmodeling/cli/inference/text_generate.py",
            common_file="msmodeling/cli/utils.py",
            cli_module="cli.inference.text_generate",
        )

        with gr.Blocks() as demo:
            comps = r.build_accordions(open_groups={"General Options", "LLM Options"})
            out = gr.Markdown()
            btn = gr.Button("Run")
            btn.click(r.make_handler(), inputs=list(comps.values()), outputs=out)
    """

    def __init__(
        self,
        cli_file: Path | str,
        *,
        common_file: Path | str | None = None,
        cli_module: str,
    ) -> None:
        self.cli_file = Path(cli_file)
        self.common_file = Path(common_file) if common_file else None
        self.cli_module = cli_module

        self._groups: list[GroupDef] = []
        self._components: dict[str, gr.components.Component] = {}
        self._arg_defs: dict[str, ArgDef] = {}
        self._load()

    def _load(self) -> None:
        seen: set[str] = set()

        def _merge(file_groups: list[GroupDef]) -> None:
            for g in file_groups:
                new_args = [a for a in g.args if a.dest not in seen]
                seen.update(a.dest for a in new_args)
                if not new_args:
                    continue
                existing = next((x for x in self._groups if x.name == g.name), None)
                if existing:
                    existing.args.extend(new_args)
                else:
                    self._groups.append(GroupDef(g.name, list(new_args)))

        # Common args first so positional model_id comes first
        if self.common_file and self.common_file.exists():
            _merge(parse_cli_file(self.common_file))
        _merge(parse_cli_file(self.cli_file))

        for g in self._groups:
            for a in g.args:
                self._arg_defs[a.dest] = a

    # ── Gradio UI construction ─────────────────────────────────────────────────

    def build_accordions(
        self,
        *,
        skip_dests: set[str] | None = None,
        open_groups: set[str] | None = None,
        translations: dict[str, str] | None = None,
    ) -> dict[str, gr.components.Component]:
        """
        Render all argument groups as gr.Accordion blocks.

        Must be called inside a ``with gr.Blocks():`` context.

        Returns an ordered ``{dest: component}`` dict that should be passed as
        ``inputs=list(result.values())`` in the button click handler.

        Args:
            translations: Optional ``{f"{cli_module}:{dest}": chinese_text}`` dict
                to replace the default help text with Chinese descriptions.
        """
        skip_dests = skip_dests or set()
        open_groups = open_groups or set()
        translations = translations or {}
        self._components.clear()

        for group in self._groups:
            visible = [a for a in group.args if a.dest not in skip_dests]
            if not visible:
                continue
            is_open = group.name in open_groups or any(a.required for a in visible)
            group_key = f"{self.cli_module}:group:{group.name}"
            group_label = translations.get(group_key, group.name)
            with gr.Accordion(group_label, open=is_open):
                for arg in visible:
                    key = f"{self.cli_module}:{arg.dest}"
                    zh = translations.get(key)
                    self._components[arg.dest] = arg.to_gradio(translation=zh)

        return dict(self._components)

    # ── Command building ───────────────────────────────────────────────────────

    def build_command(self, values: dict[str, Any]) -> list[str]:
        """Build the full CLI command list from a ``{dest: value}`` mapping."""
        cmd = ["python", "-m", self.cli_module]

        # Positional arg (model_id) must immediately follow the module name
        for arg in self._arg_defs.values():
            if arg.is_positional:
                cmd.extend(arg.to_cli_args(values.get(arg.dest, "")))

        for dest, arg in self._arg_defs.items():
            if not arg.is_positional:
                cmd.extend(arg.to_cli_args(values.get(dest)))

        return cmd

    # ── Gradio handler factory ─────────────────────────────────────────────────

    def make_handler(self) -> Any:
        """
        Return a function for ``gr.Button.click()``.

        The function accepts values in the same order as the components returned
        by ``build_accordions()``, plus a ``gr.Progress`` keyword argument.
        """
        dests = list(self._components)
        arg_defs = self._arg_defs

        def _handler(*args: Any, progress: gr.Progress = gr.Progress()) -> str:
            values: dict[str, Any] = dict(zip(dests, args))

            for arg in arg_defs.values():
                if arg.required and not values.get(arg.dest):
                    return f"## ❌ 错误\n**{arg.flags[0]}** 是必选参数，请填写后再试。"

            cmd = self.build_command(values)

            if progress:
                progress(0.5, desc="正在执行 msmodeling…")

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                out = f"**执行命令:** `{' '.join(cmd)}`\n\n"
                if result.stdout:
                    out += f"#### 标准输出 (stdout):\n```text\n{result.stdout}\n```\n"
                if result.stderr:
                    out += f"#### 标准错误 (stderr):\n```text\n{result.stderr}\n```\n"
                if result.returncode != 0:
                    out += f"\n> ⚠️ **命令执行失败，退出码: {result.returncode}**"
                return out
            except Exception as exc:
                return f"## ❌ 执行错误\n无法运行命令：{exc}"

        return _handler

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def groups(self) -> list[GroupDef]:
        return self._groups

    @property
    def components(self) -> dict[str, gr.components.Component]:
        return dict(self._components)


# ── CLI tool ──────────────────────────────────────────────────────────────────


def _print_groups(groups: list[GroupDef]) -> None:
    for g in groups:
        print(f"\n[{g.name}]")
        for a in g.args:
            choices = a.resolved_choices()
            choices_repr = (
                f"[{choices[0]!r}…]({len(choices)})"
                if choices and len(choices) > 4
                else repr(choices)
            )
            print(
                f"  {a.dest:<42} type={str(a.type_name):<28} "
                f"default={str(a.default)!r:<18} "
                f"req={str(a.required):<5} "
                f"action={a.action or '':<12} "
                f"nargs={str(a.nargs):<4} "
                f"choices={choices_repr}"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <cli_file.py> [common_utils.py]")
        sys.exit(1)

    _cli = Path(sys.argv[1])
    _common = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    print(f"Parsing CLI file : {_cli}")
    _groups = parse_cli_file(_cli)

    if _common:
        print(f"Parsing common   : {_common}")
        _seen: set[str] = set()
        _merged: list[GroupDef] = []
        for _g in parse_cli_file(_common) + _groups:
            _new = [a for a in _g.args if a.dest not in _seen]
            _seen.update(a.dest for a in _new)
            if _new:
                _ex = next((x for x in _merged if x.name == _g.name), None)
                if _ex:
                    _ex.args.extend(_new)
                else:
                    _merged.append(GroupDef(_g.name, _new))
        _groups = _merged

    _print_groups(_groups)
