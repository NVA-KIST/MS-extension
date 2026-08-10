"""Port ScribbleTool + ModuleLauncher (UI-only modules) into extension_new."""
from __future__ import annotations

import ast
from pathlib import Path

EXT = Path(r"e:\KUPETCTMS\extension")
OUT_ROOT = Path(r"e:\KUPETCTMS\extension\extension_new\slicer_modules")


def split_module(src_name: str, folder: str, module: str, widget: str, logic: str, note: str):
    src = (EXT / src_name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)

    def extract(name: str) -> str:
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return "".join(lines[node.lineno - 1 : node.end_lineno])
        raise SystemExit(f"{src_name}: missing {name}")

    # preamble before Module class
    preamble = ""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == module:
            preamble = "".join(lines[: node.lineno - 1])
            break

    extras = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name not in (module, widget, logic):
            # helper classes like _SliceEventFilter
            extras.append("".join(lines[node.lineno - 1 : node.end_lineno]))

    out = OUT_ROOT / folder
    out.mkdir(parents=True, exist_ok=True)

    logic_src = extract(logic)
    (out / f"{logic}.py").write_text(
        f'''\
"""
{logic} — thin / UI-support Logic.
{note}
"""
from slicer.ScriptedLoadableModule import ScriptedLoadableModuleLogic

{logic_src}
''',
        encoding="utf-8",
    )

    extras_txt = ("\n\n".join(extras) + "\n\n") if extras else ""
    entry = f'''\
"""
{module} — Module + Widget (UI + triggers).
{note}
"""
{preamble}
try:
    from {logic} import {logic}
except ImportError:
    import importlib.util, os as _os
    _p = _os.path.join(_os.path.dirname(__file__), "{logic}.py")
    _spec = importlib.util.spec_from_file_location("{logic}", _p)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    {logic} = getattr(_mod, "{logic}")

{extras_txt}{extract(module)}

{extract(widget)}
'''
    (out / f"{module}.py").write_text(entry, encoding="utf-8")
    (out / "__init__.py").write_text(f'"""{folder} package."""\n', encoding="utf-8")
    print("ported", folder)


split_module(
    "ScribbleTool.py",
    "ScribbleTool",
    "ScribbleTool",
    "ScribbleToolWidget",
    "ScribbleToolLogic",
    "Almost all work is Markups UI — no lib algorithm required.",
)
split_module(
    "ModuleLauncher.py",
    "ModuleLauncher",
    "ModuleLauncher",
    "ModuleLauncherWidget",
    "ModuleLauncherLogic",
    "Launcher UI only — no lib algorithm required.",
)
print("done")
