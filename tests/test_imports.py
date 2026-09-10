"""Направление зависимостей: app/analysis/ не смотрит наружу (Задача 6).

Правило проверяется статически обходом AST-импортов, чтобы его не сломали
случайно. Обратное направление (bot/ и config.py импортируют analysis/)
разрешено и необходимо.
"""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_PREFIXES = ("app.bot", "app.data", "app.config")
ANALYSIS_DIR = Path(__file__).resolve().parent.parent / "app" / "analysis"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # относительные импорты внутри пакета
                continue
            if node.module:
                found.add(node.module)
    return found


def test_analysis_layer_does_not_import_outwards():
    offenders: list[str] = []
    for path in sorted(ANALYSIS_DIR.glob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_PREFIXES):
                offenders.append(f"{path.name} → {module}")
    assert not offenders, (
        "app/analysis/ импортирует наружу: " + ", ".join(offenders)
    )


def test_analysis_dir_is_not_empty():
    """Страховка от «зелёного» теста при опечатке в пути."""
    assert list(ANALYSIS_DIR.glob("*.py"))
