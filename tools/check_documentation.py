"""Check English source contracts and authored delivery content.

Run with the project interpreter from the workspace root. Supplied historical
references and third-party license texts are immutable inputs, not authored
project documentation. The language scan is a focused regression guard, not a
general-purpose language classifier.
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
missing = []
documented = 0
for path in (ROOT / "src/gregorithm").glob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if ast.get_docstring(node):
                documented += 1
            else:
                missing.append(
                    {
                        "file": path.name,
                        "name": getattr(node, "name", "module"),
                        "line": getattr(node, "lineno", 1),
                    }
                )

italian = re.compile(
    r"\b(?:analisi|esempio|verifica|campioni|corrente|riposo|provenienza|sintetici|sintetica|capacità|funzioni|stato|licenza|ambiente|implementata|completata|milestone\s+di)\b",
    re.I,
)
paths = [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md", ROOT / "CONTRIBUTING.md"]
paths += list((ROOT / "docs").rglob("*.md")) + list((ROOT / "src/gregorithm").glob("*.py"))
paths += list((ROOT / "tools").glob("*.py"))
paths.append(ROOT / "examples/getting_started.ipynb")
language_issues = []
for path in paths:
    if path.name == "check_documentation.py":
        continue  # This file deliberately contains the regression vocabulary.
    matches = sorted(set(italian.findall(path.read_text(encoding="utf8"))))
    if matches:
        language_issues.append({"file": str(path.relative_to(ROOT)), "matches": matches})

report = {
    "documented_core_nodes": documented,
    "missing_core_docstrings": missing,
    "authored_language_scan_issues": language_issues,
    "immutable_input_exceptions": "Supplied reference documents/historical sources and original dependency notices",
}
(ROOT / ".cache").mkdir(exist_ok=True)
(ROOT / ".cache/DOCUMENTATION_COVERAGE.json").write_text(
    json.dumps(report, indent=2), encoding="utf8"
)
print(json.dumps(report))
raise SystemExit(bool(missing or language_issues))
