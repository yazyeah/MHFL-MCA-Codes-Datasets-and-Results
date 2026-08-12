from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dependency-free static audit for the reviewer suite.")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path)
    return parser.parse_args()


def module_constants(config_path: Path) -> Set[str]:
    tree = ast.parse(config_path.read_text(encoding="utf-8"), filename=str(config_path), feature_version=(3, 9))
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def audit_file(path: Path, config_names: Set[str]) -> Dict[str, object]:
    source = path.read_text(encoding="utf-8")
    findings: List[Dict[str, object]] = []
    try:
        tree = ast.parse(source, filename=str(path), feature_version=(3, 9))
    except SyntaxError as exc:
        return {"file": str(path), "status": "FAIL", "findings": [{"level": "FAIL", "id": "PY39_PARSE", "message": str(exc)}]}

    for node in ast.walk(tree):
        match_node = getattr(ast, "Match", None)
        if match_node is not None and isinstance(node, match_node):
            findings.append({"level": "FAIL", "id": "PY39_MATCH", "line": node.lineno, "message": "match/case requires Python 3.10."})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Type-union syntax appears as BitOr in annotations and requires 3.10.
            parent_text = ast.get_source_segment(source, node) or ""
            if re.search(r"\b(?:None|str|int|float|Path|Dict|List|Tuple|Mapping|Sequence)\b", parent_text):
                findings.append({"level": "FAIL", "id": "PY39_UNION", "line": node.lineno, "message": "PEP 604 union syntax is not Python 3.9 compatible."})
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "config":
            if node.attr not in config_names:
                findings.append({"level": "FAIL", "id": "CONFIG_ATTR", "line": node.lineno, "message": "Unknown config attribute: {0}".format(node.attr)})
    if path.name != "static_audit.py":
        if re.search(r"bbox_inches\s*=\s*[\"']tight[\"']", source):
            findings.append({"level": "WARN", "id": "FIGURE_RESIZE", "message": "bbox_inches='tight' can silently change final physical dimensions."})
        if re.search(r"(?:cmap|colormap|palette)\s*=\s*[\"'](?:jet|rainbow|hsv)[\"']", source, re.IGNORECASE):
            findings.append({"level": "WARN", "id": "RAINBOW_CMAP", "message": "Potential non-publication-safe color mapping."})
    status = "FAIL" if any(item["level"] == "FAIL" for item in findings) else ("WARN" if findings else "PASS")
    return {"file": str(path), "status": status, "findings": findings}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_names = module_constants(root / "mhfl_review" / "config.py")
    files = sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    reports = [audit_file(path, config_names) for path in files]
    summary = {level: sum(report["status"] == level for report in reports) for level in ("PASS", "WARN", "FAIL")}
    payload = {"root": str(root), "python_target": "3.9", "summary": summary, "files": reports}
    output = args.json or (root / "provenance" / "static_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("Report:", output)
    if summary["FAIL"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
