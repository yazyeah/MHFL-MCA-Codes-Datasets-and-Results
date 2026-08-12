from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from . import config
from .provenance import sha256_file, write_json


# Colorblind-safe, print-safe palette. Semantic colors must remain consistent
# across every reviewer figure (proposed/full = hero blue; degraded/negative = red).
PALETTE = {
    "hero": "#1F5A99",
    "hero_light": "#5B8FC4",
    "secondary": "#2A8C82",
    "positive": "#2E8540",
    "negative": "#B33F3F",
    "accent": "#D47A1F",
    "violet": "#7567B5",
    "neutral_dark": "#4D4D4D",
    "neutral_mid": "#8C8C8C",
    "neutral_light": "#D9D9D9",
    "black": "#222222",
}
CATEGORICAL = [
    PALETTE["hero"],
    PALETTE["accent"],
    PALETTE["positive"],
    PALETTE["violet"],
    PALETTE["secondary"],
    PALETTE["negative"],
    PALETTE["neutral_mid"],
]


@dataclass(frozen=True)
class FigureContract:
    """Machine-readable figure plan, adapted from the nature-figure workflow."""

    figure_id: str
    core_conclusion: str
    archetype: str
    evidence_hierarchy: Sequence[str]
    width_mm: float
    height_mm: float
    source_data: Sequence[str]
    replicate_unit: str
    center_statistic: str
    spread_definition: str
    claim_limits: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)
    target_journal: str = "Advanced Engineering Informatics"
    backend: str = "Python/Matplotlib"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def mm_to_inch(value_mm: float) -> float:
    return float(value_mm) / 25.4


def point_to_mm(value_pt: float) -> float:
    return float(value_pt) * 25.4 / 72.0


def apply_publication_style(font_size: float = config.FIGURE_FONT_PT) -> None:
    """Apply editable-text, final-size and visual-hierarchy defaults."""
    base = float(font_size)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": base,
            "axes.titlesize": base,
            "axes.titleweight": "normal",
            "axes.labelsize": base,
            "xtick.labelsize": max(base - 0.25, 5.5),
            "ytick.labelsize": max(base - 0.25, 5.5),
            "legend.fontsize": max(base - 0.5, 5.5),
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "errorbar.capsize": 2.0,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 0.9,
            "legend.handletextpad": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": True,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
        }
    )


def add_panel_label(ax: Any, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        str(label).lower(),
        transform=ax.transAxes,
        fontsize=config.PANEL_LABEL_PT,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["black"],
        clip_on=False,
    )


def figure_size(width_mm: float, height_mm: float) -> Tuple[float, float]:
    return mm_to_inch(width_mm), mm_to_inch(height_mm)


def _copy_source_data(source_data: Iterable[Path], destination: Path) -> List[Dict[str, Any]]:
    destination.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for source in source_data:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError("Figure source data not found: {0}".format(path))
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        target = destination / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(str(path), str(target))
        rows.append(
            {
                "source_path": str(path),
                "file": target.name,
                "sha256": sha256_file(target),
                "size_bytes": target.stat().st_size,
            }
        )
    pd.DataFrame(rows).to_csv(destination.parent / "source_data_index.csv", index=False)
    return rows


def _numeric(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return None if not match else float(match.group(0))


def audit_svg(svg_path: Path, expected_width_mm: float, expected_height_mm: float) -> Dict[str, Any]:
    root = ET.parse(str(svg_path)).getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    text_nodes = root.findall(".//svg:text", namespace)
    path_nodes = root.findall(".//svg:path", namespace)
    width = _numeric(root.attrib.get("width"))
    height = _numeric(root.attrib.get("height"))
    # Matplotlib normally writes pt. Convert explicitly when unit says pt.
    if root.attrib.get("width", "").strip().endswith("pt") and width is not None:
        width = point_to_mm(width)
    if root.attrib.get("height", "").strip().endswith("pt") and height is not None:
        height = point_to_mm(height)
    font_sizes: List[float] = []
    for node in text_nodes:
        candidate = node.attrib.get("font-size")
        style = node.attrib.get("style", "")
        if not candidate:
            match = re.search(r"font-size:\s*([0-9.]+)(?:px|pt)?", style)
            candidate = None if not match else match.group(1)
        number = _numeric(candidate)
        if number is not None:
            font_sizes.append(float(number))
    result = {
        "svg": str(svg_path),
        "editable_text_nodes": len(text_nodes),
        "path_nodes": len(path_nodes),
        "pass_editable_text": len(text_nodes) > 0,
        "width_mm": width,
        "height_mm": height,
        "expected_width_mm": float(expected_width_mm),
        "expected_height_mm": float(expected_height_mm),
        "minimum_detected_svg_font_size": min(font_sizes) if font_sizes else None,
    }
    if not result["pass_editable_text"]:
        raise RuntimeError("SVG contains no editable <text> nodes: {0}".format(svg_path))
    if width is not None:
        result["pass_width"] = abs(width - expected_width_mm) <= config.FIGURE_SIZE_TOLERANCE_MM
    if height is not None:
        result["pass_height"] = abs(height - expected_height_mm) <= config.FIGURE_SIZE_TOLERANCE_MM
    return result


def audit_pdf(pdf_path: Path, expected_width_mm: float, expected_height_mm: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {"pdf": str(pdf_path)}
    try:
        import fitz  # type: ignore

        document = fitz.open(str(pdf_path))
        if document.page_count != 1:
            result["page_count"] = document.page_count
        page = document[0]
        width_mm = point_to_mm(page.rect.width)
        height_mm = point_to_mm(page.rect.height)
        sizes: List[float] = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        sizes.append(float(span.get("size", 0.0)))
        result.update(
            {
                "width_mm": width_mm,
                "height_mm": height_mm,
                "pass_width": abs(width_mm - expected_width_mm) <= config.FIGURE_SIZE_TOLERANCE_MM,
                "pass_height": abs(height_mm - expected_height_mm) <= config.FIGURE_SIZE_TOLERANCE_MM,
                "minimum_pdf_text_pt": min(sizes) if sizes else None,
                "pass_pdf_text_floor": (min(sizes) >= config.MIN_PDF_GLYPH_PT) if sizes else None,
            }
        )
        document.close()
    except Exception as exc:
        result["fitz_audit_error"] = str(exc)

    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        try:
            process = subprocess.run(
                [pdffonts, str(pdf_path)],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            lines = [line for line in process.stdout.splitlines() if line.strip()]
            result["pdffonts_output"] = lines
            data_lines = lines[2:] if len(lines) > 2 else []
            result["pass_font_embedding"] = bool(data_lines) and all(" yes " in " {0} ".format(line.lower()) for line in data_lines)
        except Exception as exc:
            result["pdffonts_error"] = str(exc)
    else:
        result["pdffonts_unavailable"] = True
    return result


def audit_raster(path: Path, expected_width_mm: float, expected_height_mm: float, expected_dpi: int) -> Dict[str, Any]:
    with Image.open(str(path)) as image:
        dpi = image.info.get("dpi", (None, None))
        x_dpi = None if not dpi else float(dpi[0])
        y_dpi = None if not dpi else float(dpi[1])
        expected_px = (
            round(mm_to_inch(expected_width_mm) * expected_dpi),
            round(mm_to_inch(expected_height_mm) * expected_dpi),
        )
        return {
            "file": str(path),
            "pixel_width": int(image.width),
            "pixel_height": int(image.height),
            "dpi_x": x_dpi,
            "dpi_y": y_dpi,
            "expected_pixel_width": expected_px[0],
            "expected_pixel_height": expected_px[1],
            "pass_pixel_size": abs(image.width - expected_px[0]) <= 2 and abs(image.height - expected_px[1]) <= 2,
            "pass_dpi": x_dpi is None or x_dpi >= 0.98 * expected_dpi,
        }


def _manual_checklist(contract: FigureContract) -> str:
    items = [
        "[ ] Core claim is supported by the primary panel and is not stronger than the data.",
        "[ ] Replicate unit, center statistic, and uncertainty definition match the caption.",
        "[ ] No observations or conditions were silently excluded for visual convenience.",
        "[ ] Panel labels, legends, axes, units, model names, and SNR signs are consistent.",
        "[ ] All text is readable at the final {0:.1f} mm width; no labels collide or clip.".format(contract.width_mm),
        "[ ] Color remains interpretable in grayscale and for common color-vision deficiencies.",
        "[ ] Source-data files reproduce every quantitative mark in the figure.",
        "[ ] The exported SVG/PDF remains editable and the TIFF has submission resolution.",
    ]
    if contract.claim_limits:
        items.append("[ ] Claim limits are retained in the text/caption: {0}".format("; ".join(contract.claim_limits)))
    return "# Manual panel audit\n\n" + "\n".join("- " + item for item in items) + "\n"


def finalize_figure(
    fig: Any,
    output_stem: Path,
    contract: FigureContract,
    source_data: Iterable[Path],
    extra_qa: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    """Export a complete figure bundle without changing the declared physical size.

    Unlike automatic tight-bounding-box export, fixed-canvas export preserves the exact target
    dimensions. Layout must therefore be solved in the plotting function and any
    clipping becomes a visible QA failure rather than a silently resized figure.
    """
    stem = Path(output_stem)
    bundle_dir = stem.parent / (stem.name + "_bundle")
    bundle_dir.mkdir(parents=True, exist_ok=True)
    export_stem = bundle_dir / stem.name
    fig.set_size_inches(*figure_size(contract.width_mm, contract.height_mm), forward=True)

    exports = {
        "svg": export_stem.with_suffix(".svg"),
        "pdf": export_stem.with_suffix(".pdf"),
        "tiff": export_stem.with_suffix(".tiff"),
        "png": export_stem.with_suffix(".png"),
    }
    fig.savefig(str(exports["svg"]), bbox_inches=None, pad_inches=0)
    fig.savefig(str(exports["pdf"]), bbox_inches=None, pad_inches=0)
    fig.savefig(str(exports["tiff"]), dpi=config.RASTER_DPI, bbox_inches=None, pad_inches=0, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(str(exports["png"]), dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    source_manifest = _copy_source_data(source_data, bundle_dir / "source_data")
    contract_payload = contract.to_dict()
    contract_payload["source_data_manifest"] = source_manifest
    write_json(bundle_dir / "figure_contract.json", contract_payload)
    (bundle_dir / "manual_panel_audit.md").write_text(_manual_checklist(contract), encoding="utf-8")

    qa: Dict[str, Any] = {
        "figure_id": contract.figure_id,
        "target_width_mm": float(contract.width_mm),
        "target_height_mm": float(contract.height_mm),
        "font_family": mpl.rcParams["font.family"],
        "sans_serif_stack": mpl.rcParams["font.sans-serif"],
        "svg_fonttype": mpl.rcParams["svg.fonttype"],
        "pdf_fonttype": mpl.rcParams["pdf.fonttype"],
        "raster_dpi": config.RASTER_DPI,
        "minimum_pdf_glyph_pt_target": config.MIN_PDF_GLYPH_PT,
        "manual_panel_audit_required": True,
    }
    qa["svg_audit"] = audit_svg(exports["svg"], contract.width_mm, contract.height_mm)
    qa["pdf_audit"] = audit_pdf(exports["pdf"], contract.width_mm, contract.height_mm)
    qa["tiff_audit"] = audit_raster(exports["tiff"], contract.width_mm, contract.height_mm, config.RASTER_DPI)
    qa["png_audit"] = audit_raster(exports["png"], contract.width_mm, contract.height_mm, 300)
    if extra_qa:
        qa.update(dict(extra_qa))

    failures: List[str] = []
    for section in ("svg_audit", "pdf_audit", "tiff_audit", "png_audit"):
        for key, value in qa.get(section, {}).items():
            if key.startswith("pass_") and value is False:
                failures.append("{0}.{1}".format(section, key))
    qa["automatic_failures"] = failures
    qa["automatic_status"] = "FAIL" if failures else "PASS_WITH_MANUAL_REVIEW"
    write_json(bundle_dir / "qa_report.json", qa)

    return {
        "bundle_dir": bundle_dir,
        **exports,
        "contract": bundle_dir / "figure_contract.json",
        "qa": bundle_dir / "qa_report.json",
        "source_index": bundle_dir / "source_data_index.csv",
        "manual_audit": bundle_dir / "manual_panel_audit.md",
    }


def write_source_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
