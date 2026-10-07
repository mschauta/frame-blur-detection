"""Render the publication summaries as editable SVG, using published aggregates.

Python standard library only. Run from any directory. No private images, training,
inference, or image editing is performed. See README.md and codec_criterion.json.
"""

import csv
import html
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BG, PANEL, INK, MUTED = "#111b2b", "#1b2a40", "#f2f6fc", "#b6c7dd"
COLORS = ["#72b7ff", "#c4adff", "#6be1ce"]


class SVG:
    def __init__(self, width, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            f'<rect width="{width}" height="{height}" fill="{BG}"/>',
            '<g font-family="DejaVu Sans, Arial, sans-serif">',
        ]

    def rect(self, x, y, w, h, fill=PANEL, radius=14):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{fill}"/>')

    def text(self, x, y, value, size=23, color=INK, weight="400", anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{html.escape(str(value))}</text>')

    def lines(self, x, y, values, size=22, color=MUTED, step=32):
        for i, value in enumerate(values):
            self.text(x, y + i * step, value, size, color)

    def save(self, path):
        path.write_text("\n".join(self.parts + ["</g>", "</svg>", ""]), encoding="utf-8")


def published_metrics():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    section = text.split("Last epoch (epoch 99):", 1)[1].split("On the held-out photos", 1)[0]
    rows = re.findall(r"\| (RGB|grayscale|edge fingerprint) \| ([\d.]+) \| "
                      r"[\d.]+% / ([\d.]+)% / [\d.]+% \| ([\d.]+) \|", section)
    if [r[0] for r in rows] != ["RGB", "grayscale", "edge fingerprint"]:
        raise ValueError("Expected the three published epoch-99 rows")
    return rows


def heldout_completed():
    """Change the status only for the complete published six-checkpoint report."""
    path = ROOT / "reproduction" / "heldout" / "heldout_results.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    cohort = report.get("cohort", {})
    expected_counts = {"frames": 6592, "sharp_frames": 3296, "blurred_frames": 3296,
                       "video_groups": 7, "photo_recipes": 4000, "photo_originals": 2000}
    if any(cohort.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("Held-out report does not describe the complete frozen test cohort")
    expected_models = {"rgb_ep099", "gray_ep099", "p99_ep099",
                       "rgb_ep073", "gray_ep058", "p99_ep053"}
    models = report.get("models", {})
    if set(models) != expected_models:
        raise ValueError("Held-out report must include all six fixed checkpoints")
    for key, model in models.items():
        expected_key = f"{model.get('input')}_ep{int(model.get('epoch', -1)):03d}"
        role = "primary" if model.get("epoch") == 99 else "secondary"
        if key != expected_key or model.get("selection_role") != role:
            raise ValueError("Held-out checkpoint identity or selection role is inconsistent")
        if any(not re.fullmatch(r"[0-9a-f]{64}", model.get(field, ""))
               for field in ("weights_sha256", "score_sha256", "score_sidecar_sha256")):
            raise ValueError("Held-out checkpoints require completed-score provenance hashes")
        for section in ("video_cluster_bootstrap", "photo_original_bootstrap"):
            for metric in ("auc", "sharp_kept_r95", "blurred_caught_r95"):
                estimate = model.get(section, {}).get(metric, {}).get("estimate")
                if not isinstance(estimate, (int, float)) or not math.isfinite(estimate) or not 0 <= estimate <= 1:
                    raise ValueError("Held-out checkpoint metrics are incomplete or invalid")
        if len(model.get("per_video", [])) != 7 or len(model.get("leave_one_video_out", [])) != 7:
            raise ValueError("Held-out report requires all seven group and sensitivity results")
    return True


COPY = {
    "en": {
        "title": "Frame Blur Detection",
        "subtitle": "Three input representations • one fixed study snapshot • reported validation",
        "dataset": "MATERIAL AND SUPERVISION",
        "cards": [
            ("Historical video snapshot", ["37 processed video files", "36 unique video groups", "One duplicate copy excluded"]),
            ("Native blur, including motion blur", ["Filtered VLM reference labels", "Blur types are not separated", "Measurement selects; never relabels"]),
            ("Synthetic construction labels", ["Added L = 0–3 px: sharp label", "Added L = 16–30 px: blurred label", "Strictly 3 < L < 16 excluded"]),
        ],
        "inputs": "INPUTS COMPARED WITH THE SAME TRAINING RECIPE",
        "inputcards": [
            ("RGB image", ["Full colour image", "ImageNet normalisation", "ConvNeXt-Small backbone"]),
            ("Grayscale image", ["Linear-light luminance", "sRGB encoding, three channels", "Full spatial frequency band"]),
            ("Signed edge fingerprint", ["Mesh-derived residual Y − K * Y", "Published bfloat16 rounding", "Per-image percentile + floor"]),
        ],
        "results": "EPOCH 99 • 6,602 VALIDATION FRAMES • 11 UNSEEN VIDEO GROUPS",
        "headers": ["Input", "Frame AUC", "Sharp kept at 95% recall", "Agreement-subset AUC"],
        "names": ["RGB", "Grayscale", "Edge fingerprint"],
        "reference": "Frame metrics measure agreement with VLM labels; they do not establish objective accuracy.",
        "reading": "WHAT THE COMPARISON SUPPORTS",
        "left": ["On this material, the signed residual alone", "supports discrimination close to full-image runs.", "One saved training seed per input; similar scores", "do not establish statistical equivalence."],
        "right": ["RGB ↔ grayscale transfer well; calibration differs.", "The residual also contains texture and artefacts.", "Ideal I − K is a weighted discrete Laplacian;", "rounding and the floor qualify scale invariance."],
        "footer": ["Internal 7-group test: not yet evaluated. External sets also selected alternative checkpoints.", "L = 0 means no added blur; it is not a guarantee that the original photo is objectively sharp.", "Native SVG generated by reproduction/render_figures.py • Full definitions and limits: METHOD.md"],
    },
    "hu": {
        "title": "Képkockák elmosódásának felismerése",
        "subtitle": "Három képi bemenet • rögzített kísérleti anyag • közölt validációs eredmények",
        "dataset": "KÍSÉRLETI ANYAG ÉS CÍMKÉK",
        "cards": [
            ("Történeti videós snapshot", ["37 feldolgozott videófájl", "36 egyedi videócsoport", "Egy duplikált másolat kihagyva"]),
            ("Natív blur, köztük motion blur", ["Szűrt VLM-referenciacímkék", "A blur típusai nincsenek szétválasztva", "A mérés válogat; nem címkéz át"]),
            ("Szintetikus konstrukciós címkék", ["Hozzáadott L = 0–3 px: éles címke", "Hozzáadott L = 16–30 px: blur címke", "Szigorúan 3 < L < 16 kizárva"]),
        ],
        "inputs": "ÖSSZEHASONLÍTOTT BEMENETEK, AZONOS TANÍTÁSI RECEPT",
        "inputcards": [
            ("RGB kép", ["Teljes színes kép", "ImageNet-normalizálás", "ConvNeXt-Small gerincháló"]),
            ("Szürkeárnyalatos kép", ["Lineáris fényben számított luminancia", "sRGB-kódolás, három csatorna", "Teljes térbeli frekvenciasáv"]),
            ("Előjeles él-ujjlenyomat", ["Meshből származó residual Y − K * Y", "Publikált bfloat16-kerekítés", "Képenkénti percentilis és floor"]),
        ],
        "results": "99. EPOCH • 6 602 VALIDÁCIÓS FRAME • 11 NEM TANÍTOTT VIDEÓCSOPORT",
        "headers": ["Bemenet", "Frame AUC", "Élesek megtartva, r95", "Méréssel egyező AUC"],
        "names": ["RGB", "Szürkeárnyalatos", "Él-ujjlenyomat"],
        "reference": "A frame-metrikák a VLM címkéivel való egyezést mérik, nem objektív pontosságot.",
        "reading": "MIT TÁMASZT ALÁ AZ ÖSSZEHASONLÍTÁS?",
        "left": ["Ezen az anyagon az előjeles residual önmagában", "a teljes képi futásokhoz közeli diszkriminációt ad.", "Bemenetenként egy tanítási seed; a hasonló", "értékek nem bizonyítanak statisztikai ekvivalenciát."],
        "right": ["RGB ↔ grayscale: jó transzfer, eltérő kalibráció.", "A residual textúrát és artefaktumokat is tartalmaz.", "Az ideális I − K súlyozott diszkrét Laplace-operátor.", "A kerekítés és a floor korlátozza a skálainvarianciát."],
        "footer": ["A 7 csoportos belső teszt még nincs kiértékelve. A külső anyagokon checkpointot is választottunk.", "L = 0: nincs hozzáadott blur; az eredeti fotó ettől még lehet elmosódott vagy defókuszos.", "Generált SVG: reproduction/render_figures.py • Részletes definíciók és korlátok: METHOD.md"],
    },
}


def infographic(language):
    c = COPY[language]
    footer = list(c["footer"])
    if heldout_completed():
        footer[0] = {
            "en": "Internal 7-group test completed: reproduction/heldout/RESULTS.md. External sets also selected checkpoints.",
            "hu": "A 7 csoportos teszt elkészült: reproduction/heldout/RESULTS.md. A külső anyagokon checkpointot is választottunk.",
        }[language]
    s = SVG(1400, 1120, c["title"], c["reference"])
    s.text(40, 65, c["title"], 43, weight="700")
    s.text(40, 107, c["subtitle"], 23, MUTED)
    s.text(40, 154, c["dataset"], 21, COLORS[0], "700")
    for i, (heading, body) in enumerate(c["cards"]):
        x = 40 + i * 445
        s.rect(x, 175, 430, 160)
        s.text(x + 18, 209, heading, 22, COLORS[i], "700")
        s.lines(x + 18, 247, body, 21, step=31)
    s.text(40, 375, c["inputs"], 21, COLORS[0], "700")
    for i, (heading, body) in enumerate(c["inputcards"]):
        x = 40 + i * 445
        s.rect(x, 397, 430, 156)
        s.text(x + 18, 431, heading, 24, COLORS[i], "700")
        s.lines(x + 18, 466, body, 21, step=31)
    s.text(40, 595, c["results"], 21, COLORS[0], "700")
    s.rect(40, 616, 1320, 213)
    positions = [62, 493, 818, 1148]
    for i, header in enumerate(c["headers"]):
        s.text(positions[i], 652, header, 20, MUTED, anchor="start" if i == 0 else "middle")
    for i, row in enumerate(published_metrics()):
        y = 701 + i * 49
        for j, value in enumerate([c["names"][i], row[1], row[2] + "%", row[3]]):
            s.text(positions[j], y, value, 26, COLORS[i], "700", "start" if j == 0 else "middle")
    s.text(40, 861, c["reference"], 21, MUTED)
    s.text(40, 903, c["reading"], 21, COLORS[0], "700")
    s.lines(40, 940, c["left"], 22, step=29)
    s.lines(731, 940, c["right"], 22, step=29)
    s.lines(40, 1072, footer, 17, step=21)
    s.save(ROOT / "figures" / ("hungarian.svg" if language == "hu" else "frame_blur_detection_infographic.svg"))


def codec_figure():
    with (ROOT / "reproduction" / "codec_group_medians.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # These anonymous labels are stable public aggregate IDs, not video identities.
    def median(group):
        matches = [r for r in rows if r["group"] == group]
        if len(matches) != 1:
            raise ValueError(f"Missing unique codec group {group!r}")
        return float(matches[0]["flat_e3"])
    groups = [("video_A", "Video A (100 frames)"), ("video_B", "Video B (100 frames)"),
              ("video_C", "Video C (100 frames)"), ("photo_png", "Uncoded PNG (40 photos)"),
              ("photo_crf18", "H.264 CRF 18 (40 photos)"), ("photo_crf23", "H.264 CRF 23 (40 photos)"),
              ("photo_crf28", "H.264 CRF 28 (40 photos)"), ("photo_crf33", "H.264 CRF 33 (40 photos)")]
    s = SVG(1400, 870, "Why photos go through H.264: exploratory residual statistics",
            "Group medians from 40 unblurred photos and 300 teacher-sharp native video frames. "
            "CRF 23 is closest by a composite heuristic, not by flat-region amplitude alone.")
    s.text(40, 65, "Why photos go through H.264", 43, weight="700")
    s.text(40, 108, "Exploratory probe: 40 unblurred photos • 3 × 100 teacher-sharp video frames", 24, MUTED)
    s.text(40, 157, "FLAT-REGION RESIDUAL AMPLITUDE", 21, COLORS[0], "700")
    s.text(40, 191, "Median of 1,000 × mean |d| in the lowest-residual 30% of 16 px cells", 22, MUTED)
    for i, (group, label) in enumerate(groups):
        y = 239 + i * 47
        value = median(group)
        s.text(40, y, label, 23)
        s.rect(432, y - 24, value * 850, 29, COLORS[0] if i < 3 else COLORS[2], 3)
        s.text(445 + value * 850, y, f"{value:.4f}", 23)
    s.rect(40, 634, 1320, 203)
    s.text(62, 674, "CRF 23 was closest by the combined criterion, not by amplitude alone.", 25, COLORS[2], "700")
    s.lines(62, 714, [
        "The heuristic combines flat/edge ratio, near-zero fraction, 8 px block structure and horizontal correlation.",
        "Residual proxies also include texture and coding artefacts; they are not isolated sensor-noise measurements.",
        "Ideal float64 residual, single intra-coded frame: no test of bfloat16 input or temporal P/B artefacts.",
        "This probe does not establish matching of the final mixed sharp/blurred training distribution.",
    ], 21, step=30)
    s.text(40, 861, "Data and exact criterion: reproduction/codec_group_medians.csv and codec_criterion.json • METHOD §6", 18, MUTED)
    s.save(ROOT / "figures" / "why_video_coding.svg")


if __name__ == "__main__":
    infographic("en")
    infographic("hu")
    codec_figure()
    print("Rendered three native SVG figures from published text and codec aggregates.")
