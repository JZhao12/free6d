"""
Collect the local bop_toolkit scores of every evaluated BOP dataset into a table aligned
with Table 1 of the CNOS paper (segmentation AP, BOP'19 test targets).

usage: python -m src.scripts.summarize_bop_results [--root datasets/bop23_challenge] [--markdown out.md]

Two Free6D rows are reported when their scores exist: pyrender templates (method name
free6d, predictions *_template_pyrender0_*.json) and templates picked from the
BlenderProc4BOP train_pbr frames (method name free6dpbr, *_template_pbr0_*.json).
Datasets without local scores (ITODD and HB ship no test ground truth) fall back to the
scores returned by the BOP website, stored in docs/bop_website_scores/<method>_<ds>-test*_scores_bop22_coco_<task>.json.
"""
import argparse
import json
import os.path as osp

PAPER_ORDER = ["lmo", "tless", "tudl", "icbin", "itodd", "hb", "ycbv"]
PAPER_NAMES = {"lmo": "LM-O", "tless": "T-LESS", "tudl": "TUD-L", "icbin": "IC-BIN",
               "itodd": "ITODD", "hb": "HB", "ycbv": "YCB-V"}
# Table 1 of the CNOS paper (ICCV-W 2023): segmentation AP, model-based, BOP'19 targets
PAPER = {
    "CNOS (SAM + DINOv2 + Pyrender)":        {"lmo": 33.3, "tless": 38.3, "tudl": 35.8, "icbin": 27.2, "itodd": 14.8, "hb": 45.9, "ycbv": 57.6},
    "CNOS (SAM + DINOv2 + BlenderProc)":     {"lmo": 39.6, "tless": 39.7, "tudl": 39.1, "icbin": 28.4, "itodd": 28.2, "hb": 48.0, "ycbv": 59.5},
    "CNOS (FastSAM + DINOv2 + BlenderProc)": {"lmo": 39.7, "tless": 37.4, "tudl": 48.0, "icbin": 27.0, "itodd": 25.4, "hb": 51.1, "ycbv": 59.9},
}
# bop_toolkit method name -> (table name, rendering_type used in the prediction file name,
#                             BOP-website submission dir under <root>/results/)
OURS = {
    "free6d":    ("Free6D (SAM2 + DINOv3 + Pyrender)", "pyrender", "bop_submission"),
    "free6dpbr": ("Free6D (SAM2 + DINOv3 + BlenderProc)", "pbr", "bop_submission_pbr"),
}
SPLIT_SUFFIX = {"tless": "-primesense", "hb": "-primesense"}
PRED_FILE = "CustomSam2AutomaticMaskGenerator_template_{rendering}0_aggavg_5_{ds}.json"


WEBSITE_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "docs", "bop_website_scores")


def load_scores(root, method, ds):
    """Local bop_toolkit scores, else the scores returned by the BOP website (marked with 'web')."""
    name = f"{method}_{ds}-test{SPLIT_SUFFIX.get(ds, '')}"
    out = {}
    for t in ["segm", "bbox"]:
        local = osp.join(root, "results", "bop_eval", name, f"scores_bop22_coco_{t}.json")
        web = osp.join(WEBSITE_DIR, f"{name}_scores_bop22_coco_{t}.json")
        if osp.exists(local):
            out[t] = json.load(open(local)); out[t]["_source"] = "local"
        elif osp.exists(web):
            out[t] = json.load(open(web)); out[t].setdefault("_source", "BOP website")
    return out


def fmt(x):
    return "-" if x is None else f"{x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets/bop23_challenge")
    ap.add_argument("--markdown", default=None, help="also write the tables to this markdown file")
    args = ap.parse_args()

    pred_dir = osp.join(args.root, "results", "seg_exps")
    scores = {m: {ds: load_scores(args.root, m, ds) for ds in PAPER_ORDER} for m in OURS}
    methods = [m for m in OURS if any("segm" in s for s in scores[m].values())]

    def row(name, vals):
        cells = [fmt(vals.get(ds)) for ds in PAPER_ORDER]
        present = [vals[ds] for ds in PAPER_ORDER if vals.get(ds) is not None]
        mean = sum(present) / len(present) if present else None
        return f"| {name} | " + " | ".join(cells) + f" | {fmt(mean)} |"

    header = "| Method | " + " | ".join(PAPER_NAMES[d] for d in PAPER_ORDER) + " | Mean |\n|---|" + "---|" * (len(PAPER_ORDER) + 1)
    segm = {m: {ds: s["segm"]["AP"] * 100 for ds, s in scores[m].items() if "segm" in s} for m in methods}
    bbox = {m: {ds: s["bbox"]["AP"] * 100 for ds, s in scores[m].items() if "bbox" in s} for m in methods}
    web = {m: [ds for ds in PAPER_ORDER if scores[m][ds].get("segm", {}).get("_source", "local") != "local"] for m in methods}
    missing = {m: [ds for ds in PAPER_ORDER if ds not in segm[m]] for m in methods}

    lines = []
    lines.append("### 2D segmentation AP (BOP'19 test targets, 42 template views)")
    lines.append("")
    lines.append(header)
    for name, vals in PAPER.items():
        lines.append(row(name, vals))
    for m in methods:
        lines.append(row(OURS[m][0], segm[m]))
    lines.append("")
    lines.append(f"The first {len(PAPER)} rows are Table 1 of the CNOS paper (ICCV-W 2023).")
    lines.append("")
    lines.append("### 2D detection AP (amodal bbox), ours only")
    lines.append("")
    lines.append(header)
    for m in methods:
        lines.append(row(OURS[m][0], bbox[m]))
    bbox_missing = {m: [ds for ds in PAPER_ORDER if ds not in bbox[m]] for m in methods}
    if any(bbox_missing.values()):
        def bbox_note(miss):
            return (f"No bbox score for {', '.join(PAPER_NAMES[d] for d in miss)} (only the segmentation task was scored for them), "
                    f"so the Mean column is over the other {len(PAPER_ORDER) - len(miss)} datasets and is not comparable to the "
                    f"{len(PAPER_ORDER)}-dataset segmentation mean.")
        lines.append("")
        if all(bbox_missing[m] == bbox_missing[methods[0]] for m in methods):
            lines.append(bbox_note(bbox_missing[methods[0]]))
        else:
            for m in methods:
                if bbox_missing[m]:
                    lines.append(f"{OURS[m][0]}: {bbox_note(bbox_missing[m])}")
    for m in methods:
        name, rendering, submission = OURS[m]
        lines.append("")
        lines.append(f"### Per-dataset details: {name}")
        lines.append("")
        lines.append("| Dataset | images | detections | segm AP | AP50 | AP75 | AR100 | bbox AP | time / image | scored |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for ds in PAPER_ORDER:
            s = scores[m][ds]
            pred = osp.join(pred_dir, PRED_FILE.format(rendering=rendering, ds=ds))
            n_img = n_det = "-"
            if osp.exists(pred):
                d = json.load(open(pred))
                n_det = len(d); n_img = len({(x["scene_id"], x["image_id"]) for x in d})
            if "segm" in s:
                g, b = s["segm"], s.get("bbox", {})
                src = "local" if g.get("_source", "local") == "local" else "BOP website"
                bbox_ap = fmt(b["AP"] * 100) if "AP" in b else "-"
                lines.append(f"| {PAPER_NAMES[ds]} | {n_img} | {n_det} | {g['AP']*100:.1f} | {g['AP50']*100:.1f} | {g['AP75']*100:.1f} | {g['AR100']*100:.1f} | {bbox_ap} | {g['average_time_per_image']:.2f} s | {src} |")
            else:
                note = "no test GT in the BOP release, score on bop.felk.cvut.cz" if osp.exists(pred) else "not run"
                lines.append(f"| {PAPER_NAMES[ds]} | {n_img} | {n_det} | - | - | - | - | - | - | {note} |")
        if web[m]:
            lines.append("")
            sub_path = osp.normpath(osp.join(args.root, "results", submission))
            lines.append(f"{', '.join(PAPER_NAMES[d] for d in web[m])}: the BOP'19 test archives ship no scene_gt.json, so these scores come from the "
                         f"BOP website (method name `{m}`, same eval_bop22_coco.py); prediction files in `{sub_path}/`, "
                         f"raw score files in `docs/bop_website_scores/`. "
                         f"Their bbox AP is listed only if the 2D detection task was also scored there.")
        if missing[m]:
            lines.append("")
            lines.append(f"Not evaluated: {', '.join(PAPER_NAMES[d] for d in missing[m])} (the mean of the {name} row is over the evaluated datasets only).")
    text = "\n".join(lines)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(text + "\n")
        print(f"\nwritten to {args.markdown}")


if __name__ == "__main__":
    main()
