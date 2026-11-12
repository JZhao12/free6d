"""
Collect the bop_toolkit scores of the 6D pose stage (run_pose.py + eval_bop_pose.sh) of every
BOP'19-23 core dataset into one markdown table, plus the coarse-pose AR of FoundPose and Pos3R.

usage: python -m src.scripts.summarize_pose_results [--root ./datasets/bop23_challenge]
           [--method free6d] [--tag ""] [--website_dir <repo>/docs/bop_website_scores]

Scores come from <root>/results/bop_eval_pose/<method><tag>_<ds>-<split>/scores_bop19.json; when that
is missing (ITODD and HB ship no test ground truth) the file <website_dir>/<method><tag>_<ds>-<split>_scores_bop19.json
with the scores returned by the BOP website is used instead (cells marked with *). Run statistics
come from <root>/results/pose_exps/<method><tag>_<ds>-<split>_stats.json.
"""
import argparse
import json
import os.path as osp

from src.scripts.summarize_bop_results import PAPER_NAMES, PAPER_ORDER, WEBSITE_DIR

ORDER = PAPER_ORDER
NAMES = PAPER_NAMES
# Pos3R paper: coarse pose estimation, AR (x100) in the order of ORDER, and the mean as printed there
REFERENCE = {
    "FoundPose (coarse)": ([39.6, 33.8, 46.7, 23.9, 20.4, 50.8, 45.2], 37.2),
    "Pos3R (coarse)": ([32.3, 31.5, 47.3, 33.1, 25.1, 53.7, 53.9], 39.5),
}
# row name -> (key of scores_bop19.json, scale, decimals)
SCORE_ROWS = [
    ("AR_VSD", "bop19_average_recall_vsd", 100.0, 1),
    ("AR_MSSD", "bop19_average_recall_mssd", 100.0, 1),
    ("AR_MSPD", "bop19_average_recall_mspd", 100.0, 1),
    ("AR", "bop19_average_recall", 100.0, 1),
    ("time/img (s)", "bop19_average_time_per_image", 1.0, 2),
]


def split_dashed(ds):
    return "test-primesense" if ds in ("tless", "hb") else "test"


def load_json(path):
    if not osp.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_cell(vals, decimals):
    """Mean over the available values, with the count when fewer than all datasets have one."""
    present = [v for v in vals if v is not None]
    if not present:
        return "n/a"
    cell = f"{sum(present) / len(present):.{decimals}f}"
    return cell if len(present) == len(vals) else f"{cell} ({len(present)}/{len(vals)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./datasets/bop23_challenge")
    ap.add_argument("--method", default="free6d")
    ap.add_argument("--tag", default="")
    ap.add_argument("--website_dir", default=WEBSITE_DIR)
    args = ap.parse_args()

    scores, from_web, stats = {}, set(), {}
    for ds in ORDER:
        stem = f"{args.method}{args.tag}_{ds}-{split_dashed(ds)}"
        s = load_json(osp.join(args.root, "results", "bop_eval_pose", stem, "scores_bop19.json"))
        if s is None:
            s = load_json(osp.join(args.website_dir, f"{stem}_scores_bop19.json"))
            if s is not None:
                from_web.add(ds)
        scores[ds] = s
        stats[ds] = load_json(osp.join(args.root, "results", "pose_exps", f"{stem}_stats.json"))

    def table_row(name, vals, decimals, mean=True, star=False):
        """vals: one float / str / None per dataset in ORDER."""
        cells = []
        for ds, v in zip(ORDER, vals):
            if v is None:
                cells.append("n/a")
            elif isinstance(v, str):
                cells.append(v)
            else:
                cells.append(f"{v:.{decimals}f}" + ("*" if star and ds in from_web else ""))
        last = mean_cell(vals, decimals) if mean else "-"
        return f"| {name} | " + " | ".join(cells) + f" | {last} |"

    header = "| | " + " | ".join(NAMES[d] for d in ORDER) + " | mean |\n|---|" + "---|" * (len(ORDER) + 1)
    lines = [f"### 6D pose of unseen objects, {args.method}{args.tag} (BOP'19 test targets)", "", header]
    for name, key, scale, dec in SCORE_ROWS:
        vals = [scores[ds][key] * scale if scores[ds] and key in scores[ds] else None for ds in ORDER]
        lines.append(table_row(name, vals, dec, star=True))

    def stat(ds, getter):
        try:
            return getter(stats[ds])
        except (TypeError, KeyError):
            return None

    lines.append(table_row(
        "pose time/img (s)", [stat(ds, lambda s: s["mean_pose_time_per_image"]) for ds in ORDER], 2))
    est = [stat(ds, lambda s: (s["counts"]["estimates"], s["counts"]["instances"])) for ds in ORDER]
    lines.append(table_row(
        "estimates / instances", [None if e is None else f"{e[0]} / {e[1]}" for e in est], 0, mean=False))
    lines.append(table_row(
        "fallbacks", [stat(ds, lambda s: s["counts"]["fallbacks"]) for ds in ORDER], 0, mean=False))
    if from_web:
        lines.append("")
        lines.append(f"\\* scores returned by the BOP website for {', '.join(NAMES[d] for d in ORDER if d in from_web)} "
                     f"(copied from `{osp.relpath(args.website_dir)}/`), not computed locally.")

    ours = [
        scores[ds]["bop19_average_recall"] * 100.0 if scores[ds] and "bop19_average_recall" in scores[ds] else None
        for ds in ORDER
    ]
    lines += ["", "### AR comparison, coarse pose estimation (Pos3R paper)", "", header]
    for name, (vals, mean) in REFERENCE.items():
        lines.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {mean:.1f} |")
    lines.append(table_row(f"{args.method}{args.tag} (ours)", ours, 1, star=True))

    # the reference means are over all seven datasets; compare on the scored ones when fewer
    scored = [i for i, v in enumerate(ours) if v is not None]
    if 0 < len(scored) < len(ORDER):
        ref_means = [sum(vals[i] for i in scored) / len(scored) for vals, _ in REFERENCE.values()]
        our_mean = sum(ours[i] for i in scored) / len(scored)
        names = ", ".join(NAMES[ORDER[i]] for i in scored)
        lines += ["", f"Mean AR over the {len(scored)} datasets scored here ({names}): "
                      f"FoundPose {ref_means[0]:.1f}, Pos3R {ref_means[1]:.1f}, "
                      f"{args.method}{args.tag} {our_mean:.1f}."]

    print("\n".join(lines))


if __name__ == "__main__":
    main()
