#!/usr/bin/env python3
"""
Build the `testing_analytics_<model>.xlsx` files consumed by
notebooks/results_analysis/test_result_analysis_flowity.ipynb, from the Flowity
TEST runs produced by run_flowity_test.sh (logs under
experiments/<exp>/outputx5/seed_<S>/test/<ts>/<ts>.log).

One row per (model, seed): aAcc, mIoU, mAcc, mDice, mFscore, mPrecision, mRecall,
iter (checkpoint iter) and per-class {IoU,Acc,Dice,Fscore,Precision,Recall}.

NOTE: metrics use config_test.py's test pipeline; they may differ slightly from
the original thesis numbers if the original used a different test-time scale.

Usage:  python scripts/parse_test_logs.py --out notebooks/results_analysis
"""
import argparse, glob, os, re
import pandas as pd

CLASSES = ["bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks",
           "fretting", "pothole", "manhole", "pole_shadow"]

# exp dir -> (descargas model key, notebook xlsx suffix)
MODELS = {
    "swin-T-512x512":             ("swin",       "swin"),
    "BeiT2-T":                    ("beit",       "beit"),
    "HRNet-T-512x512":            ("hrnet",      "hrnet"),
    "flashInternImage-T-512x512": ("flash",      "flash"),
    "InterImage-T-512x512":       ("interimage", "internImage"),
}
SEEDS = ["42", "91", "777", "1337", "2026"]

RE_TESTN = re.compile(r"Iter\(test\)\s*\[\s*(\d+)/")
RE_TEST = re.compile(
    r"Iter\(test\).*?aAcc:\s*([\d.]+)\s+mIoU:\s*([\d.]+)\s+mAcc:\s*([\d.]+)"
    r"\s+mDice:\s*([\d.]+)\s+mFscore:\s*([\d.]+)\s+mPrecision:\s*([\d.]+)\s+mRecall:\s*([\d.]+)")
RE_CKPT = re.compile(r"best_mIoU_iter_(\d+)\.pth")


def _f(s):
    try: return float(s)
    except (TypeError, ValueError): return float("nan")


def parse_class_table(lines):
    out = {}
    for ln in lines:
        row = re.findall(r"\|\s*([^|]+?)\s*(?=\|)", ln)
        if len(row) >= 7 and row[0] in CLASSES:
            c = row[0]
            iou, acc, dice, fs, pr, rc = row[1:7]
            out[f"IoU.{c}"] = _f(iou); out[f"Acc.{c}"] = _f(acc)
            out[f"Dice.{c}"] = _f(dice); out[f"Fscore.{c}"] = _f(fs)
            out[f"Precision.{c}"] = _f(pr); out[f"Recall.{c}"] = _f(rc)
    return out


def parse_test_log(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()
    summary = None
    for ln in lines:
        m = RE_TEST.search(ln)
        if m:
            aAcc, mIoU, mAcc, mDice, mFscore, mPrec, mRec = map(_f, m.groups())
            summary = {"aAcc": aAcc, "mIoU": mIoU, "mAcc": mAcc, "mDice": mDice,
                       "mFscore": mFscore, "mPrecision": mPrec, "mRecall": mRec}
    if summary is None:
        return None
    row = summary
    row.update(parse_class_table(lines))
    it = -1
    for ln in lines:
        mc = RE_CKPT.search(ln)
        if mc:
            it = int(mc.group(1)); break
    row["iter"] = it
    # how many images this log evaluated (to prefer the full run over a smoke)
    ns = [int(x) for x in RE_TESTN.findall("\n".join(lines))]
    row["_nimgs"] = max(ns) if ns else 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="experiments")
    ap.add_argument("--out", default="notebooks/results_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for exp, (mkey, suffix) in MODELS.items():
        rows = []
        for seed in SEEDS:
            logs = glob.glob(os.path.join(args.exp, exp, "outputx5", f"seed_{seed}", "test", "**", "*.log"),
                             recursive=True)
            best = None
            for log in logs:
                r = parse_test_log(log)
                if r and (best is None or r["_nimgs"] > best["_nimgs"]):
                    best = r
            if best is None:
                print(f"  [!] {mkey}/seed_{seed}: no test result found")
                continue
            best.pop("_nimgs", None)
            best["seed"] = f"seed_{seed}"
            rows.append(best)
        if not rows:
            print(f"[!] {mkey}: no rows"); continue
        df = pd.DataFrame(rows)
        front = ["seed", "iter", "aAcc", "mIoU", "mAcc", "mDice", "mFscore", "mPrecision", "mRecall"]
        cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
        df = df[cols]
        outp = os.path.join(args.out, f"testing_analytics_{suffix}.xlsx")
        df.to_excel(outp, index=False, engine="openpyxl")
        print(f"-> {outp}  ({df.shape[0]} rows x {df.shape[1]} cols) | mIoU: "
              + ", ".join(f"{r['seed'].replace('seed_','')}={r['mIoU']:.1f}" for r in rows))


if __name__ == "__main__":
    main()
