#!/usr/bin/env python3
"""
Regenerate the `training_analytics_<model>.xlsx` files consumed by
notebooks/results_analysis/traning_result_analysis.ipynb, by parsing the mmseg
training logs already downloaded under descargas_azure/<model>/seed_<S>/.

Each output row = one validation point of a training run, with:
  seed, iter, loss, loss_cls, loss_mask, loss_dice, lr, grad_norm,
  aAcc, mIoU, mAcc, mDice, mFscore, mPrecision, mRecall,
  train_time, train_data_time,
  {Acc,Dice,Fscore,IoU,Precision,Recall}.<class> for the 9 classes.

Losses/lr/grad_norm/time come from the nearest preceding `Iter(train)` line;
the per-class table + `Iter(val)` summary provide the validation metrics.

Usage:
    python scripts/logs/parse_training_logs.py \
        --azure descargas_azure --out notebooks/results_analysis
"""
import argparse
import glob
import os
import re

import pandas as pd

CLASSES = ["bg", "cracks", "cracks_alligator", "cracks_severe", "edge_cracks",
           "fretting", "pothole", "manhole", "pole_shadow"]

# descargas_azure dir  ->  notebook file suffix (model_files keys)
MODEL_MAP = {
    "swin": "swin",
    "beit": "beit",
    "interimage": "internImage",
    "flash": "flash",
    "hrnet": "hrnet",
}

# --- content-based identification (the download mislabeled some folders) ---
BACKBONE_TO_MODEL = [
    ("FlashInternImage", "flash"),   # check before InternImage (substring)
    ("SwinTransformer", "swin"),
    ("InternImage", "interimage"),
    ("HRNet", "hrnet"),
    ("BEiT", "beit"),
]


def identify_log(path, head_lines=4000):
    """Return (model, seed) that a training log ACTUALLY belongs to, read from
    its own config dump — robust to folders that hold the wrong run."""
    model = seed = None
    with open(path, "r", errors="ignore") as fh:
        for n, line in enumerate(fh):
            if model is None and "type=" in line:
                for token, m in BACKBONE_TO_MODEL:
                    if f"type='{token}'" in line:
                        model = m
                        break
            if seed is None:
                ms = re.search(r"seed=(\d+)", line)
                if ms:
                    seed = int(ms.group(1))
            if model and seed:
                break
            if n > head_lines:
                break
    return model, (f"seed_{seed}" if seed is not None else None)


RE_TRAIN = re.compile(r"Iter\(train\)\s*\[\s*(\d+)/")
RE_VAL = re.compile(
    r"Iter\(val\).*?aAcc:\s*([\d.]+)\s+mIoU:\s*([\d.]+)\s+mAcc:\s*([\d.]+)"
    r"\s+mDice:\s*([\d.]+)\s+mFscore:\s*([\d.]+)\s+mPrecision:\s*([\d.]+)\s+mRecall:\s*([\d.]+)"
)


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def parse_train_line(line):
    """Pull loss / lr / grad_norm / time from an Iter(train) log line."""
    def grab(key):
        m = re.search(rf"(?:^|\s){re.escape(key)}:\s*([-\d.eE+]+)", line)
        return _f(m.group(1)) if m else float("nan")
    return {
        "loss": grab("loss"),
        "loss_cls": grab("decode.loss_cls"),
        "loss_mask": grab("decode.loss_mask"),
        "loss_dice": grab("decode.loss_dice"),
        "lr": grab("lr"),
        "grad_norm": grab("grad_norm"),
        "train_time": grab("time"),
        "train_data_time": grab("data_time"),
    }


def parse_class_table(lines, start):
    """Parse the per-class metric table beginning near index `start`.
    Returns dict {Metric.class: value} and the index after the table."""
    out = {}
    i = start
    while i < len(lines):
        row = re.findall(r"\|\s*([^|]+?)\s*(?=\|)", lines[i])
        # a class row looks like: | cracks | 26.9 | 45.4 | 53.9 | ... |
        if len(row) >= 7 and row[0] in CLASSES:
            cls = row[0]
            iou, acc, dice, fscore, prec, rec = row[1:7]
            out[f"IoU.{cls}"] = _f(iou)
            out[f"Acc.{cls}"] = _f(acc)
            out[f"Dice.{cls}"] = _f(dice)
            out[f"Fscore.{cls}"] = _f(fscore)
            out[f"Precision.{cls}"] = _f(prec)
            out[f"Recall.{cls}"] = _f(rec)
        # stop once we passed the table and hit the val summary
        if "Iter(val)" in lines[i]:
            break
        i += 1
    return out


def parse_log(path):
    with open(path, "r", errors="ignore") as fh:
        lines = fh.readlines()

    rows = []
    cur_iter = 0
    last_train = {}
    pending_table = {}

    for i, line in enumerate(lines):
        mt = RE_TRAIN.search(line)
        if mt:
            cur_iter = int(mt.group(1))
            last_train = parse_train_line(line)
            continue
        # per-class table header -> collect the table that follows
        if "Class" in line and "IoU" in line and "|" in line:
            pending_table = parse_class_table(lines, i)
            continue
        mv = RE_VAL.search(line)
        if mv:
            aAcc, mIoU, mAcc, mDice, mFscore, mPrec, mRec = map(_f, mv.groups())
            row = {"iter": cur_iter,
                   "aAcc": aAcc, "mIoU": mIoU, "mAcc": mAcc,
                   "mDice": mDice, "mFscore": mFscore,
                   "mPrecision": mPrec, "mRecall": mRec}
            row.update(last_train)
            row.update(pending_table)
            rows.append(row)
            pending_table = {}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--azure", default="descargas_azure")
    ap.add_argument("--out", default="notebooks/results_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1) scan EVERY log, identify (model, seed) BY CONTENT (folders were mislabeled).
    #    Heavy runs crashed and were RESTARTED FROM SCRATCH several times, so a
    #    (model, seed) has several independent attempts (NOT resumes -> never merge).
    #    The COMPLETE run is the one that finished properly: it either early-stopped
    #    ("...did not improve in the last N records. best score: X") or reached
    #    max_iters. Among completed attempts keep the best-mIoU one (matches the
    #    thesis PDF). Crashed restarts (no early-stop line, iter < max) are discarded.
    RE_TOTAL = re.compile(r"Iter\(train\)\s*\[\s*\d+/(\d+)\]")
    RE_EARLYSTOP = re.compile(r"did not improve in the last \d+ records\. best score:\s*(\d+\.\d+)")

    attempts = {}  # (model, seed) -> list of attempt dicts
    all_logs = glob.glob(os.path.join(args.azure, "**", "*.log"), recursive=True)
    for log in all_logs:
        model, seed = identify_log(log)
        if not model or not seed:
            continue  # system log / not a mmseg training log
        rows = parse_log(log)
        if not rows:
            continue
        total = 0
        best_score = None
        with open(log, "r", errors="ignore") as fh:
            for line in fh:
                mt = RE_TOTAL.search(line)
                if mt:
                    total = int(mt.group(1))
                me = RE_EARLYSTOP.search(line)
                if me:
                    best_score = float(me.group(1))
        max_iter = max((r["iter"] for r in rows), default=0)
        peak = max((r["mIoU"] for r in rows), default=0.0)
        completed = (best_score is not None) or (total and max_iter >= total)
        attempts.setdefault((model, seed), []).append(
            dict(peak=peak, max_iter=max_iter, total=total, completed=completed,
                 best_score=best_score, log=log, rows=rows))

    best = {}  # (model, seed) -> (max_iter, path, rows)
    for key, atts in attempts.items():
        comp = [a for a in atts if a["completed"]]
        pool = comp if comp else atts
        a = max(pool, key=lambda x: x["peak"])
        if not comp:
            print(f"  [!] {key}: NO completed attempt found -> using best-peak crashed run")
        best[key] = (a["max_iter"], a["log"], a["rows"])

    # 2) report coverage across the 25 expected model x seed runs
    SEEDS = ["seed_42", "seed_91", "seed_777", "seed_1337", "seed_2026"]
    print("Coverage (identified by log content, folder-agnostic):")
    for adir in MODEL_MAP:  # adir == the model key produced by identify_log
        found = [s for s in SEEDS if (adir, s) in best]
        missing = [s for s in SEEDS if (adir, s) not in best]
        print(f"  {adir:11s}: {len(found)}/5 ok"
              + (f"  MISSING: {missing}" if missing else ""))

    # 3) write one xlsx per model
    for adir, suffix in MODEL_MAP.items():
        all_rows = []
        for seed in SEEDS:
            if (adir, seed) in best:
                max_iter, log, rows = best[(adir, seed)]
                for r in rows:
                    r["seed"] = seed
                all_rows.extend(rows)
                print(f"  {adir}/{seed}: {len(rows)} val rows (to iter {max_iter}) "
                      f"<- {log.split('descargas_azure/')[-1]}")
        if not all_rows:
            print(f"[!] {adir}: no rows, skipping")
            continue
        df = pd.DataFrame(all_rows)
        front = ["seed", "iter", "loss", "loss_cls", "loss_mask", "loss_dice",
                 "lr", "grad_norm", "aAcc", "mIoU", "mAcc", "train_time", "train_data_time"]
        cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
        df = df[cols]
        out = os.path.join(args.out, f"training_analytics_{suffix}.xlsx")
        df.to_excel(out, index=False, engine="openpyxl")
        print(f"-> {out}  ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
