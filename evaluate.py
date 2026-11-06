"""Compare the base model and a LoRA adapter on a held-out set.

Reports ROUGE-1/2/L against reference notes and a groundedness score: the
share of generated points supported by the dialogue (same checker as the app).

    python evaluate.py --test-csv data/test.csv --limit 50
    python evaluate.py --test-csv data/test.csv --limit 50 --adapter adapters/qwen-soap
"""
import argparse
import json
import statistics

import pandas as pd
from rouge_score import rouge_scorer

from clinical import grounding, llm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-csv", required=True)
    p.add_argument("--dialogue-col", default="dialogue")
    p.add_argument("--note-col", default="note")
    p.add_argument("--model", default=llm.DEFAULT_MODEL)
    p.add_argument("--adapter", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default="predictions.jsonl")
    args = p.parse_args()

    df = pd.read_csv(args.test_csv)
    if args.limit:
        df = df.head(args.limit)
    tok, model = llm.load_llm(args.model, args.adapter)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rows = []
    with open(args.out, "w") as f:
        for i, (dialogue, ref) in enumerate(zip(df[args.dialogue_col].astype(str), df[args.note_col].astype(str)), 1):
            lines = [l.strip() for l in dialogue.splitlines() if l.strip()]
            pred, _ = llm.summarize(tok, model, lines)
            checks = grounding.Grounder(lines).check_note(llm.parse_note(pred))
            r = scorer.score(ref, pred)
            row = {k: r[k].fmeasure for k in r}
            row["grounded"] = grounding.support_rate(checks)
            rows.append(row)
            f.write(json.dumps({"prediction": pred, "reference": ref, **row}) + "\n")
            print(f"[{i}/{len(df)}] rougeL={row['rougeL']:.3f} grounded={row['grounded']}")

    def mean(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return statistics.mean(vals) if vals else float("nan")

    print(f"\nModel: {args.model}" + (f" + {args.adapter}" if args.adapter else ""))
    for k in ["rouge1", "rouge2", "rougeL", "grounded"]:
        print(f"{k:>9}: {mean(k):.3f}")


if __name__ == "__main__":
    main()
