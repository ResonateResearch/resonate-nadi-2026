#!/usr/bin/env python3
"""Release adapter: concatenate E11 base/augmentation manifests and add dialects.

This performs schema projection only. It does not change sampling, audio, or text.
The original assembly command was not present in the inspected source scripts.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from common import dialect_of


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expected-rows", type=int, help="optional historical count check")
    a = ap.parse_args()
    rows = []
    for path in a.inputs:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "original_val" in row["audio"] or "original_val" in row.get("src", ""):
                    raise ValueError("validation audio in training input")
                if not row["text"].startswith("language Arabic<asr_text>"):
                    raise ValueError("missing Arabic training prefix")
                dialect = dialect_of(row)
                if not dialect:
                    raise ValueError(f"unresolved dialect: {row['audio']}")
                rows.append({"audio": row["audio"], "text": row["text"], "dialect": dialect})
    if a.expected_rows is not None and len(rows) != a.expected_rows:
        raise ValueError(f"got {len(rows)} rows; expected {a.expected_rows}; inspect upstream counts")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[assemble] {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
