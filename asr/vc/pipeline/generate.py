#!/usr/bin/env python3
"""WITHIN-DIALECT voice augmentation with kNN-VC. For each dialect D: take the top-N richest >=min-dur
voices from D's own bank as TARGETS, convert every D train clip (SOURCE content) into each target voice.
Output = a new dataset labelled with the SOURCE transcript + dialect (only timbre changes; same-dialect so
no accent drift). Self-pairs (source clip inside a target's own cluster) are skipped. N<=0 = all voices.

RESUMABLE: an output wav that already exists is skipped, and each finished clip is logged immediately to
<D>/metadata.jsonl. On relaunch it picks up exactly where it stopped (per (source,target) pair). A source
whose every target is already done is skipped without re-extracting features. metadata.json (array) is
rebuilt from the jsonl at each dialect's end. tqdm shows live progress.

Usage:
  python asr/vc/pipeline/generate.py --exp clean_neural__redimnet__thr0.45_coh0.7 --n-targets 10
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torchaudio
from tqdm import tqdm
import common as C

def load_meta(dataset, dialect):
    return json.load(open(f"{C.DATA}/{dataset}/meta_{dialect}.json"))

def load_bank(exp, dialect):
    p = f"{C.EXP}/{exp}/bank/bank_{dialect}.json"
    return json.load(open(p)) if os.path.exists(p) else []

def load_done(jsonl):
    """ids already logged (crash-safe: skip malformed trailing line)."""
    done = {}
    if os.path.exists(jsonl):
        for line in open(jsonl, encoding="utf-8"):
            line = line.strip()
            if not line: continue
            try: r = json.loads(line); done[r["id"]] = r
            except Exception: pass
    return done

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment providing the voice bank (embedder+cluster)")
    ap.add_argument("--n-targets", type=int, required=True, help="target voices per dialect; <=0 = ALL >=min-dur voices")
    ap.add_argument("--min-target-dur", type=float, default=60.0)
    ap.add_argument("--pool", default="same", choices=["same"])
    ap.add_argument("--dialects", default="")
    ap.add_argument("--limit", type=int, default=0, help="cap #source clips per dialect (0=all; for testing)")
    ap.add_argument("--gen-tag", default="")
    a = ap.parse_args()
    cfg = C.load_config(a.exp)
    dataset = cfg["dataset"]; topk = cfg.get("knnvc", {}).get("topk", 4)
    ds = C.load_dataset(dataset)
    dialects = a.dialects.split(",") if a.dialects else ds["countries"]
    tag = a.gen_tag or ("same_nall" if a.n_targets <= 0 else f"same_n{a.n_targets}")
    genroot = f"{C.EXP}/{a.exp}/gen/{tag}"
    os.makedirs(genroot, exist_ok=True)
    json.dump({"exp": a.exp, "dataset": dataset, "pool": a.pool, "n_targets": a.n_targets,
               "min_target_dur": a.min_target_dur, "topk": topk},
              open(f"{genroot}/gen_config.json", "w"), indent=1)

    knn = C.load_knnvc(prematched=cfg.get("knnvc", {}).get("prematched", True))
    grand = 0
    for D in dialects:
        bank = [v for v in load_bank(a.exp, D) if v["dur"] >= a.min_target_dur]
        bank.sort(key=lambda v: -v["dur"])
        targets = bank if a.n_targets <= 0 else bank[: a.n_targets]
        if not targets:
            print(f"{D}: no >= {a.min_target_dur}s voices -> skip"); continue
        outdir = f"{genroot}/{D}/wavs"; os.makedirs(outdir, exist_ok=True)
        jsonl = f"{genroot}/{D}/metadata.jsonl"
        done = load_done(jsonl)
        sources = load_meta(dataset, D)
        if a.limit: sources = sources[: a.limit]

        # build matching set per target once (vad off; skip empty feature clips)
        tms = []
        for k, t in enumerate(targets):
            feats = [knn.get_features(p, vad_trigger_level=0) for p in t["wav_paths"]]
            ms = torch.concat([f.cpu() for f in feats if f.shape[0] > 0], dim=0)
            tms.append({"k": k, "ms": ms, "clips": set(t["clips"]), "label": t["clips"][0]})

        def row_for(oid, s, t, dur):
            return {"id": oid, "wav_path": f"{outdir}/{oid}.wav", "transcription": s["transcription"],
                    "country": D, "split": "train", "duration_s": round(dur, 3), "sample_rate": 16000,
                    "source_id": s["id"], "target_voice": t["label"]}

        jf = open(jsonl, "a", encoding="utf-8")
        made = skipped = 0; t0 = time.time()
        for s in tqdm(sources, desc=f"{D} ({len(targets)}v)", unit="src"):
            plan = [(t, f"{s['id']}__v{t['k']}") for t in tms if s["id"] not in t["clips"]]
            # GROUND TRUTH = wav existence. Only (re)generate pairs whose wav is missing.
            need = [(t, oid) for t, oid in plan if not os.path.exists(f"{outdir}/{oid}.wav")]
            skipped += len(plan) - len(need)
            if not need:
                continue
            q = knn.get_features(s["wav_path"])                 # extract source features once
            for t, oid in need:
                out = knn.match(q, t["ms"], topk=topk)
                torchaudio.save(f"{outdir}/{oid}.wav", out[None].cpu(), 16000)
                r = row_for(oid, s, t, out.shape[-1]/16000)
                jf.write(json.dumps(r, ensure_ascii=False) + "\n"); jf.flush(); done[oid] = r
                made += 1
        jf.close()
        # rebuild metadata.json from the wavs that actually exist (jsonl = duration cache; else read wav)
        rows = []
        for s in sources:
            for t in tms:
                if s["id"] in t["clips"]: continue
                oid = f"{s['id']}__v{t['k']}"; op = f"{outdir}/{oid}.wav"
                if not os.path.exists(op): continue
                if oid in done: rows.append(done[oid])
                else:
                    info = torchaudio.info(op)
                    rows.append(row_for(oid, s, t, info.num_frames / info.sample_rate))
        json.dump(rows, open(f"{genroot}/{D}/metadata.json", "w", encoding="utf-8"), ensure_ascii=False)
        grand += len(rows)
        print(f"{D}: {len(sources)} src x {len(targets)} voices -> {len(rows)} clips "
              f"(made {made}, resumed/skipped {skipped}) {time.time()-t0:.0f}s")
    print(f"GEN DONE: {grand} clips -> {genroot}")

if __name__ == "__main__":
    main()
