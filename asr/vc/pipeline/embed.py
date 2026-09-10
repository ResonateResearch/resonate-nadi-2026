#!/usr/bin/env python3
"""Embed a dataset's clips with a chosen speaker embedder ->
   data/<dataset>/emb_<embedder>_<dialect>.npy + meta_<dialect>.json (meta is embedder-independent).
Config-independent (depends only on dataset audio + embedder), so run once per (dataset, embedder).
Uses ONLY the splits listed in the dataset's dataset.yaml (raw = train only; validation excluded).

Usage:
  python asr/vc/pipeline/embed.py --dataset clean_neural --embed-model redimnet
"""
import os, sys, time, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from embedders import get_embedder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--embed-model", choices=["redimnet"], default="redimnet")
    ap.add_argument("--dialects", default="", help="comma list; default = all in dataset.yaml")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ds = C.load_dataset(a.dataset)
    dialects = a.dialects.split(",") if a.dialects else ds["countries"]
    os.makedirs(f"{C.DATA}/{a.dataset}", exist_ok=True)
    print(f"dataset={a.dataset} embedder={a.embed_model} splits={ds['splits']}")
    emb = None
    for c in dialects:
        ep, mp = C.emb_paths(a.dataset, a.embed_model, c)
        if os.path.exists(ep) and not a.force:
            print(f"{c}: cached"); continue
        clips = C.iter_clips(ds, c)
        if not clips:
            print(f"{c}: no clips (skipped)"); continue
        if emb is None:
            emb = get_embedder(a.embed_model)   # load model lazily, once
        t0 = time.time(); vecs = emb.embed(clips, desc=c)
        os.makedirs(os.path.dirname(ep), exist_ok=True)
        __import__("numpy").save(ep, vecs)
        json.dump(clips, open(mp, "w", encoding="utf-8"), ensure_ascii=False)
        print(f"{c}: {len(clips)} clips -> {os.path.basename(ep)} {vecs.shape} ({time.time()-t0:.0f}s)")
    print("EMBED DONE:", a.dataset, a.embed_model)

if __name__ == "__main__":
    main()
