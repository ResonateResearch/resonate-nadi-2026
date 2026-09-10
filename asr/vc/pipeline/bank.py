#!/usr/bin/env python3
"""Cluster a dataset's embeddings into a target voice bank for one experiment config.
Reads exp/<exp_id>/config.yaml (dataset, embed_model, cluster knobs). Writes
exp/<exp_id>/bank/bank_<dialect>.json + bank_summary.json and prints the per-dialect table.

Sweep mode (--sweep) prints total voice counts over a grid of (threshold, cohesion) to pick a config
per dataset; it writes nothing.

Usage:
  python asr/vc/pipeline/bank.py --exp clean_neural__redimnet__thr0.45_coh0.7
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

def load_all(dataset, embedder, countries):
    out = {}
    for c in countries:
        try:
            out[c] = C.load_emb(dataset, embedder, c)
        except FileNotFoundError:
            pass
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--min-dur", type=float, default=None, help="override min_dur_s for reporting")
    a = ap.parse_args()
    cfg = C.load_config(a.exp)
    dataset = cfg["dataset"]; embedder = cfg.get("embed_model", "redimnet")
    cl = cfg.get("cluster", {})
    thr = cl.get("threshold", 0.45); link = cl.get("linkage", "complete")
    coh = cl.get("cohesion", 0.70); min_dur = a.min_dur if a.min_dur is not None else cl.get("min_dur_s", 30)
    ds = C.load_dataset(dataset)
    data = load_all(dataset, embedder, ds["countries"])
    if not data:
        sys.exit(f"no embeddings for dataset={dataset} embedder={embedder}. Run embed.py first.")

    if a.sweep:
        print(f"SWEEP exp={a.exp} dataset={dataset} embedder={embedder} {link} | cells = total voices >=60s / >=30s")
        thrs = [0.35,0.40,0.45,0.50,0.55]; cohs = [0.65,0.70,0.75,0.80]
        print("thr\\coh " + "  ".join(f"{ch:>11.2f}" for ch in cohs))
        for t in thrs:
            row = []
            for ch in cohs:
                v60 = v30 = 0
                for c,(emb,meta) in data.items():
                    vs = C.cluster_voices(emb, meta, threshold=t, linkage=link, coh_min=ch, min_dur=30)
                    v30 += len(vs); v60 += sum(1 for v in vs if v["dur"]>=60)
                row.append(f"{v60:>4}/{v30:<6}")
            print(f"{t:>5}   " + "  ".join(row))
        return

    bankdir = f"{C.EXP}/{a.exp}/bank"; os.makedirs(bankdir, exist_ok=True)
    print(f"exp={a.exp} dataset={dataset} embedder={embedder} | {link} thr={thr} coh>={coh} min_dur={min_dur}s")
    print(f"{'dialect':<11} {'clips':>5} {'hrs':>5} | {'v>=60s':>6} {'min60':>6} | {'v>=Nmin':>7} {'minN':>6}")
    print("-"*62)
    summary = {}; T = {"v60":0,"m60":0,"vN":0,"mN":0,"clips":0,"h":0.0}
    for c in ds["countries"]:
        if c not in data:
            print(f"{c:<11} (not embedded)"); continue
        emb, meta = data[c]
        voices = C.cluster_voices(emb, meta, threshold=thr, linkage=link, coh_min=coh, min_dur=min_dur)
        json.dump(voices, open(f"{bankdir}/bank_{c}.json","w"), ensure_ascii=False, indent=1)
        v60 = [v for v in voices if v["dur"]>=60]; m60 = sum(v["dur"] for v in v60)
        mN = sum(v["dur"] for v in voices); hrs = sum(m["dur"] for m in meta)/3600
        print(f"{c:<11} {len(meta):>5} {hrs:>5.2f} | {len(v60):>6} {m60/60:>6.0f} | {len(voices):>7} {mN/60:>6.0f}")
        summary[c] = {"v60": len(v60), f"v_ge_{int(min_dur)}s": len(voices)}
        T["v60"]+=len(v60); T["m60"]+=m60; T["vN"]+=len(voices); T["mN"]+=mN; T["clips"]+=len(meta); T["h"]+=hrs
    print("-"*62)
    print(f"{'TOTAL':<11} {T['clips']:>5} {T['h']:>5.2f} | {T['v60']:>6} {T['m60']/60:>6.0f} | {T['vN']:>7} {T['mN']/60:>6.0f}")
    print(f"\nBANK >=60s : {T['v60']} voices, {T['m60']/60:.0f} min")
    print(f"BANK >={int(min_dur)}s : {T['vN']} voices, {T['mN']/60:.0f} min")
    json.dump({"exp":a.exp,"dataset":dataset,"embed_model":embedder,"cluster":cl,"totals":T,"per_dialect":summary},
              open(f"{bankdir}/bank_summary.json","w"), indent=1)
    print(f"\nwrote {bankdir}/bank_*.json")

if __name__ == "__main__":
    main()
