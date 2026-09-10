"""Shared helpers for the NADI voice-cloning augmentation pipeline.

Runtime layout beneath NADI_VC_WORKSPACE (see ../README.md):
  data/<dataset>/   dataset.yaml, embeddings, metadata
  exp/<exp_id>/     config.yaml, bank/, gen/
"""
import os, json, re
import numpy as np
import yaml

VC = os.path.abspath(os.environ.get("NADI_VC_WORKSPACE", "work/vc"))
WORKSPACE = VC

DATA = f"{VC}/data"
EXP = f"{VC}/exp"

DIALECTS = ["Algeria","Egypt","Jordan","Mauritania","Morocco","Palestine","UAE","Yemen"]

# ---------------- dataset / experiment config ----------------
def load_dataset(name):
    """Return dataset.yaml dict for data/<name>/. Keys: name, audio_root, layout, countries, splits, description."""
    with open(f"{DATA}/{name}/dataset.yaml") as f:
        d = yaml.safe_load(f)
    d.setdefault("countries", DIALECTS)
    d.setdefault("splits", ["train"])
    if d["splits"] != ["train"]:
        raise ValueError("The NADI ASR augmentation package accepts only the train split")
    d.setdefault("layout", "country_split")
    for k in ("audio_root", "metadata_root"):     # shared datasets/ resolve against the WORKSPACE root (move-safe)
        if d.get(k) and not os.path.isabs(d[k]):
            d[k] = os.path.join(WORKSPACE, d[k])
    return d

def load_config(exp_id):
    with open(f"{EXP}/{exp_id}/config.yaml") as f:
        return yaml.safe_load(f)

def iter_clips(ds, country):
    """Yield clip dicts for one dialect from a dataset. wav paths come from audio_root; the clip list
    (ids/transcripts/durations) comes from metadata_root (defaults to audio_root) -- so a denoised audio
    copy without its own metadata can reuse the raw dataset's metadata. layout 'country_split' =
    <root>/<Country>/<split>/{metadata.json, wavs/<id>.wav}."""
    root = ds["audio_root"]
    meta_root = ds.get("metadata_root", root)
    layout = ds.get("layout", "country_split")
    ext = ds.get("ext", "wav")
    clips = []
    for sp in ds["splits"]:
        mp = f"{meta_root}/{country}/{sp}/metadata.json"   # clip list always from metadata_root (country_split)
        if not os.path.exists(mp):
            continue
        for d in json.load(open(mp)):
            cid = d["id"]
            if layout == "country_flat":                   # <audio_root>/<Country>/<id>.<ext>  (e.g. clean-neural flac)
                wp = f"{root}/{country}/{cid}.{ext}"
            else:                                          # country_split: <audio_root>/<Country>/<split>/wavs/<id>.<ext>
                wp = f"{root}/{country}/{sp}/wavs/{cid}.{ext}"
            clips.append({"id": cid, "prefix": cid.split("_")[0], "country": country,
                          "split": sp, "dur": float(d["duration_s"]),
                          "transcription": d.get("transcription",""), "wav_path": wp})
    return clips

# ---------------- embeddings io (keyed by embedder; meta is embedder-independent) ----------------
def emb_paths(dataset, embedder, country):
    return (f"{DATA}/{dataset}/emb_{embedder}_{country}.npy",
            f"{DATA}/{dataset}/meta_{country}.json")

def load_emb(dataset, embedder, country):
    ep, mp = emb_paths(dataset, embedder, country)
    return np.load(ep), json.load(open(mp))

# ---------------- models ----------------
def load_knnvc(device="cuda", prematched=True):
    import torch
    # External dependency; see ../README.md for the upstream license condition.
    local_repo = os.environ.get("KNNVC_LOCAL_REPO")
    repo = local_repo or os.environ.get(
        "KNNVC_HUB_REPO", "bshall/knn-vc:c616845c4e309e24d5927f15adbdf277a3d65358")
    return torch.hub.load(repo, 'knn_vc', source="local" if local_repo else "github", prematched=prematched,
                          trust_repo=True, pretrained=True, device=device)

# ---------------- clustering ----------------
def cohesion(E):
    """Mean off-diagonal cosine of L2-normalized rows E (n,d)."""
    n = len(E)
    if n == 1: return 1.0
    S = E @ E.T
    return float((S.sum() - n) / (n * (n - 1)))

def cluster_voices(emb, meta, threshold=0.45, linkage="complete", coh_min=0.70, min_dur=30.0):
    """Agglomerative (cosine) clustering -> list of voice dicts passing cohesion+duration gates,
    sorted by duration desc. Each: clips, wav_paths, dur, n_video, cohesion, dialects."""
    from sklearn.cluster import AgglomerativeClustering
    from collections import defaultdict
    dur = np.array([m["dur"] for m in meta])
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage=linkage,
                                  distance_threshold=threshold).fit_predict(emb)
    cl = defaultdict(list)
    for i, l in enumerate(lab): cl[l].append(i)
    voices = []
    for l, idx in cl.items():
        d = float(dur[idx].sum()); ch = cohesion(emb[idx])
        if ch >= coh_min and d >= min_dur:
            voices.append({"clips": [meta[i]["id"] for i in idx],
                           "wav_paths": [meta[i]["wav_path"] for i in idx],
                           "dur": round(d, 1), "n_video": len(set(meta[i]["prefix"] for i in idx)),
                           "cohesion": round(ch, 3),
                           "dialects": sorted(set(meta[i]["country"] for i in idx))})
    voices.sort(key=lambda v: -v["dur"])
    return voices

# ---------------- arabic text ----------------
_TASH = re.compile(r'[ً-ْٰـ]')
def norm_ar(t):
    t = _TASH.sub('', t)
    for a,b in [('أ','ا'),('إ','ا'),('آ','ا'),('ى','ي'),('ة','ه'),('ؤ','و'),('ئ','ي')]:
        t = t.replace(a,b)
    t = re.sub(r'[^ء-ي\s]', ' ', t)
    return re.sub(r'\s+',' ',t).strip()

EMPHATIC = set("صضطظقعحخ")
def emph_count(t): return sum(1 for c in t if c in EMPHATIC)
