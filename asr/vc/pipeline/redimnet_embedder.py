"""ReDimNet2 (PalabraAI) speaker embedder. 192-dim, cosine-ready after L2-norm.

IMPORTANT: ReDimNet2.forward(x) takes ONLY the raw 16kHz waveform (1,T) and computes log-mel
internally. There is NO wav_lens/length mask, and its NormalizeAudio + ASTP pooling run over the
whole time axis -> zero-padding a batch corrupts embeddings. So we embed PER CLIP (no padding).
Default variant b6/lm/vox2 (best Vox1-O). For out-of-domain audio, dataset='vb2+vox2+cnc2_v0' (b3/b6)
generalizes better -- try as a separate embedder key if needed.
"""
import os
import numpy as np, torch, torch.nn.functional as F, torchaudio
from tqdm import tqdm

class RedimnetEmbedder:
    name = "redimnet"; dim = 192
    def __init__(self, device="cuda", model_name="b6", train_type="lm", dataset="vox2", cap_s=12):
        self.device = device; self.cap_s = cap_s
        repo = os.environ.get("REDIMNET_HUB_REPO", "PalabraAI/redimnet2")
        self.model = torch.hub.load(repo, "redimnet2",
                                    model_name=model_name, train_type=train_type, dataset=dataset,
                                    pretrained=True, trust_repo=True).eval().to(device)
        self.variant = f"{model_name}/{train_type}/{dataset}"

    def _load(self, p):
        w, sr = torchaudio.load(p)
        if w.shape[0] > 1: w = w.mean(0, keepdim=True)
        if sr != 16000: w = torchaudio.functional.resample(w, sr, 16000)
        w = w.squeeze(0)[: int(self.cap_s * 16000)]
        return w

    @torch.inference_mode()
    def embed(self, clips, desc=None):
        embs = np.zeros((len(clips), self.dim), dtype=np.float32)
        for i, c in enumerate(tqdm(clips, desc=f"redimnet {desc or ''}".strip(), unit="clip", leave=False)):
            w = self._load(c["wav_path"]).to(self.device).unsqueeze(0)  # (1,T)
            e = self.model(w)                                           # (1,192)
            embs[i] = F.normalize(e, dim=-1).squeeze(0).cpu().numpy()
        torch.cuda.empty_cache()
        return embs   # already unit-norm
