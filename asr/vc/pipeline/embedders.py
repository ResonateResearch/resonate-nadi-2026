"""Dispatcher extracted for the ReDimNet2 ASR voice-bank configuration.

Historical ECAPA/ECAPA2 alternatives are outside this release's retained recipe.
"""


def get_embedder(name, device="cuda"):
    if name == "redimnet":
        from redimnet_embedder import RedimnetEmbedder
        return RedimnetEmbedder(device)
    raise ValueError(f"unknown embed_model: {name}; this package exports redimnet")
