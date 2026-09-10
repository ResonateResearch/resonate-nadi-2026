"""Context-Aware Multi-Head Factorized Attentive Pooling (CA-MHFA).

A lightweight speaker-extractor back-end for SSL-based speaker verification.
It consumes the layer-wise (frame-level) representations of a pre-trained SSL
model and produces an utterance-level speaker embedding.

CA-MHFA has three parts (Fig. 1 of the paper):
    1. Frame-Level Extractor with compression: two learnable sets of softmax
       layer-weights (factors) aggregate the SSL layers into a *key* and a
       *value* stream, each compressed by a linear layer  F -> D.
    2. Context-Aware Attentive Pooling: G groups of L learnable global queries
       attend over a context window of L frames of the key stream. This is
       implemented as a 1-D convolution (kernel size L, D in-channels, G
       out-channels) whose kernels *are* the queries, followed by a softmax
       over time. Setting L = 1 recovers MHFA; G = 1 recovers self-attentive
       pooling.
    3. Utterance-Level Extractor: per-group attentive pooling of the value
       stream, concatenation over groups, and a final linear projection to the
       speaker embedding.

Reuses SpeechBrain blocks: ``nnet.linear.Linear``, ``nnet.CNN.Conv1d`` and
``dataio.dataio.length_to_mask``.

Reference: J. Peng et al., "CA-MHFA: A Context-Aware Multi-Head Factorized
Attentive Pooling for SSL-Based Speaker Verification", 2024.
https://arxiv.org/abs/2409.15234

Authors
 * Generated for the nadi project, 2026

Public-release modification (2026): relocated from the SpeechBrain model
package to this local module; the pooling implementation is unchanged.
"""

import torch
import torch.nn.functional as F

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.linear import Linear


class CA_MHFA(torch.nn.Module):
    """Context-Aware Multi-Head Factorized Attentive Pooling back-end.

    Arguments
    ---------
    input_size : int
        Feature dimension F of a single SSL layer output.
    num_layers : int
        Number of SSL layer outputs stacked in the input (N + 1, i.e. the CNN
        encoder output plus every transformer block).
    compressed_dim : int
        Dimension D the key/value streams are compressed to. The default of
        128 keeps the back-end at ~2.3M params (matching the paper) since the
        final projection is the dominant term (G * D * emb_dim).
    n_heads : int
        Number of groups (heads) G.
    context : int
        Context length L (number of neighbouring frames each query attends to).
        Should be odd so the window is symmetric (R = (L - 1) / 2 on each side).
        L = 1 reduces CA-MHFA to MHFA.
    emb_dim : int
        Dimension of the output speaker embedding.
    layernorm : bool
        If True, layer-normalise every SSL layer before the weighted sum.

    Example
    -------
    >>> feats = torch.rand([4, 13, 50, 768])  # [B, num_layers, T, F]
    >>> model = CA_MHFA(input_size=768, num_layers=13)
    >>> emb = model(feats)
    >>> emb.shape
    torch.Size([4, 256])
    """

    def __init__(
        self,
        input_size,
        num_layers,
        compressed_dim=128,
        n_heads=64,
        context=9,
        emb_dim=256,
        layernorm=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.compressed_dim = compressed_dim
        self.n_heads = n_heads
        self.context = context
        self.layernorm = layernorm

        # --- Frame-Level Extractor: two sets of layer factors (omega^k, omega^v)
        # Softmax-normalised over layers, like WeightedSSLModel but one set per
        # stream. Zero init => uniform weights at start.
        self.weights_k = torch.nn.Parameter(torch.zeros(num_layers))
        self.weights_v = torch.nn.Parameter(torch.zeros(num_layers))

        # Compression matrices S^k, S^v : F -> D (pure linear, no bias).
        self.linear_k = Linear(
            input_size=input_size, n_neurons=compressed_dim, bias=False
        )
        self.linear_v = Linear(
            input_size=input_size, n_neurons=compressed_dim, bias=False
        )

        # --- Context-Aware Attentive Pooling ---
        # The G groups of L global queries are the kernels of this convolution:
        # in-channels D, out-channels G, kernel size L. weight shape [G, D, L].
        # "same" padding keeps the time axis length; zero (constant) padding
        # matches attending over out-of-utterance frames as empty context.
        self.query_conv = Conv1d(
            in_channels=compressed_dim,
            out_channels=n_heads,
            kernel_size=context,
            padding="same",
            padding_mode="constant",
            bias=False,
        )

        # --- Utterance-Level Extractor: concat(G * D) -> embedding.
        self.linear_out = Linear(
            input_size=n_heads * compressed_dim, n_neurons=emb_dim
        )

    def forward(self, feats, lengths=None):
        """Compute the utterance-level speaker embedding.

        Arguments
        ---------
        feats : torch.Tensor
            Stacked SSL layer outputs, shape ``[B, num_layers, T, F]``
            (e.g. ``torch.stack(hidden_states, dim=1)``).
        lengths : torch.Tensor, optional
            Relative lengths of each utterance in the batch (values in
            ``(0, 1]``); used to mask padded frames in the attention softmax.

        Returns
        -------
        emb : torch.Tensor
            Speaker embedding of shape ``[B, emb_dim]``.
        """
        if feats.dim() != 4:
            raise ValueError(
                "CA_MHFA expects [B, num_layers, T, F], got "
                f"{tuple(feats.shape)}"
            )
        if feats.shape[1] != self.num_layers:
            raise ValueError(
                f"Expected {self.num_layers} layers, got {feats.shape[1]}"
            )

        if self.layernorm:
            feats = F.layer_norm(feats, (feats.shape[-1],))

        # Factorized frame-level extractor: two softmax-weighted layer sums.
        w_k = F.softmax(self.weights_k, dim=0)
        w_v = F.softmax(self.weights_v, dim=0)
        k_in = torch.einsum("l,bltf->btf", w_k, feats)  # [B, T, F]
        v_in = torch.einsum("l,bltf->btf", w_v, feats)  # [B, T, F]

        k = self.linear_k(k_in)  # [B, T, D]
        v = self.linear_v(v_in)  # [B, T, D]

        # Zero out padded frames in both streams so the conv sees them as empty
        # context (identical to the zero-padding it already applies beyond the
        # utterance) -- otherwise a padded frame leaks into the logit of a
        # valid neighbour inside the L-frame window.
        T = k.shape[1]
        mask = None
        if lengths is not None:
            mask = length_to_mask(
                lengths * T, max_len=T, device=feats.device
            )  # [B, T]
            frame_mask = mask.unsqueeze(-1)  # [B, T, 1]
            k = k * frame_mask
            v = v * frame_mask

        # Context-aware attention logits: conv (queries) over the key stream,
        # averaged over the L-frame window (1 / L factor of Eq. 2).
        logits = self.query_conv(k) / self.context  # [B, T, G]
        logits = logits.transpose(1, 2)  # [B, G, T]

        # Mask padded frames before the softmax over time.
        if mask is not None:
            logits = logits.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = F.softmax(logits, dim=-1)  # [B, G, T], softmax over time

        # Utterance-level extractor: per-group attentive pooling of the values.
        c = torch.bmm(attn, v)  # [B, G, D]
        c = c.flatten(start_dim=1)  # [B, G * D]

        emb = self.linear_out(c)  # [B, emb_dim]
        return emb
