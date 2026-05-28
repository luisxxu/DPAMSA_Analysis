import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

"""
    Referenced from Github: https://github.com/huggingface/transformers

    Improvement 7: Replaced the custom single-head SelfAttention with
    nn.MultiheadAttention (n_heads=4). Multi-head attention allows the model
    to simultaneously attend to alignment patterns at different positions and
    representation subspaces, improving feature extraction quality without
    increasing inference time significantly. With d_model=64 and n_heads=4,
    each head operates on a 16-dimensional subspace.

    Embedding improvement A: IUPAC ambiguous nucleotide token IDs
    Vocabulary is expanded from 6 to 11 tokens. Ambiguous code embeddings
    are initialised as the mean of their constituent canonical base embeddings
    so the model starts from a biologically meaningful point rather than noise.

    Embedding improvement B: Embedding scale (sqrt(d_model))
    Following the original Transformer paper (Vaswani et al., 2017), token
    embeddings are multiplied by sqrt(d_model) before positional encoding is
    added. Without this, the positional signal — whose magnitude is fixed at
    ~1 — dominates the token signal because randomly-initialised embedding
    weights are typically small, causing the model to underweight token
    identity relative to position.
"""


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)  # (1, n_position, d_hid)

    def forward(self, x):
        y = self.pos_table[:, :x.size(1)].clone().detach()
        return x + y


class Encoder(nn.Module):
    """Transformer encoder using multi-head self-attention.

    Improvement 7: The original single-head SelfAttention is replaced with
    nn.MultiheadAttention. This uses PyTorch's optimised (fused) attention
    kernel and supports 4 attention heads, enabling the encoder to capture
    richer alignment features in parallel.

    Embedding improvement A: vocabulary expanded from 6 → 11 to give IUPAC
    ambiguous codes (N, R, W, K, Y) their own IDs. Their embeddings are
    initialised as the mean of constituent base embeddings.

    Embedding improvement B: token embeddings are scaled by sqrt(d_model)
    before positional encoding is added, following Vaswani et al. (2017).

    d_k and d_v are retained in the signature for backward compatibility
    but are no longer used; head_dim is derived as d_model // n_heads.
    """

    def __init__(
            self, n_src_vocab, d_model, n_position, n_heads=4,
            d_k=164, d_v=164, pad_idx=0, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        # Embedding improvement B: precompute the scale factor
        self.scale = math.sqrt(d_model)

        self.src_word_emb = nn.Embedding(n_src_vocab, d_model, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_model, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        # Improvement 7: multi-head attention — with d_model=64 and n_heads=4, head_dim=16
        self.attn = nn.MultiheadAttention(
            d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Embedding improvement A: seed ambiguous-code rows with constituent means
        self._init_iupac_embeddings()

    def _init_iupac_embeddings(self):
        """Set ambiguous IUPAC embedding rows to the mean of their constituent bases.

        Token IDs: pad=0, A=1, T=2, C=3, G=4, gap=5, N=6, R=7, W=8, K=9, Y=10

        IUPAC definitions used:
            N (6) — any nucleotide: A, T, C, G
            R (7) — purine:         A, G
            W (8) — weak bond:      A, T
            K (9) — keto:           G, T
            Y (10) — pyrimidine:    C, T
        """
        with torch.no_grad():
            w = self.src_word_emb.weight  # (vocab_size, d_model)
            w[6] = (w[1] + w[2] + w[3] + w[4]) / 4  # N = mean(A, T, C, G)
            w[7] = (w[1] + w[4]) / 2                  # R = mean(A, G)
            w[8] = (w[1] + w[2]) / 2                  # W = mean(A, T)
            w[9] = (w[4] + w[2]) / 2                  # K = mean(G, T)
            w[10] = (w[3] + w[2]) / 2                 # Y = mean(C, T)

    def forward(self, src_seq, mask=None):
        # Embedding improvement B: scale by sqrt(d_model) so token signal is
        # not drowned out by the fixed-magnitude positional encoding
        enc_output = self.src_word_emb(src_seq) * self.scale
        enc_output = self.position_enc(enc_output)
        enc_output = self.dropout(enc_output)
        enc_output = self.layer_norm(enc_output)

        # Build padding mask: True for pad positions (to be ignored by attention)
        key_padding_mask = (src_seq == self.pad_idx)
        enc_output, _ = self.attn(
            enc_output, enc_output, enc_output,
            key_padding_mask=key_padding_mask)

        return enc_output
