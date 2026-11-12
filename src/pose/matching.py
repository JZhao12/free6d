"""Patch-level matching between query crops and rendered templates."""
import torch

# above this many similarity entries (T * Nq * N) the templates are processed in chunks
MAX_SIM_ENTRIES = 60e6


def mutual_nn(sim):
    """
    Mutual nearest neighbours of a similarity matrix.

    `sim`: (A, B) float, -inf for forbidden pairs. Returns (idx_a, idx_b), both (K,)
    long: the pairs (a, b) that are each other's argmax. Rows whose max is -inf are dropped.
    """
    device = sim.device
    if sim.shape[0] == 0 or sim.shape[1] == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty.clone()
    row_max, best_b = sim.max(dim=1)  # (A,)
    best_a = sim.argmax(dim=0)  # (B,)
    idx_a = torch.arange(sim.shape[0], device=device)
    keep = (best_a[best_b] == idx_a) & (row_max > float("-inf"))
    return idx_a[keep], best_b[keep]


def match_templates(q_feats, t_feats, t_valid):
    """
    Match the valid query patches against every template with mutual nearest neighbours.

    `q_feats`: (Nq, C) float32, valid query patches only. `t_feats`: (T, N, C) float16 or
    float32. `t_valid`: (T, N) bool. Returns a dict of tensors on the input device:
      nn_qt (T, Nq) long: best template patch of every query patch,
      best_sim (T, Nq) float32: its similarity (0 where the template has no valid patch),
      mutual (T, Nq) bool: the query patch and its best template patch are mutual argmax,
      score_sum (T,) float32: sum of best_sim over the mutual matches,
      count (T,) long: number of mutual matches.
    """
    T, N, _ = t_feats.shape
    Nq = q_feats.shape[0]
    device = q_feats.device
    nn_qt = torch.zeros(T, Nq, dtype=torch.long, device=device)
    best_sim = torch.zeros(T, Nq, dtype=torch.float32, device=device)
    mutual = torch.zeros(T, Nq, dtype=torch.bool, device=device)
    if Nq > 0 and N > 0 and T > 0:
        q = q_feats.float()
        chunk = max(1, int(MAX_SIM_ENTRIES // (Nq * N)))
        arange_q = torch.arange(Nq, device=device)[None]
        for s in range(0, T, chunk):
            e = min(T, s + chunk)
            sim = torch.einsum("qc,tnc->tqn", q, t_feats[s:e].float())
            sim = sim.masked_fill(~t_valid[s:e, None, :], float("-inf"))
            best, nn = sim.max(dim=-1)  # (t, Nq)
            nn_tq = sim.argmax(dim=1)  # (t, N)
            is_mutual = nn_tq.gather(1, nn) == arange_q
            finite = best > float("-inf")
            nn_qt[s:e] = nn
            best_sim[s:e] = torch.where(finite, best, torch.zeros_like(best))
            mutual[s:e] = is_mutual & finite
    score_sum = (best_sim * mutual).sum(dim=1)
    count = mutual.sum(dim=1)
    return {
        "nn_qt": nn_qt,
        "best_sim": best_sim,
        "mutual": mutual,
        "score_sum": score_sum,
        "count": count,
    }
