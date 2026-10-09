"""Paper-aligned reward functions for EGPO (pure Python, no torch / trl dependency).

Notation follows the paper:

    R_total = alpha_f * R_fmt + alpha_r(t) * R_ref + alpha_a * R_ans
    R_fmt   = 1[output matches  <think> ... </think> <answer> ... </answer>  exactly]
    R_ans   = 1[extracted answer == ground-truth answer]                (exact match)
    R_ref   = 1 - d_edit(reasoning, reference) / max(|reasoning|, |reference|)
    alpha_r(t) = alpha_r0 * max(rho - t / S, delta)                     (linear annealing)

R_ref is *gated* by answer correctness: a rollout whose answer is wrong receives alpha_r = 0, and a
prompt without a valid reference trace (reference == "*") receives no reference reward.

The ground-truth string ("solution") of a training example is either

    "<think>REFERENCE REASONING</think>GOLD_ANSWER"      (a valid reference exists), or
    "GOLD_ANSWER"                                        (no valid reference; Step 3 of the paper).

A reference longer than ``MAX_REF_CHARS`` characters is treated as invalid.
"""

import re

MAX_REF_CHARS = 3000
INVALID_REF = "*"

_STRICT = re.compile(r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$", re.DOTALL)
_LOOSE = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
_ANSWER_ONLY = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_GOLD_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def levenshtein_distance(s1, s2):
    """Levenshtein distance (insert / delete / substitute, unit cost), O(min(m, n)) memory."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (c1 != c2)))
        prev = curr
    return prev[-1]


def edit_similarity(a, b):
    """1 - d_edit(a, b) / max(|a|, |b|), in [0, 1]; two empty strings are identical (1.0)."""
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - levenshtein_distance(a, b) / m


def split_solution(solution):
    """Split a ground-truth string into (reference_reasoning, gold_answer).

    ``reference_reasoning`` is ``INVALID_REF`` ("*") when no usable reference exists.
    """
    solution = solution or ""
    if "</think>" in solution:
        head, _, tail = solution.rpartition("</think>")
        ref = head.split("<think>", 1)[-1].strip()
        if not ref or len(ref) > MAX_REF_CHARS:
            ref = INVALID_REF
    else:
        ref, tail = INVALID_REF, solution
    m = _GOLD_TAG.search(tail)
    gold = (m.group(1) if m else tail).strip()
    return ref, gold


def parse_completion(text):
    """Extract (strict_format_ok, reasoning, answer) from a model output.

    ``strict_format_ok`` is True only for ``<think>..</think><answer>..</answer>`` with nothing else.
    Reasoning / answer are still extracted from a looser match so that R_ans and R_ref are defined
    for slightly malformed outputs; they are ``None`` when the corresponding tag pair is absent.
    """
    text = str(text)
    m = _STRICT.match(text)
    if m:
        return True, m.group(1).strip(), m.group(2).strip()
    m = _LOOSE.search(text)
    if m:
        return False, m.group(1).strip(), m.group(2).strip()
    m = _ANSWER_ONLY.search(text)
    return False, None, (m.group(1).strip() if m else None)


def anneal_weight(step, max_steps, alpha_r0=1.0, rho=0.7, delta=0.0, eps=1e-11):
    """alpha_r(t) = alpha_r0 * max(rho - t / S, delta)."""
    return alpha_r0 * max(rho - step / (max_steps + eps), delta)


def egpo_reward_components(text, solution, step, max_steps,
                           alpha_f=0.1, alpha_r0=1.0, alpha_a=1.0, rho=0.7, delta=0.0):
    """Return a dict with the three sub-rewards, the annealed weight and the total reward."""
    ref, gold = split_solution(solution)
    fmt_ok, reasoning, answer = parse_completion(text)
    r_fmt = 1.0 if fmt_ok else 0.0
    r_ans = 1.0 if (answer is not None and answer == gold) else 0.0
    a_r = anneal_weight(step, max_steps, alpha_r0, rho, delta)
    r_ref = 0.0
    if r_ans > 0 and ref != INVALID_REF and reasoning:  # answer gating + valid reference
        r_ref = edit_similarity(reasoning, ref)
    total = alpha_f * r_fmt + a_r * r_ref + alpha_a * r_ans
    return {"r_fmt": r_fmt, "r_ref": r_ref, "r_ans": r_ans, "alpha_r": a_r, "total": total}


def egpo_rewards(completions, solutions, step, max_steps, **cfg):
    """Batch version: ``completions`` is a list of strings, ``solutions`` a list of ground-truth strings."""
    return [egpo_reward_components(c, s, step, max_steps, **cfg)["total"]
            for c, s in zip(completions, solutions)]


def sparse_rewards(completions, solutions, **_):
    """Sparse 0/1 correctness reward used by the GRPO baselines (no format or reference component)."""
    out = []
    for c, s in zip(completions, solutions):
        _, gold = split_solution(s)
        _, _, answer = parse_completion(c)
        out.append(1.0 if (answer is not None and answer == gold) else 0.0)
    return out
