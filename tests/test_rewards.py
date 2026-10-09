"""Sanity tests for the paper-aligned reward (run with: python tests/test_rewards.py)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "virft", "src", "open_r1"))
import rewards as R  # noqa: E402


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_levenshtein():
    assert R.levenshtein_distance("kitten", "sitting") == 3
    assert R.levenshtein_distance("", "abc") == 3
    assert R.levenshtein_distance("abc", "abc") == 0
    assert approx(R.edit_similarity("", ""), 1.0)
    assert approx(R.edit_similarity("abcd", "abcf"), 0.75)


def test_split_solution():
    assert R.split_solution("<think> step 1 </think>B") == ("step 1", "B")
    assert R.split_solution("B") == ("*", "B")
    assert R.split_solution("<think>x</think><answer>C</answer>") == ("x", "C")
    assert R.split_solution("<think>" + "a" * 3001 + "</think>D") == ("*", "D")  # over-long reference is invalid


def test_parse_completion():
    ok, th, an = R.parse_completion("<think>r</think><answer>A</answer>")
    assert ok and th == "r" and an == "A"
    ok, th, an = R.parse_completion("blah <think>r</think> <answer>A</answer> tail")
    assert (not ok) and th == "r" and an == "A"
    ok, th, an = R.parse_completion("<answer>A</answer>")
    assert (not ok) and th is None and an == "A"
    assert R.parse_completion("no tags") == (False, None, None)


def test_gating_and_annealing():
    sol = "<think>the cat sat</think>A"
    good = "<think>the cat sat</think><answer>A</answer>"
    wrong = "<think>the cat sat</think><answer>B</answer>"
    # step 0: alpha_r = rho = 0.7
    c = R.egpo_reward_components(good, sol, 0, 100)
    assert approx(c["alpha_r"], 0.7) and approx(c["r_ref"], 1.0)
    assert approx(c["total"], 0.1 * 1 + 0.7 * 1 + 1.0 * 1)
    # wrong answer -> reference term gated to zero, format + (zero) answer remain
    w = R.egpo_reward_components(wrong, sol, 0, 100)
    assert w["r_ref"] == 0.0 and w["r_ans"] == 0.0 and approx(w["total"], 0.1)
    # at/after step rho*S the reference weight is zero (delta = 0)
    late = R.egpo_reward_components(good, sol, 70, 100)
    assert approx(late["alpha_r"], 0.0, 1e-6) and approx(late["total"], 1.1, 1e-6)
    # prompt without a valid reference -> no reference reward
    nr = R.egpo_reward_components(good, "A", 0, 100)
    assert nr["r_ref"] == 0.0 and approx(nr["total"], 1.1)
    # malformed format: R_fmt = 0 but a correct answer is still rewarded
    loose = R.egpo_reward_components("x <think>the cat sat</think><answer>A</answer>", sol, 0, 100)
    assert loose["r_fmt"] == 0.0 and loose["r_ans"] == 1.0


def test_sparse():
    sols = ["<think>r</think>A", "B"]
    outs = ["<think>z</think><answer>A</answer>", "<answer>C</answer>"]
    assert R.sparse_rewards(outs, sols) == [1.0, 0.0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
