# -*- coding: utf-8 -*-
"""Phase 1 of EGPO: offline reference-trace construction from the base model (question only).

For every training prompt, the *base policy itself* is queried with exactly the prompt used for training and
inference (system prompt + question, **no gold answer in the context**).  Sampling stops at the first sample whose
extracted answer equals the ground truth, with at most K samples per prompt:

    Step 1  sample one trace; adopt it if its answer is correct.
    Step 2  otherwise keep sampling (early stop at the first correct trace), up to K samples in total.
    Step 3  if none is correct, no reference is stored (the paper's  y_ref = "*" ); the example keeps only its
            ground-truth answer and is supervised by the format / answer rewards alone.

Output: a copy of the annotation file in which the last assistant message becomes
    "<think>REFERENCE REASONING</think>GOLD_ANSWER"      (valid reference)   or   "GOLD_ANSWER"   (no reference).
It is written to the ``annotation_path`` of the ``<dataset_name>_with_thinking2`` entry of ``data_config.py``.

Serving the base model (any OpenAI-compatible server, e.g. vLLM; the model must be the *base* policy that is
later trained):

    vllm serve /path/to/Qwen3-VL-8B-Instruct --served-model-name base --port 8000

Usage (K and the sampling temperature are required: use the values reported in the paper):

    python build_reference.py --dataset_name scienceqa_data --max_tries K --temperature T \
        --base_url http://localhost:8000/v1/chat/completions --model base --concurrency 64
"""
import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))  # open_r1/
from data_config import data_dict  # noqa: E402  # pylint: disable=wrong-import-position
from prompts import SYSTEM_PROMPT  # noqa: E402  # pylint: disable=wrong-import-position
from rewards import parse_completion, split_solution  # noqa: E402  # pylint: disable=wrong-import-position

BLANK_IMAGE = os.path.join(HERE, "..", "blank_tmp.jpg")  # same placeholder the trainer uses for text-only items
_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}


def image_to_data_uri(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{_MIME.get(os.path.splitext(path)[1].lower(), 'image/jpeg')};base64,{b64}"


def _msg_text(msg):
    return msg.get("value") or msg.get("content") or ""


def _is_user(msg):
    return msg.get("from", "") in ("user", "human") or msg.get("role", "") in ("user", "human")


def build_messages(item, image_root):
    """Rebuild the training-time prompt (see ``transform_conversation`` in grpo_egpo.py) as OpenAI chat messages."""
    users = [m for m in item["conversations"] if _is_user(m)]
    text = _msg_text(users[-1]) if users else ""
    images = list(item.get("image", []) or [])
    if not images:  # text-only item: the trainer feeds a blank placeholder image
        images = [BLANK_IMAGE]
        if "<image>" not in text:
            text = "<image>\n" + text
    else:
        missing = len(images) - text.count("<image>")
        if missing > 0:
            text = "<image>\n" * missing + text
    paths = [p if os.path.isabs(p) else os.path.join(image_root, p) for p in images]
    text = SYSTEM_PROMPT + "\n" + text
    parts, chunks = [], text.split("<image>")
    for i, chunk in enumerate(chunks):
        if chunk.strip():
            parts.append({"type": "text", "text": chunk.strip()})
        if i < len(paths) and i < len(chunks) - 1:
            parts.append({"type": "image_url", "image_url": {"url": image_to_data_uri(paths[i])}})
    return [{"role": "user", "content": parts}]


def sample_once(messages, args):
    resp = requests.post(
        args.base_url,
        headers={"Authorization": f"Bearer {os.environ.get('LLM_API_KEY', 'EMPTY')}"},
        json={"model": args.model, "messages": messages, "temperature": args.temperature,
              "top_p": args.top_p, "max_tokens": args.max_tokens, "stream": False},
        timeout=600,
    )
    return resp.json()["choices"][0]["message"]["content"] or ""


def build_one(item, args, image_root):
    """Return (new_item, has_reference)."""
    gold_raw = _msg_text(item["conversations"][-1])
    _, gold = split_solution(gold_raw)
    messages = build_messages(item, image_root)
    reference = None
    for _ in range(args.max_tries):
        try:
            out = sample_once(messages, args)
        except Exception as e:  # network / server error: count as a failed try
            print(f"sampling failed: {e}")
            continue
        _, reasoning, answer = parse_completion(out)
        if answer is not None and answer == gold and reasoning:
            reference = reasoning
            break  # first-correct early stop
    new = json.loads(json.dumps(item))
    last = new["conversations"][-1]
    value = f"<think>{reference}</think>{gold}" if reference else gold
    last["value" if "value" in last else "content"] = value
    return new, reference is not None


def main():
    ap = argparse.ArgumentParser(description="EGPO Phase 1: offline reference construction (question only)")
    ap.add_argument("--dataset_name", required=True, help="raw dataset key in data_config.py, e.g. scienceqa_data")
    ap.add_argument("--max_tries", type=int, required=True, help="K: maximum number of samples per prompt")
    ap.add_argument("--temperature", type=float, required=True, help="sampling temperature (paper value)")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_tokens", type=int, default=1500, help="same as the training max completion length")
    ap.add_argument("--base_url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", default="base", help="served name of the BASE policy")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: process only the first N items")
    args = ap.parse_args()

    raw = data_dict[args.dataset_name]
    out_cfg = data_dict[args.dataset_name + "_with_thinking2"]
    with open(raw["annotation_path"], "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    print(f"{len(data)} items from {raw['annotation_path']}; K={args.max_tries}, T={args.temperature}")

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(lambda it: build_one(it, args, raw["data_path"]), data))

    new_items = [r[0] for r in results]
    n_ref = sum(1 for r in results if r[1])
    print(f"valid references: {n_ref}/{len(results)} = {100.0 * n_ref / max(1, len(results)):.2f}%")
    os.makedirs(os.path.dirname(out_cfg["annotation_path"]), exist_ok=True)
    with open(out_cfg["annotation_path"], "w", encoding="utf-8") as f:
        json.dump(new_items, f, ensure_ascii=False, indent=2)
    print("saved to", out_cfg["annotation_path"])


if __name__ == "__main__":
    main()
