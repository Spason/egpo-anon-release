# -*- coding: utf-8 -*-
"""
Add reasoning process (thinking) to each sample in the training data.
Reads train_data.json, calls an LLM API to generate reasoning, then prepends it to the answer.

Usage:
    python add_thinking.py --dataset_name scienceqa_data --refresh --concurrency 80 --max_tokens 8000

Note: This script requires an OpenAI-compatible API endpoint.
      Set the following environment variables before running:
        - LLM_API_URL: The API endpoint URL (e.g., http://localhost:8000/v1/chat/completions)
        - LLM_API_KEY: The API key for authentication
        - LLM_MODEL_NAME: The model name to use (default: "default")
"""
import argparse
import base64
import json
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_config import data_dict  # noqa: E402  # pylint: disable=wrong-import-position


# ==================== LLM API Configuration ====================
LLM_API_URL = os.environ.get("LLM_API_URL", "http://localhost:8000/v1/chat/completions")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "YOUR_API_KEY")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "default")


def llm_prompt_api(prompt, model=None, max_tokens=16000):
    """
    Call an OpenAI-compatible LLM API to generate a response.

    Args:
        prompt: List of message dicts (OpenAI chat format)
        model: Model name (defaults to LLM_MODEL_NAME env var)
        max_tokens: Maximum tokens for generation

    Returns:
        Tuple of (response_text, None)
    """
    if model is None:
        model = LLM_MODEL_NAME

    max_retries = 2
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    for attempt in range(1, max_retries + 1):
        json_data = {
            "model": model,
            "messages": prompt,
            "temperature": 0.01,
            "top_p": 1,
            "max_tokens": max_tokens,
            "stream": False,
        }

        try:
            resp = requests.post(LLM_API_URL, headers=headers, json=json_data, timeout=120)
            return_json = resp.json()
            return_text = return_json["choices"][0]["message"]["content"]
            if return_text:
                return return_text, None
        except Exception as e:
            print(f"API call failed (attempt {attempt}/{max_retries}): {e}")

    return "", None


# ==================== Image Utilities ====================
def image_file_to_base64(image_path):
    """Read a local image file and convert to base64 data URI."""
    try:
        with open(image_path, "rb") as f:
            img_data = f.read()
        base64_str = base64.b64encode(img_data).decode("utf-8")
        ext = os.path.splitext(image_path)[1].lower()
        mime_map = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                    '.gif': 'image/gif', '.webp': 'image/webp'}
        mime_type = mime_map.get(ext, 'image/jpeg')
        return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return None


def process_single_item(item, pic_dir_url, max_tokens=8000):
    """
    Process a single data item: generate thinking/reasoning using LLM.

    Args:
        item: Data item dict with 'conversations' and optional 'image' fields
        pic_dir_url: Base directory for image files
        max_tokens: Max tokens for LLM generation

    Returns:
        Updated item with thinking prepended to the answer, or None on failure
    """
    try:
        conversations = item.get("conversations", [])
        if not conversations:
            return None

        # Build prompt from conversations
        user_msg = None
        assistant_msg = None
        for msg in conversations:
            role = msg.get("from", msg.get("role", ""))
            content = msg.get("value", msg.get("content", ""))
            if role in ("user", "human"):
                user_msg = content
            elif role in ("gpt", "assistant"):
                assistant_msg = content

        if not user_msg or not assistant_msg:
            return None

        # Build LLM prompt for generating thinking
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Please provide step-by-step reasoning."},
            {"role": "user", "content": f"Given the following question and answer, generate a detailed reasoning process.\n\nQuestion: {user_msg}\n\nAnswer: {assistant_msg}\n\nPlease provide the reasoning process that leads to this answer."}
        ]

        thinking, _ = llm_prompt_api(prompt, max_tokens=max_tokens)
        if thinking:
            # Prepend thinking to the answer
            new_answer = f"<think>{thinking}</think>{assistant_msg}"
            # Update the last assistant message
            for msg in conversations:
                role = msg.get("from", msg.get("role", ""))
                if role in ("gpt", "assistant"):
                    if "value" in msg:
                        msg["value"] = new_answer
                    elif "content" in msg:
                        msg["content"] = new_answer
            return item

    except Exception as e:
        print(f"Error processing item: {e}")
        traceback.print_exc()

    return None


def main():
    parser = argparse.ArgumentParser(description="Add thinking/reasoning to training data")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset in data_config")
    parser.add_argument("--refresh", action="store_true", help="Regenerate thinking even if output file exists")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent API calls")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Max tokens for LLM generation")
    args = parser.parse_args()

    dataset_config = data_dict[args.dataset_name]
    input_file = dataset_config["annotation_path"]
    pic_dir_url = dataset_config["data_path"]
    output_file = input_file.replace(".json", "_with_thinking.json")

    if os.path.exists(output_file) and not args.refresh:
        print(f"Output file already exists: {output_file}. Use --refresh to regenerate.")
        return

    print(f"Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} items with concurrency={args.concurrency}")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(process_single_item, item, pic_dir_url, args.max_tokens): i
                   for i, item in enumerate(data)}
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    print(f"Successfully processed {len(results)}/{len(data)} items")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
