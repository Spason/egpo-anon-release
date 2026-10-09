# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# ---- Patch: vLLM < 0.10 does not have StatelessProcessGroup, but trl 0.27 imports it unconditionally ----
# try:
#     from vllm.distributed.utils import StatelessProcessGroup  # noqa: F401
# except ImportError:
#     import vllm.distributed.utils as _vllm_dist_utils
#     if not hasattr(_vllm_dist_utils, "StatelessProcessGroup"):
#         _vllm_dist_utils.StatelessProcessGroup = type("StatelessProcessGroup", (), {})
# ---- End Patch ----

import json  # noqa: E402  # pylint: disable=wrong-import-position
import random  # noqa: E402  # pylint: disable=wrong-import-position
import re  # noqa: E402  # pylint: disable=wrong-import-position
from dataclasses import dataclass, field  # noqa: E402  # pylint: disable=wrong-import-position
from typing import Optional  # noqa: E402  # pylint: disable=wrong-import-position

from datasets import load_dataset  # noqa: E402  # pylint: disable=wrong-import-position
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config  # noqa: E402  # pylint: disable=wrong-import-position

from open_r1.trainer import Qwen2VLGRPOTrainer  # noqa: E402  # pylint: disable=wrong-import-position
from data_config import data_dict  # noqa: E402  # pylint: disable=wrong-import-position
import rewards as egpo_rewards_lib  # noqa: E402  # pylint: disable=wrong-import-position

# try:
#     from open_r1.trainer import Qwen2VLGRPOVLLMTrainer
# except ImportError:
#     Qwen2VLGRPOVLLMTrainer = None


# A small epsilon added to denominators to avoid division by zero.
_EPS = 1e-11


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    pg_name: Optional[str] = field(
        # Options: grpo, gpg, pinsker
        default="grpo",
        metadata={"help": "Only train the specified part of the model (grpo, gpg, pinsker)"},
    )
    adjust_gd: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to adjust the gradient of the model"},
    )
    min_inverse_alpha: Optional[float] = field(
        default=0.4,
        metadata={"help": "Minimum inverse alpha for the model"},
    )
    supervise_thinking_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "Annealing horizon ratio rho of the reference reward (paper: rho = 0.7)"},
    )
    reward_mode: str = field(
        default="egpo",
        metadata={
            "help": (
                "'egpo'  : paper reward  alpha_f*R_fmt + alpha_r(t)*R_ref + alpha_a*R_ans  (answer-gated R_ref);  "
                "'sparse': sparse 0/1 answer-correctness reward (GRPO baselines);  "
                "'legacy': earlier hierarchical format reward kept for backward compatibility."
            )
        },
    )
    alpha_f: float = field(default=0.1, metadata={"help": "Weight of the format reward (paper: 0.1)"})
    alpha_r0: float = field(default=1.0, metadata={"help": "Initial weight of the reference reward (paper: 1.0)"})
    alpha_a: float = field(default=1.0, metadata={"help": "Weight of the answer reward (paper: 1.0)"})
    anneal_floor: float = field(default=0.0, metadata={"help": "Floor delta of the annealed reference weight (paper: 0)"})
    monitor_kl_interval: int = field(
        default=0,
        metadata={
            "help": (
                "GPG/EGPO only. If > 0, load a frozen copy of the initial model and log KL(pi_theta || pi_base) every "
                "this many steps. Monitoring only: it never enters the loss. 0 (default) = reference model not loaded, "
                "i.e. a single resident model."
            )
        },
    )
    gpg_alpha_rescale: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, divide the loss by the fraction of non-degenerate groups (GPG's alpha rescaling). "
                "The paper uses alpha = 1, i.e. False (default)."
            )
        },
    )
    system_prompt_lang: Optional[str] = field(
        default="en",
        metadata={"help": "System prompt language: 'en' for English, 'zh' for Chinese. Default: en"},
    )
    system_prompt_prob: Optional[int] = field(
        default=100,
        metadata={
            "help": (
                "Probability of adding system prompt: random.randint(0,9) < this value. "
                "Range 0-10, default 10 means always add."
            )
        },
    )


def levenshtein_distance(s1, s2):
    """Compute the Levenshtein distance (edit distance) between two strings."""
    if len(s1) < len(s2):
        # Ensure s1 is the longer string for space optimization.
        return levenshtein_distance(s2, s1)

    m, n = len(s1), len(s2)

    # Space optimization: keep only two rows (prev and curr).
    prev = list(range(n + 1))  # Previous row of edit distances
    curr = [0] * (n + 1)       # Current row of edit distances

    for i in range(1, m + 1):
        curr[0] = i  # Initialize first column of current row (deletion)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]  # Characters match, no operation needed
            else:
                curr[j] = min(
                    prev[j] + 1,       # Deletion
                    curr[j - 1] + 1,   # Insertion
                    prev[j - 1] + 1,   # Substitution
                )
        prev = curr[:]  # Update previous row

    return prev[n]  # Return final edit distance


def accuracy_reward(completions, solution, global_step, max_steps, **kwargs):
    """Reward function: compute edit distance similarity between completion and target string."""
    if not solution:
        return [0.0] * len(completions)  # Return 0 reward when no target string

    rewards = []
    for completion, target_string in zip(completions, solution):
        content = completion[0]["content"]  # Extract completion content
        distance = levenshtein_distance(target_string, content)
        max_len = max(len(target_string), len(content))
        similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0
        # Decay factor: similarity weight gradually decreases to 0 as training progresses.
        similarity = similarity * (1 - global_step / (max_steps + _EPS)) * 0.02
        rewards.append(similarity)
    return rewards


def is_valid_json(s):
    """Check if a string is valid JSON format."""
    if not isinstance(s, str):
        return False
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return eval(s)
    except Exception:
        pass
    return False


def key_intersec_arr(key, arr):
    """Return True if any item in `arr` is contained in `key`."""
    for item in arr:
        if item in key:
            return True
    return False


unstandard_data_key = ['图片描述', '推理', "理由", "分析", "综合评价"]
# Default value, overridden by script_args.supervise_thinking_ratio in main().
SUPERVISE_THINKING_RATIO = 0.2
# Default value, overridden by script_args.system_prompt_prob in main().
# System prompt is added when random.randint(0,99) < this value.
SYSTEM_PROMPT_PROB = 100


def format_reward(
    completions,
    solution,
    global_step,
    max_steps,
    input_raw_data,
    write_file=True,
    return_result_score=False,
    dataset_name="test",
    **kwargs,
):
    """Reward function: compute similarity when both solution and content are JSON or both are not."""
    display_result = 1 if random.randint(0, 50) == 1 else 0
    if not solution:
        return [0.0] * len(completions)  # Return 0 reward when no target string

    rewards = []
    supervise_thinking_ratio = SUPERVISE_THINKING_RATIO
    for completion, target_string in zip(completions, solution):
        content = completion[0]["content"]  # Extract completion content
        format_reward_score = 0
        full_score = 0

        # Extract optional thinking part from target string.
        if len(target_string.split("</think>")) >= 2:
            target_think_part = target_string.split("</think>")[0]
            target_string = target_string.split("</think>")[-1]
            # Truncate if thinking part is too long.
            if len(target_think_part) > 3000:
                target_think_part = "*"
            print("Has thinking part")
        else:
            target_think_part = "*"

        # Check if think tags exist in input_raw_data, extract think and answer parts via regex.
        # Hierarchical scoring: strict match > basic match > partial match > fragment match > non-compliant (penalty).
        # Also handle the case where only aaa<answer>xx</answer> exists (aaa is treated as think).
        if (
            ("<think>" in str(input_raw_data) and "<answer>" in str(input_raw_data))
            or ("<answer>" in str(input_raw_data))
        ):
            think_part = None
            answer_part = None
            format_level = 0  # 0 = not matched
            tags_present = 0
            content_str = str(content)

            # Level 4 (strictest): complete <think>...</think><answer>...</answer> with no extra content.
            strict_match = re.search(
                r"^\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*$",
                content_str,
                re.DOTALL,
            )
            if strict_match:
                think_part = strict_match.group(1).strip()
                answer_part = strict_match.group(2).strip()
                format_level = 4

            # Level 3: has all four tags but may have extra content before/after.
            if format_level == 0:
                level3_match = re.search(
                    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
                    content_str,
                    re.DOTALL,
                )
                if level3_match:
                    think_part = level3_match.group(1).strip()
                    answer_part = level3_match.group(2).strip()
                    format_level = 3

            # Level 2: has <think>...</think> and <answer> but missing </answer>.
            if format_level == 0:
                level2_match = re.search(
                    r"<think>(.*?)</think>\s*<answer>(.*)",
                    content_str,
                    re.DOTALL,
                )
                if level2_match:
                    think_part = level2_match.group(1).strip()
                    answer_part = level2_match.group(2).strip()
                    format_level = 2

            # Level 1.5: no <think>/</think> tags, but has aaa<answer>xx</answer> form.
            if format_level == 0:
                level1_5_match = re.search(
                    r"^(.*?)<answer>(.*?)</answer>\s*$",
                    content_str,
                    re.DOTALL,
                )
                if level1_5_match and level1_5_match.group(1).strip():
                    think_part = level1_5_match.group(1).strip()
                    answer_part = level1_5_match.group(2).strip()
                    format_level = 1

            # Level 1: only </think> exists, missing <think> or <answer> tag.
            if format_level == 0:
                level1_match = re.search(r"(.*)</think>\s*(.*)", content_str, re.DOTALL)
                if level1_match:
                    think_part = level1_match.group(1).strip()
                    answer_part = level1_match.group(2).strip()
                    format_level = 1

            format_level_scores = {4: 2.0, 3: 1.5, 2: 1.0, 1: 1.0}
            if format_level > 0:
                format_reward_score += format_level_scores[format_level]
            else:
                # Level 0: no regex matched; check individual tag presence and score proportionally.
                # Four tags: <think>, </think>, <answer>, </answer>
                tag_checks = ["<think>", "</think>", "<answer>", "</answer>"]
                tags_present = sum(1 for tag in tag_checks if tag in content_str)
                tag_ratio_scores = {0: -3.0, 1: -2.0, 2: -1.0, 3: -0.5}
                format_reward_score += tag_ratio_scores[tags_present]
                # Extra: if tag order is correct, add small bonus.
                if tags_present >= 2:
                    pos_think_open = content_str.find("<think>")
                    pos_think_close = content_str.find("</think>")
                    pos_answer_open = content_str.find("<answer>")
                    pos_answer_close = content_str.find("</answer>")
                    order_bonus = 0.0
                    if pos_think_open != -1 and pos_think_close != -1 and pos_think_open < pos_think_close:
                        order_bonus += 0.02  # <think> before </think>
                    if pos_answer_open != -1 and pos_answer_close != -1 and pos_answer_open < pos_answer_close:
                        order_bonus += 0.02  # <answer> before </answer>
                    if pos_think_close != -1 and pos_answer_open != -1 and pos_think_close < pos_answer_open:
                        order_bonus += 0.02  # </think> before <answer>
                    format_reward_score += order_bonus

            if format_level > 0 and answer_part is not None:
                content = answer_part
                # If target_think_part exists, compute similarity between think_part and target_think_part.
                if target_think_part != "*" and think_part:
                    distance = levenshtein_distance(target_think_part, think_part)
                    max_len = max(len(target_think_part), len(think_part))
                    similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0
                    if (
                        len(target_think_part) > len(think_part) * 2.5
                        or len(target_think_part) < min(80, len(think_part) / 2)
                    ) or global_step < max_steps * supervise_thinking_ratio:
                        format_reward_score += (
                            similarity
                            * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                            * 1
                        )
                    else:
                        format_reward_score += (
                            1
                            * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                            * 1
                        )
                if random.randint(0, 100) == 1:
                    print(
                        f"Format matched (level={format_level}) "
                        f"global_step:{global_step},max_steps:{max_steps},"
                        f"thinking:{think_part},answer_part:{answer_part}"
                    )
                print(f"Format matched (level={format_level}) global_step:{global_step},max_steps:{max_steps}")
            else:
                # format_level == 0: no regex matched, score based on tag count.
                if random.randint(0, 100) == 1:
                    print(
                        f"Format non-compliant (level=0, tags={tags_present}/4) "
                        f"global_step:{global_step},max_steps:{max_steps},"
                        f"content:{content_str[:200]}"
                    )
                print(
                    f"Format non-compliant (level=0, tags={tags_present}/4) "
                    f"global_step:{global_step},max_steps:{max_steps}"
                )

            if target_think_part != "*":
                full_score += 2
            else:
                full_score += 0

        # Check if both solution and content are JSON or both are not JSON.
        target_json = is_valid_json(target_string)
        content_json = is_valid_json(content)
        if target_json and content_json:
            try:
                # Compare different keys separately; first check for matching keys.
                if isinstance(target_json, dict) and isinstance(content_json, dict):
                    # Compute similarity as intersection size / union size of keys.
                    target_keys = set(target_json.keys())
                    content_keys = set(content_json.keys())
                    intersection = target_keys.intersection(content_keys)
                    union = target_keys.union(content_keys)
                    similarity = len(intersection) / len(union) if union else 1.0
                    format_reward_score += similarity
                    full_score += 1
                    if len(intersection) == len(union) and len(union) > 0:
                        # Compute actual value length; if short, use unweighted edit distance.
                        flag_all_key_unstandard = True
                        for key in intersection:
                            # Check if the key belongs to non-standard data keys.
                            if not key_intersec_arr(key, unstandard_data_key):
                                flag_all_key_unstandard = False
                                break
                        for key in intersection:
                            # Check if key belongs to non-standard data keys.
                            if key_intersec_arr(key, unstandard_data_key):
                                value = target_json[key]
                                # Compute edit distance similarity; if ground truth is "*", default similarity to 1.
                                if str(value) == "*":
                                    similarity = 1
                                else:
                                    distance = levenshtein_distance(str(value), str(content_json.get(key, "")))
                                    max_len = max(len(str(value)), len(str(content_json.get(key, ""))))
                                    similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0
                                # Decay factor: weight decreases to 0 as training progresses.
                                if flag_all_key_unstandard:
                                    format_reward_score += (
                                        similarity * (1 - global_step / (max_steps + _EPS) + 0.05) * 0.1
                                    )
                                    full_score += 1 * (1 - global_step / (max_steps + _EPS) + 0.05) * 0.1
                                else:
                                    if str(value) != "*":
                                        if (
                                            len(value) > len(str(content_json.get(key, ""))) * 2.5
                                            or len(value) < min(80, len(str(content_json.get(key, ""))) / 2)
                                        ) or global_step < max_steps * supervise_thinking_ratio:
                                            format_reward_score += (
                                                similarity
                                                * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                                                * 0.05
                                            )
                                        else:
                                            format_reward_score += (
                                                1
                                                * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                                                * 0.05
                                            )
                                    else:
                                        format_reward_score += (
                                            1
                                            * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                                            * 0.05
                                        )
                                    full_score += (
                                        1
                                        * max(supervise_thinking_ratio - global_step / (max_steps + _EPS), 0)
                                        * 0.05
                                    )
                            else:
                                # For standard keys, evaluate using edit distance.
                                value = target_json[key]
                                distance = levenshtein_distance(str(value), str(content_json.get(key, "")))
                                max_len = max(len(str(value)), len(str(content_json.get(key, ""))))
                                similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0
                                format_reward_score += similarity
                                full_score += 1
            except Exception:
                print("error json")
            # Reset similarity.
            similarity = 0
            rewards.append(format_reward_score + similarity)
            if display_result == 1:
                print(
                    f'\033[32m JSON format reward \n target_string: ', target_string,
                    '\033[0m \n \033[31m content: ', completion[0]["content"],
                    'similarity(json): ', format_reward_score + similarity, "\033[0m",
                )
        elif (target_json and not content_json) or (not target_json and content_json):
            # If target is JSON but content is not, give minimal reward.
            distance = levenshtein_distance(target_string, content)
            max_len = max(len(target_string), len(content))
            similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0

            full_score += 1 * 0.01
            rewards.append(format_reward_score + similarity * 0.01)
            if display_result == 1:
                print(
                    '\033[32m JSON format mismatch target_string: ', target_string,
                    '\033[0m \n \033[31m content: ', completion[0]["content"],
                    'similarity(string_json): ', 0.0, "\033[0m",
                )
            continue
        else:
            # Fallback: compute similarity when reward conditions are met.
            distance = levenshtein_distance(target_string, content)
            max_len = max(len(target_string), len(content))
            similarity = 1 - (float(distance) / float(max_len)) if max_len > 0 else 1.0
            full_score += 1
            if display_result == 1:
                print(
                    '\033[32m Non-JSON reward target_string: ', target_string,
                    '\033[0m \n \033[31m content: ', content,
                    'similarity(string): ', similarity, "\033[0m",
                )
            rewards.append(format_reward_score + similarity)

    # Save intermediate results for debugging.
    if write_file:
        try:
            dataset_name = script_args.dataset_name  # noqa: F823
        except Exception:
            dataset_name = "test"
        os.makedirs("middle_result", exist_ok=True)
        with open(f"middle_result/{dataset_name}_format_reward_results.json", "a", encoding="utf-8") as f:
            tmp_result = []
            target = ""
            for completion, target_string, reward in zip(completions, solution, rewards):
                target = target_string
                tmp_result.append({
                    "output": str(completion[0]["content"]),
                    "reward": str(reward),
                })
            record = {
                "input": str(input_raw_data),
                "target": str(target),
                "result": tmp_result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if display_result == 1:
        print(rewards)
    if return_result_score:
        return rewards, full_score
    return rewards


# Reward hyper-parameters; overwritten in main() from the command line.
EGPO_REWARD_CFG = dict(alpha_f=0.1, alpha_r0=1.0, alpha_a=1.0, rho=0.7, delta=0.0)


def egpo_reward(completions, solution, global_step, max_steps, **kwargs):
    """Paper reward: alpha_f*R_fmt + alpha_r(t)*R_ref + alpha_a*R_ans (see rewards.py)."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return egpo_rewards_lib.egpo_rewards(texts, solution, global_step, max_steps, **EGPO_REWARD_CFG)


def sparse_reward(completions, solution, **kwargs):
    """Sparse 0/1 answer-correctness reward used by the GRPO baselines."""
    texts = [c[0]["content"] if isinstance(c, list) else str(c) for c in completions]
    return egpo_rewards_lib.sparse_rewards(texts, solution)


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "egpo": egpo_reward,
    "sparse": sparse_reward,
}

from prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_zh  # noqa: E402  # pylint: disable=wrong-import-position


def json_to_jsonl(json_file, jsonl_file, if_split=False, split_ratio=0.01):
    """Convert JSON file to JSONL format, optionally splitting into train/test."""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not if_split:
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        return jsonl_file
    # Sample data at equal intervals by ratio, split into two files.
    total_len = len(data)
    indices = set(random.sample(range(total_len), int(total_len * split_ratio)))
    train_file = jsonl_file.replace(".jsonl", "_train.jsonl")
    test_file = jsonl_file.replace(".jsonl", "_test.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f_train:
        for idx, item in enumerate(data):
            if idx not in indices:
                f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f_test:
        for idx, item in enumerate(data):
            if idx in indices:
                f_test.write(json.dumps(item, ensure_ascii=False) + '\n')
    return train_file, test_file


def main(script_args, training_args, model_args):
    global SUPERVISE_THINKING_RATIO, SYSTEM_PROMPT, SYSTEM_PROMPT_PROB
    SUPERVISE_THINKING_RATIO = script_args.supervise_thinking_ratio
    SYSTEM_PROMPT_PROB = script_args.system_prompt_prob
    # Select system prompt language based on configuration.
    if script_args.system_prompt_lang == "zh":
        SYSTEM_PROMPT = SYSTEM_PROMPT_zh
        print("Using Chinese System Prompt")
    else:
        print("Using English System Prompt")

    train_json_file = data_dict[script_args.dataset_name]["annotation_path"]
    # valid_path
    if "valid_path" in data_dict[script_args.dataset_name]:
        valid_json_file = data_dict[script_args.dataset_name]["valid_path"]
    else:
        valid_json_file = train_json_file
    pic_dir_url = data_dict[script_args.dataset_name]["data_path"]
    train_jsonl_file = train_json_file.replace(".jsonl", ".json").replace(".json", ".jsonl")
    valid_jsonl_file = valid_json_file.replace(".jsonl", ".json").replace(".json", ".jsonl")

    training_args.pg_name = script_args.pg_name
    assert training_args.pg_name in ["grpo", "gpg", "pinsker"], (
        f"pg_name {training_args.pg_name} is not supported"
    )
    training_args.adjust_gd = script_args.adjust_gd
    training_args.min_inverse_alpha = script_args.min_inverse_alpha
    training_args.monitor_kl_interval = script_args.monitor_kl_interval
    training_args.gpg_alpha_rescale = script_args.gpg_alpha_rescale

    # Get reward functions.
    assert script_args.reward_mode in ("egpo", "sparse", "legacy"), script_args.reward_mode
    EGPO_REWARD_CFG.update(
        alpha_f=script_args.alpha_f,
        alpha_r0=script_args.alpha_r0,
        alpha_a=script_args.alpha_a,
        rho=script_args.supervise_thinking_ratio,
        delta=script_args.anneal_floor,
    )
    script_args.reward_funcs = {"egpo": ["egpo"], "sparse": ["sparse"], "legacy": ["format"]}[script_args.reward_mode]
    print(f"Reward mode: {script_args.reward_mode}; reward functions: {script_args.reward_funcs}")
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # ---- Data preprocessing ----
    from datasets import DatasetDict

    # Replace original make_conversation and make_conversation_image functions.
    def transform_conversation(example, valid=False):
        # Extract last user message (compatible with both from/value and role/content formats).
        user_messages = [
            msg for msg in example["conversations"]
            if msg.get("from", "") in ("user", "human") or msg.get("role", "") in ("user", "human")
        ]
        last_user_message = (
            (user_messages[-1].get("value") or user_messages[-1].get("content") or "")
            if user_messages else ""
        )

        # Process images.
        images = example.get("image", [])
        blank_image_path = os.path.join(os.path.dirname(__file__), "blank_tmp.jpg")
        if not images:
            # No images: add a blank placeholder image and prepend <image> tag to text.
            images = [blank_image_path]
            if "<image>" not in last_user_message:
                last_user_message = "<image>\n" + last_user_message
        else:
            # Has images but insufficient <image> tags in text: prepend missing tags.
            existing_tag_count = last_user_message.count("<image>")
            if existing_tag_count < len(images):
                missing_tags = "".join(["<image>\n" for _ in range(len(images) - existing_tag_count)])
                last_user_message = missing_tags + last_user_message

        if images:
            # Build multi-modal content.
            image_prompt = []
            imgs_arr = []
            # Modify here to support more images if token length allows.
            max_pic = 200
            len_pic = min(max_pic, len(images))
            for image_path in images[:len_pic]:
                image_prompt.append({"type": "image"})
                if os.path.isabs(image_path):
                    imgs_arr.append({'path': image_path})
                else:
                    imgs_arr.append({'path': os.path.join(pic_dir_url, image_path)})
            if len(images) > max_pic:
                # This may not work well for multi-segment images; needs future improvement.
                print("warning: image and text split error len(imgaes):", len(images))
                text_content = (
                    "".join(["<image>\n" for _ in range(len(imgs_arr))])
                    + last_user_message.replace("<image>", "").strip()
                )
            else:
                text_content = last_user_message

            # Check if data contains non-standard keys; if not, add SYSTEM_PROMPT with probability.
            if not valid:
                try:
                    _last_conv_val = (
                        example["conversations"][-1].get("value")
                        or example["conversations"][-1].get("content")
                        or ""
                    )
                    try:
                        tt_data = json.loads(_last_conv_val)
                    except Exception:
                        try:
                            tt_data = eval(_last_conv_val)
                        except Exception:
                            tt_data = None
                    flag_all_key_unstandard = False
                    if isinstance(tt_data, dict):
                        for key in tt_data:
                            # Check if key belongs to non-standard data keys.
                            if key_intersec_arr(key, unstandard_data_key):
                                flag_all_key_unstandard = True
                                break
                    if not flag_all_key_unstandard:
                        if random.randint(0, 99) < SYSTEM_PROMPT_PROB:
                            text_content = SYSTEM_PROMPT + "\n" + text_content
                except Exception:
                    if random.randint(0, 99) < SYSTEM_PROMPT_PROB:
                        text_content = SYSTEM_PROMPT + "\n" + text_content

            text_content_split = text_content.split("<image>")
            content_list = []
            if len(text_content_split) == len_pic + 1:
                for i, text_part in enumerate(text_content_split):
                    text_part = text_part.strip()
                    if text_part != "":
                        content_list.append({"type": "text", "text": text_part})
                    if i < len_pic:
                        content_list.append({"type": "image"})
            else:
                content_list = image_prompt + [{"type": "text", "text": text_content}]
                print("warning: image and text split error")

            example['image'] = imgs_arr
            return {
                "prompt": [{"role": "user", "content": content_list}],
                'solution': (
                    example["conversations"][-1].get("value")
                    or example["conversations"][-1].get("content")
                    or ""
                ),
                'image_pic': imgs_arr,
            }
        else:
            # Fallback for pure text (no image).
            print("error:", user_messages)
            return {
                "prompt": [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Who are you?"},
                ]}],
                'solution': "I am a helpful assistant.",
                'image_pic': [{
                    "path": os.path.join(os.path.dirname(__file__), "blank_tmp.jpg"),
                }],
            }

    # eval_strategy
    split_flag = training_args.eval_strategy in ["steps", "epoch"]
    if split_flag:
        print("Splitting dataset into train and validation sets")
        train_jsonl_file = os.path.join(
            "/".join(train_jsonl_file.split("/")[:-1]),
            os.path.basename(train_json_file).replace(".jsonl", ".json").replace(".json", "_train.jsonl"),
        )
        test_jsonl_file = os.path.join(
            "/".join(valid_jsonl_file.split("/")[:-1]),
            os.path.basename(valid_json_file).replace(".jsonl", ".json").replace(".json", "_test.jsonl"),
        )
        dataset_train = load_dataset('json', data_files=train_jsonl_file)
        # Limit validation set to at most 40 samples.
        dataset_test = load_dataset('json', data_files=test_jsonl_file)
        if len(dataset_test["train"]) > 40:
            dataset_test["train"] = dataset_test["train"].select(range(40))
        dataset_train = dataset_train.map(transform_conversation)
        dataset_train = dataset_train.remove_columns(['conversations', 'image'])
        dataset_train = dataset_train.rename_column("image_pic", "image")
        dataset_test = dataset_test.map(lambda x: transform_conversation(x, valid=True))
        dataset_test = dataset_test.remove_columns(['conversations', 'image'])
        dataset_test = dataset_test.rename_column("image_pic", "image")
        dataset = DatasetDict({
            "train": dataset_train["train"],
            "test": dataset_test["train"],
        })
    else:
        if ".jsonl" not in train_json_file:
            train_jsonl_file = json_to_jsonl(train_json_file, train_jsonl_file)
        print("JSONL conversion completed")
        dataset = load_dataset('json', data_files=train_jsonl_file)
        dataset = dataset.map(transform_conversation)
        dataset = dataset.remove_columns(['conversations', 'image'])
        dataset = dataset.rename_column("image_pic", "image")

    print("Data transformation completed")
    # trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    trainer_cls = Qwen2VLGRPOTrainer
    print("using: ", trainer_cls)
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if split_flag else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    trainer.train()
    # Save and push to hub.
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
