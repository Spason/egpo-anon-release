"""EGPO data splitting utility.

Convert annotation JSON/JSONL files declared in ``data_config.data_dict`` into
JSONL format, and optionally split them into train / test sets.
"""

import argparse
import json
import random

from data_config import data_dict


def json_to_jsonl(json_file, jsonl_file, if_split=False, split_ratio=0.01):
    """Convert a JSON or JSONL file to JSONL format.

    Args:
        json_file: Source ``.json`` or ``.jsonl`` file.
        jsonl_file: Destination ``.jsonl`` file path.
        if_split: When ``True``, randomly split the data into train / test files.
        split_ratio: Fraction of samples assigned to the test split.

    Returns:
        Either the destination jsonl path, or a ``(train_file, test_file)`` tuple
        when ``if_split`` is ``True``.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        if json_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)

    if not if_split:
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Converted file:", jsonl_file)
        return jsonl_file

    # Sample data at equal intervals by ratio, split into two files.
    total_len = len(data)
    indices = set(random.sample(range(total_len), int(total_len * split_ratio)))
    train_file = jsonl_file.replace(".jsonl", ".json").replace(".json", "_train.jsonl")
    test_file = jsonl_file.replace(".jsonl", ".json").replace(".json", "_test.jsonl")
    with open(train_file, 'w', encoding='utf-8') as f_train:
        for idx, item in enumerate(data):
            if idx not in indices:
                f_train.write(json.dumps(item, ensure_ascii=False) + '\n')
    with open(test_file, 'w', encoding='utf-8') as f_test:
        for idx, item in enumerate(data):
            if idx in indices:
                f_test.write(json.dumps(item, ensure_ascii=False) + '\n')
    return train_file, test_file


def parse_args():
    parser = argparse.ArgumentParser(description="EGPO Data Splitting Utility")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="scienceqa_data",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.01,
        help="Ratio for splitting the dataset.",
    )
    parser.add_argument(
        "--only_convert",
        action="store_true",
        help="Only convert JSON to JSONL format without splitting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    file_name = args.dataset_name
    # ``annotation_path`` is the training data path; ``valid_path`` is the validation path.
    # If ``valid_path`` is not available, data will be split; otherwise use it directly.
    json_file = data_dict[file_name]["annotation_path"]

    if args.only_convert:
        # Only perform format conversion, no data splitting.
        output_file = json_to_jsonl(
            json_file,
            jsonl_file=json_file.replace(".json", ".jsonl"),
            if_split=False,
        )
        print("Converted file:", output_file)
        return

    # Check if a separate validation path exists.
    if "valid_path" in data_dict[file_name] and data_dict[file_name]["valid_path"]:
        # Validation path exists, convert train and validation data separately.
        valid_json_file = data_dict[file_name]["valid_path"]
        train_file = json_to_jsonl(
            json_file,
            jsonl_file=json_file.replace(".jsonl", ".json").replace(".json", "_train.jsonl"),
            if_split=False,
        )
        test_file = json_to_jsonl(
            valid_json_file,
            jsonl_file=valid_json_file.replace(".jsonl", ".json").replace(".json", "_test.jsonl"),
            if_split=False,
        )
        print("train file:", train_file)
        print("test file:", test_file)
    else:
        # No validation path, split data by ratio.
        train_file, test_file = json_to_jsonl(
            json_file,
            jsonl_file=json_file.replace(".jsonl", ".json").replace(".json", ".jsonl"),
            if_split=True,
            split_ratio=args.split_ratio,
        )
        print("train file:", train_file)
        print("test file:", test_file)


if __name__ == "__main__":
    main()
