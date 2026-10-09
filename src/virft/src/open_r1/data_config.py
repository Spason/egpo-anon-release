"""
Data configuration for EGPO training datasets.

Each dataset entry should contain:
    - annotation_path: Path to the annotation JSON file
    - data_path: Path to the directory containing images/data
    - valid_path (optional): Path to the validation JSON file

Please update the paths below to match your local data directory structure.
"""

# ==================== Dataset Configurations ====================
# Modify the BASE_DATA_DIR to point to your local data directory
BASE_DATA_DIR = "<YOUR_DATA_DIR>"

scienceqa_data = {
    "annotation_path": f"{BASE_DATA_DIR}/scienceqa_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/scienceqa_data/",
    "valid_path": f"{BASE_DATA_DIR}/scienceqa_data/validation.json",
}
scienceqa_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/scienceqa_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/scienceqa_data/",
    "valid_path": f"{BASE_DATA_DIR}/scienceqa_data/validation.json",
}
scienceqa_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/scienceqa_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/scienceqa_data/",
    "valid_path": f"{BASE_DATA_DIR}/scienceqa_data/validation.json",
}

aqua_rat_data = {
    "annotation_path": f"{BASE_DATA_DIR}/aqua_rat_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/aqua_rat_data/",
}
aqua_rat_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/aqua_rat_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/aqua_rat_data/",
}
aqua_rat_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/aqua_rat_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/aqua_rat_data/",
}

logiqa2_data = {
    "annotation_path": f"{BASE_DATA_DIR}/logiqa2_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/logiqa2_data/",
}
logiqa2_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/logiqa2_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/logiqa2_data/",
}
logiqa2_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/logiqa2_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/logiqa2_data/",
}

mathqa_data = {
    "annotation_path": f"{BASE_DATA_DIR}/mathqa_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/mathqa_data/",
}
mathqa_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/mathqa_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/mathqa_data/",
}
mathqa_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/mathqa_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/mathqa_data/",
}

ai2_arc_data = {
    "annotation_path": f"{BASE_DATA_DIR}/ai2_arc_data/ARC-Challenge_train.json",
    "data_path": f"{BASE_DATA_DIR}/ai2_arc_data/",
}
ai2_arc_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/ai2_arc_data/ARC-Challenge_train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/ai2_arc_data/",
}
ai2_arc_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/ai2_arc_data/ARC-Challenge_train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/ai2_arc_data/",
}

csqa_data = {
    "annotation_path": f"{BASE_DATA_DIR}/csqa_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/csqa_data/",
}
csqa_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/csqa_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/csqa_data/",
}
csqa_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/csqa_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/csqa_data/",
}

aokvqa_data = {
    "annotation_path": f"{BASE_DATA_DIR}/aokvqa_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/aokvqa_data/",
}
aokvqa_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/aokvqa_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/aokvqa_data/",
}
aokvqa_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/aokvqa_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/aokvqa_data/",
}

geoqa_plus_data = {
    "annotation_path": f"{BASE_DATA_DIR}/geoqa_plus_data/train.json",
    "data_path": f"{BASE_DATA_DIR}/geoqa_plus_data/",
}
geoqa_plus_data_with_thinking = {
    "annotation_path": f"{BASE_DATA_DIR}/geoqa_plus_data/train_with_thinking.json",
    "data_path": f"{BASE_DATA_DIR}/geoqa_plus_data/",
}
geoqa_plus_data_with_thinking2 = {
    "annotation_path": f"{BASE_DATA_DIR}/geoqa_plus_data/train_with_thinking2.json",
    "data_path": f"{BASE_DATA_DIR}/geoqa_plus_data/",
}

# ==================== Dataset Registry ====================
data_dict = {
    'scienceqa_data': scienceqa_data,
    'scienceqa_data_with_thinking': scienceqa_data_with_thinking,
    'scienceqa_data_with_thinking2': scienceqa_data_with_thinking2,
    'aqua_rat_data': aqua_rat_data,
    'aqua_rat_data_with_thinking': aqua_rat_data_with_thinking,
    'aqua_rat_data_with_thinking2': aqua_rat_data_with_thinking2,
    'logiqa2_data': logiqa2_data,
    'logiqa2_data_with_thinking': logiqa2_data_with_thinking,
    'logiqa2_data_with_thinking2': logiqa2_data_with_thinking2,
    'mathqa_data': mathqa_data,
    'mathqa_data_with_thinking': mathqa_data_with_thinking,
    'mathqa_data_with_thinking2': mathqa_data_with_thinking2,
    'ai2_arc_data': ai2_arc_data,
    'ai2_arc_data_with_thinking': ai2_arc_data_with_thinking,
    'ai2_arc_data_with_thinking2': ai2_arc_data_with_thinking2,
    'csqa_data': csqa_data,
    'csqa_data_with_thinking': csqa_data_with_thinking,
    'csqa_data_with_thinking2': csqa_data_with_thinking2,
    'aokvqa_data': aokvqa_data,
    'aokvqa_data_with_thinking': aokvqa_data_with_thinking,
    'aokvqa_data_with_thinking2': aokvqa_data_with_thinking2,
    'geoqa_plus_data': geoqa_plus_data,
    'geoqa_plus_data_with_thinking': geoqa_plus_data_with_thinking,
    'geoqa_plus_data_with_thinking2': geoqa_plus_data_with_thinking2,
}
