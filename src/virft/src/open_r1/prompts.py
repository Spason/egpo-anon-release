"""System prompts shared by training (grpo_egpo.py) and reference construction (build_reference.py)."""

SYSTEM_PROMPT = (
    "You are a helpful assistant. A conversation between User and Assistant. "
    "The user asks a question, and the Assistant solves it. The Assistant first "
    "thinks about the reasoning process in the mind and then provides the user "
    "with the answer. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)
SYSTEM_PROMPT_zh = (
    "用户与助手之间的对话。用户提出一个问题，助手负责解答。助手首先在脑海中思考推理过程，"
    "然后向用户提供答案。推理过程和答案分别用 <think> </think> 和 <answer> </answer> "
    "标签包裹，即：<think> 在此处填写推理过程 </think><answer> 在此处填写答案</answer>"
)
