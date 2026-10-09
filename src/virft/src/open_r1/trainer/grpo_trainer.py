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
import gc
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from tqdm import tqdm
import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLConfig,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
try:
    from transformers import Qwen3VLForConditionalGeneration
except:
    print("Qwen3-VL model loading not supported")

# Qwen3.5 native multimodal model support (requires newer version of transformers)
HAS_QWEN3_5 = False
HAS_AUTO_IMAGE_TEXT = False
try:
    from transformers import Qwen3_5ForConditionalGeneration
    HAS_QWEN3_5 = True
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText
        HAS_AUTO_IMAGE_TEXT = True
    except ImportError:
        pass
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
import random
from contextlib import contextmanager

import logging
from PIL import Image

# ======== [EasyR1 Optimization] Flash-Attn Triton Cross Entropy ========
# Reference: EasyR1/verl/utils/torch_functional.py
# Triton kernel is ~2x faster than PyTorch log_softmax + gather, and supports inplace backward to save memory
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
    FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False

@torch.compiler.disable()
def _log_probs_from_logits_flash_attn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probs using flash_attn Triton cross entropy kernel."""
    output = cross_entropy_loss(logits, labels, inplace_backward=True)
    if not isinstance(output, tuple):
        raise ValueError("Please ensure flash-attn>=2.4.3, cross_entropy_loss should return Tuple[losses, z_losses].")
    return -output[0]

def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probs at label positions for given logits.
    
    Prefers flash_attn Triton kernel (~2x speedup), falls back to F.cross_entropy.
    Reference: EasyR1/verl/utils/torch_functional.py
    
    Args:
        logits: (*, vocab_size) model output logits
        labels: (*,) corresponding label ids
    Returns:
        log_probs: (*,) log prob at each position
    """
    batch_dim = logits.shape[:-1]
    vocab_dim = logits.shape[-1]
    logits_flat = logits.contiguous().view(-1, vocab_dim)
    labels_flat = labels.contiguous().view(-1)
    if FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        output = _log_probs_from_logits_flash_attn(logits_flat, labels_flat)
    else:
        # fallback: upcast to fp32 for numerical stability
        output = -F.cross_entropy(logits_flat.float(), labels_flat, reduction="none")
    return output.view(*batch_dim)

logger = logging.getLogger(__name__)

class SanitizeLogitsProcessor(LogitsProcessor):
    """
    Sanitize inf/nan values in logits at each generation step.
    
    Under Qwen3.5 with DeepSpeed ZeRO-3 + bf16, logits may contain inf/nan during eval,
    causing abnormal probability distributions after softmax and triggering CUDA device-side
    assert during multinomial sampling.
    
    This processor replaces anomalous values before logits enter softmax/sampling,
    preventing CUDA assert at the source, effective for both do_sample=True and False.
    """
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Detect and replace nan and inf
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            # Get a safe fill value (minimum of finite values in scores, or a large negative number)
            finite_mask = torch.isfinite(scores)
            if finite_mask.any():
                fill_value = scores[finite_mask].min().item()
            else:
                fill_value = -1e9
            # nan -> fill_value, inf -> clamp
            scores = torch.where(torch.isnan(scores), torch.tensor(fill_value, device=scores.device, dtype=scores.dtype), scores)
            scores = torch.clamp(scores, min=-1e9, max=1e9)
        return scores

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            # print("model_id:", model_id)  # Debug only
            model_id_lower = model_id.lower()
            if "qwen3.5" in model_id_lower or "qwen3_5" in model_id_lower:
                # Qwen3.5 native multimodal model
                if HAS_QWEN3_5:
                    model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        model, attn_implementation=model_init_kwargs.get('attn_implementation'),
                        torch_dtype=torch.bfloat16)
                elif HAS_AUTO_IMAGE_TEXT:
                    print("Warning: Qwen3_5ForConditionalGeneration not found, using AutoModelForImageTextToText instead")
                    model = AutoModelForImageTextToText.from_pretrained(
                        model, attn_implementation=model_init_kwargs.get('attn_implementation'),
                        torch_dtype=torch.bfloat16, trust_remote_code=True)
                else:
                    raise ImportError(
                        "Cannot load Qwen3.5 model. Please upgrade transformers to a version that supports Qwen3.5: "
                        "pip install transformers>=4.58.0 or install from source.")
            elif "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs,torch_dtype=(torch.bfloat16 if 1 else None))                
            elif "Qwen3-VL" in model_id:
                model =Qwen3VLForConditionalGeneration.from_pretrained(model, attn_implementation=model_init_kwargs.get('attn_implementation'),dtype=(torch.bfloat16 if 1 else None))
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # # Reference model
        # A frozen reference copy is needed only by (i) GRPO with a KL penalty (beta > 0), (ii) the Pinsker variant,
        # or (iii) optional KL monitoring for GPG/EGPO (monitor_kl_interval > 0). Otherwise only the policy is resident.
        _pg_name = getattr(args, "pg_name", "grpo")
        _monitor_kl_interval = int(getattr(args, "monitor_kl_interval", 0) or 0)
        self._monitor_kl_interval = _monitor_kl_interval
        self._need_ref_model = (
            (_pg_name == "grpo" and float(getattr(args, "beta", 0.0) or 0.0) > 0.0)
            or _pg_name == "pinsker"
            or (_pg_name == "gpg" and _monitor_kl_interval > 0)
        )
        if not self._need_ref_model:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            model_id_lower = model_id.lower()
            if "qwen3.5" in model_id_lower or "qwen3_5" in model_id_lower:
                # Qwen3.5 ref model
                if HAS_QWEN3_5:
                    self.ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
                        model_id, attn_implementation=model_init_kwargs.get('attn_implementation'),
                        torch_dtype=torch.bfloat16)
                elif HAS_AUTO_IMAGE_TEXT:
                    self.ref_model = AutoModelForImageTextToText.from_pretrained(
                        model_id, attn_implementation=model_init_kwargs.get('attn_implementation'),
                        torch_dtype=torch.bfloat16, trust_remote_code=True)
                else:
                    raise ImportError("Cannot load Qwen3.5 ref model.")
            elif "Qwen2-VL" in model_id:

                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs,torch_dtype=(torch.bfloat16 if 1 else None))
            elif "Qwen3-VL" in model_id:
                # self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs,torch_dtype=(torch.bfloat16 if 1 else None))
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, attn_implementation=model_init_kwargs.get('attn_implementation'),dtype=(torch.bfloat16 if 1 else None))
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None



        # Processing class
        # 
        if processing_class is None:
            model_id_lower = model_id.lower()
            is_qwen_vl = ("Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id 
                          or "qwen3.5" in model_id_lower or "qwen3_5" in model_id_lower
                          or "Aria" in model_id)
            if is_qwen_vl:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                is_qwen_model = ("Qwen" in model_id or "Qwen2.5-VL" in model_id or "Qwen3-VL" in model_id
                                 or "qwen3.5" in model_id_lower or "qwen3_5" in model_id_lower)
                if is_qwen_model:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        # Mini-batch generation: generate mini_gen_batch_size sequences at a time, concatenate across iterations
        # [Memory-speed tradeoff] Set to 8 to generate all sequences at once, avoiding repeated KV cache rebuild + ViT forward
        # Cost: KV cache peak memory doubles during generation (8 sequences vs 4)
        # If OOM, revert to min(4, self.num_generations)
        self.mini_gen_batch_size = min(8, self.num_generations)  
        assert self.num_generations % self.mini_gen_batch_size == 0, \
            f"num_generations({self.num_generations}) must be divisible by mini_gen_batch_size({self.mini_gen_batch_size})"
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=args.temperature,
            num_return_sequences=self.mini_gen_batch_size,  # generate mini_gen_batch_size sequences per iteration
            pad_token_id=pad_token_id,
        )
        
        # Add custom response_format support
        self.response_format = getattr(args, 'response_format', None)
        self.beta = args.beta
        self.epsilon = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # Note: transformers 5.x removed `warnings_issued` from PreTrainedModel, so we guard the access.
        if not hasattr(model, "warnings_issued"):
            model.warnings_issued = {}
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # ======== [EasyR1 Optimization] Performance Configuration ========
        # 1. torch.compile to accelerate log_probs computation (ref: EasyR1 dp_actor.py)
        _use_torch_compile = getattr(args, 'use_torch_compile', False)
        if _use_torch_compile:
            self._log_probs_from_logits = torch.compile(log_probs_from_logits, dynamic=True)
            logger.info("[Optimization] Enabled torch.compile for log_probs_from_logits acceleration")
        else:
            self._log_probs_from_logits = log_probs_from_logits
        
        # 2. Freeze Vision Tower (ref: EasyR1 fsdp_workers.py freeze_vision_tower)
        # ViT parameters typically do not need updating during RL training. Freezing provides:
        #   - Reduced gradient computation (ViT backward skipped)
        #   - Reduced DeepSpeed communication (ViT params excluded from allreduce)
        #   - Lower memory usage (ViT does not retain activations for backward)
        _freeze_vision_tower = getattr(args, 'freeze_vision_tower', True)
        if _freeze_vision_tower:
            unwrapped = self.accelerator.unwrap_model(self.model)
            vision_module = None
            if hasattr(unwrapped, 'model') and hasattr(unwrapped.model, 'visual'):
                vision_module = unwrapped.model.visual
            elif hasattr(unwrapped, 'visual'):
                vision_module = unwrapped.visual
            
            if vision_module is not None:
                frozen_count = 0
                for param in vision_module.parameters():
                    param.requires_grad = False
                    frozen_count += 1
                logger.info(f"[Optimization] Froze Vision Tower: {frozen_count} parameters frozen")
            else:
                logger.warning("[Optimization] Vision Tower module not found, skipping freeze")
        
        # 3. Configure forward chunk_size (directly affects training speed)
        # Larger chunk_size means fewer forward passes, but higher peak memory for logits (chunk x seq_len x vocab_size)
        # Default 2 (safe); increase to 4 or 8 via --forward_chunk_size when memory permits
        self._forward_chunk_size = getattr(args, 'forward_chunk_size', 2)
        logger.info(f"[Optimization] forward_chunk_size = {self._forward_chunk_size} (samples per ref/train forward batch)")
        
        # 4. Log optimization status
        if FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
            logger.info("[Optimization] Flash-Attn Triton Cross Entropy enabled (~2x log_probs speedup)")
        else:
            logger.info("[Optimization] Flash-Attn Triton Cross Entropy unavailable, using F.cross_entropy fallback")

        # set re-sample dataloader
        # self._current_epoch_dataloader = None
        # self._current_epoch_iterator = None

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Custom GRPO evaluation with WandB Visual Logging.
        """
        # 1. Prepare Dataset and Dataloader
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            return {}

        dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.model
        model.eval()
        
        all_rewards = []
        
        # Lists for WandB visualization
        wandb_table_data = []
        # Limit number of visualization samples to avoid slowdown from uploading too much data
        num_visual_samples = 10 
        
        # Create a lightweight generation config
        eval_generation_config = copy.deepcopy(self.generation_config)
        eval_generation_config.num_return_sequences = 1
        # Use greedy decoding during eval to completely avoid CUDA assert caused by logits inf/nan during sampling
        # Qwen3.5 + DeepSpeed ZeRO-3 + bf16 with do_sample=True is prone to abnormal probability distributions
        eval_generation_config.do_sample = False
        eval_generation_config.temperature = 1.0  # temperature should be 1.0 or unset in greedy mode
        eval_generation_config.top_k = None
        eval_generation_config.top_p = None
        # [Qwen3.5 Speed Optimization] Limit max_new_tokens during eval
        # Qwen3.5 linear_attention (chunk_gated_delta_rule) needs to update recurrent state for each token
        # during autoregressive generation, much slower than standard transformer KV cache
        # Eval only needs reward assessment, not full-length output; 512 tokens is sufficient
        eval_max_new_tokens = min(512, self.max_completion_length)
        eval_generation_config.max_new_tokens = eval_max_new_tokens
        logger.info(f"[Eval] max_new_tokens limited to {eval_max_new_tokens} (training: {self.max_completion_length})")
        
        # Create logits processor to sanitize inf/nan, preventing CUDA device-side assert
        sanitize_logits_processor = LogitsProcessorList([SanitizeLogitsProcessor()])
        
        logger.info(f"***** Running {metric_key_prefix} *****")
        
        with torch.no_grad(), unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Move unwrap_model_for_generation outside the loop to avoid gather/partition for every eval batch
            # Qwen3.5 linear_attn (chunk_gated_delta_rule) under DeepSpeed ZeRO-3:
            # repeated gather/partition can cause CUDA illegal memory access
            unwrapped_model.eval()
            for step, inputs in tqdm(enumerate(dataloader), desc=f"{metric_key_prefix} loop"):
                # --- Step 1: Preprocessing ---
                prompts_raw = [x["prompt"] for x in inputs]
                
                # Resize images, considering text length
                for index_t in range(len(inputs)):
                    if 'image' in inputs[index_t] and isinstance(inputs[index_t]['image'], list):
                        # Calculate text length (question + expected answer)
                        # Prompt may be a list structure, need to extract text content
                        prompt_data = inputs[index_t].get("prompt", "")
                        prompt_text_len = 0
                        if isinstance(prompt_data, list):
                            for msg in prompt_data:
                                if isinstance(msg, dict) and 'content' in msg:
                                    content = msg['content']
                                    if isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get('type') == 'text' and item.get('text'):
                                                prompt_text_len += len(str(item['text']))
                                    elif isinstance(content, str):
                                        prompt_text_len += len(content)
                        else:
                            prompt_text_len = len(str(prompt_data))
                        
                        # Process solution similarly
                        solution_data = inputs[index_t].get("solution", "")
                        answer_text_len = 0
                        if isinstance(solution_data, list):
                            for msg in solution_data:
                                if isinstance(msg, dict) and 'content' in msg:
                                    content = msg['content']
                                    if isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get('type') == 'text' and item.get('text'):
                                                answer_text_len += len(str(item['text']))
                                    elif isinstance(content, str):
                                        answer_text_len += len(content)
                        else:
                            answer_text_len = len(str(solution_data)) if solution_data else 0
                        
                        text_len = prompt_text_len + answer_text_len
                        
                        inputs[index_t]['image'] = [
                            self.url2resize_pic(item['path'], len(inputs[index_t]["image"]), text_len) 
                            for item in inputs[index_t]["image"]
                        ]
                
                images = [x.get("image", []) for x in inputs]
                if len(images) > 0 and len(images[0]) == 0:
                    images = []

                prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

                prompt_inputs = self.processing_class(
                    text=prompts_text,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                    max_prompt_length=self.max_prompt_length
                )
                
                prompt_inputs = super()._prepare_inputs(prompt_inputs)
                
                # --- Step 2: Generation ---
                try:
                    generated_ids = unwrapped_model.generate(
                        **prompt_inputs, 
                        generation_config=eval_generation_config,
                        logits_processor=sanitize_logits_processor,
                    )
                    
                    prompt_length = prompt_inputs["input_ids"].size(1)
                    completion_ids = generated_ids[:, prompt_length:]
                    
                    completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                except RuntimeError as e:
                    # Catch CUDA errors
                    logger.warning(f"[Eval] Generation failed at step {step}, aborting eval loop. Error: {e}")
                    break
                
                # Save original text for WandB display
                completions_text = completions 

                # Format for reward functions
                if is_conversational(inputs[0]):
                    completions = [[{"role": "assistant", "content": completion}] for completion in completions]
                
                # --- Step 3: Reward Calculation ---
                device = self.accelerator.device
                batch_rewards = torch.zeros(len(inputs), len(self.reward_funcs), device=device)
                
                for i, (reward_func, reward_processing_class) in enumerate(zip(self.reward_funcs, self.reward_processing_classes)):
                    if isinstance(reward_func, PreTrainedModel):
                        if is_conversational(inputs[0]):
                            messages = [{"messages": p + c} for p, c in zip(prompts_raw, completions)]
                            texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                        else:
                            texts = [p + c for p, c in zip(prompts_text, completions)]
                            
                        reward_inputs = reward_processing_class(
                            texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                        )
                        reward_inputs = super()._prepare_inputs(reward_inputs)
                        batch_rewards[:, i] = reward_func(**reward_inputs).logits[:, 0]
                    else:
                        reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                        for key in reward_kwargs:
                            for example in inputs:
                                reward_kwargs[key].append(example[key])
                        
                        output_reward = reward_func(
                            prompts=prompts_raw, 
                            completions=completions, 
                            global_step=self.state.max_steps, 
                            max_steps=self.state.max_steps, 
                            input_raw_data=inputs, 
                            **reward_kwargs
                        )
                        batch_rewards[:, i] = torch.tensor(output_reward, dtype=torch.float32, device=device)

                total_rewards = batch_rewards.sum(dim=1)
                gathered_rewards = self.accelerator.gather_for_metrics(total_rewards)
                all_rewards.append(gathered_rewards.cpu())  # Move to CPU to reduce GPU memory usage

                # --- WandB Data Collection (main process only) ---
                if self.is_world_process_zero() and len(wandb_table_data) < num_visual_samples:
                    # Move Tensor back to CPU
                    cpu_rewards = total_rewards.detach().cpu().tolist()
                    
                    for idx in range(len(inputs)):
                        if len(wandb_table_data) >= num_visual_samples:
                            break
                        
                        # Get image object (if available)
                        img_obj = None
                        current_images = inputs[idx].get('image', [])
                        if current_images and len(current_images) > 0:
                            # Take the first image as an example, or you can use a list
                            # Note: images here are already PIL objects (resized in Step 1)
                            img_obj = wandb.Image(current_images[0], caption="Image 0") if is_wandb_available() else None
                        
                        # Get prompt text (simplified display)
                        prompt_str = str(prompts_raw[idx])
                        if len(prompt_str) > 500: prompt_str = prompt_str[:500] + "..."

                        wandb_table_data.append([
                            img_obj, 
                            prompt_str, 
                            completions_text[idx], 
                            cpu_rewards[idx]
                        ])
                
                # [Memory Optimization] Release intermediate tensors after each eval batch
                # Qwen3.5 linear_attention recurrent state accumulates significant memory during generation
                del prompt_inputs, generated_ids, completion_ids, batch_rewards, total_rewards
                torch.cuda.empty_cache()

        # --- Step 4: Metrics & Logging ---
        if len(all_rewards) == 0:
            return {}

        all_rewards = torch.cat(all_rewards)
        metrics = {
            f"{metric_key_prefix}/reward_mean": all_rewards.mean().item(),
            f"{metric_key_prefix}/reward_std": all_rewards.std().item(),
        }

        # Log scalar metrics (HuggingFace default logging)
        self.log(metrics)
        
        # This section may be unnecessary as upload happens automatically
        # Ensure upload only on main process with WandB active
        if self.is_world_process_zero() and is_wandb_available() and wandb.run is not None:
            if wandb_table_data:
                # Create WandB table
                table = wandb.Table(
                    columns=["Image", "Prompt", "Completion", "Reward"], 
                    data=wandb_table_data
                )
                # Log table, using global_step as x-axis
                # wandb.log({f"{metric_key_prefix}/samples_table": table}, step=self.state.global_step)
                logger.info(f"Logged {len(wandb_table_data)} samples to WandB table.")

        # [Critical Fix] Thoroughly clean GPU memory after eval
        # Qwen3.5 linear_attention recurrent state + generate KV cache accumulate
        # significant memory fragmentation during eval. Without cleanup, returning to training
        # _generate_and_score_completions with GatheredParameters will OOM, causing all batches to be skipped (loss=0)
        gc.collect()
        torch.cuda.empty_cache()
        # Reset DeepSpeed parameter coordinator trace to avoid trace mismatch during eval->train transition
        self._reset_ds_param_coordinator(self.model)
        
        self.model.train()
        return metrics
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _reset_ds_param_coordinator(self, model):
        """Reset DeepSpeed ZeRO-3 parameter coordinator trace state,
        allowing multiple forward passes within the same training step.
        DeepSpeed parameter coordinator traces the parameter fetch order of the first forward pass;
        subsequent forwards must match this trace. However, multiple independent forwards
        (e.g., per-sample loops) each start from the vision encoder and require trace reset for correct matching."""
        try:
            # Try multiple paths to access DeepSpeed parameter coordinator
            ds_engine = None
            if hasattr(model, 'optimizer') and hasattr(model.optimizer, 'parameter_offload'):
                ds_engine = model
            else:
                unwrapped = self.accelerator.unwrap_model(model)
                if hasattr(unwrapped, 'optimizer') and hasattr(unwrapped.optimizer, 'parameter_offload'):
                    ds_engine = unwrapped
            
            if ds_engine is not None:
                param_coordinator = ds_engine.optimizer.parameter_offload.get_param_coordinator()
                param_coordinator.reset_step()
        except Exception:
            # Silently ignore for non-ZeRO-3 scenarios or other exceptions
            pass

    def _deep_detach(self, obj):
        """Recursively detach all tensors in nested structures.
        
        Compatible with Qwen3.5 BaseModelOutputWithPooling (ModelOutput type).
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, tuple):
            return tuple(self._deep_detach(e) for e in obj)
        elif isinstance(obj, list):
            return [self._deep_detach(e) for e in obj]
        elif hasattr(obj, '__dataclass_fields__') or (hasattr(obj, 'keys') and hasattr(obj, '__class__') and hasattr(obj.__class__, '__dataclass_fields__')):
            # ModelOutput (BaseModelOutputWithPooling, etc.) is a dataclass + OrderedDict
            new_obj = obj.__class__(**{k: self._deep_detach(v) for k, v in obj.items()})
            return new_obj
        else:
            return obj

    def _deep_repeat(self, obj, n, list_mode="repeat_elements"):
        """Recursively repeat all tensors in nested structures along dim=0.
        
        Compatible with Qwen3.5 BaseModelOutputWithPooling (ModelOutput type).
        
        Args:
            obj: object to repeat
            n: number of repetitions
            list_mode: how to handle lists
                - "repeat_elements": recursively repeat each element (default, for per-layer features like deepstack_features)
                - "duplicate_list": duplicate the entire list n times (for per-image feature lists like pooler_output)
        """
        if isinstance(obj, torch.Tensor):
            return obj.repeat(n, *([1] * (obj.dim() - 1)))
        elif isinstance(obj, tuple):
            return tuple(self._deep_repeat(e, n) for e in obj)
        elif isinstance(obj, list):
            if list_mode == "duplicate_list":
                # Per-image feature list (e.g., Qwen3.5/Qwen3VL pooler_output = list[Tensor]):
                # Duplicate the entire list n times
                return obj * n
            else:
                # Per-layer feature list (e.g., Qwen3VL deepstack_features = list[Tensor]):
                # Recursively repeat each element along dim=0
                return [self._deep_repeat(e, n) for e in obj]
        elif hasattr(obj, '__dataclass_fields__') or (hasattr(obj, 'keys') and hasattr(obj, '__class__') and hasattr(obj.__class__, '__dataclass_fields__')):
            # ModelOutput (BaseModelOutputWithPooling / BaseModelOutputWithDeepstackFeatures, etc.)
            # Use different list processing modes for different fields:
            #   - pooler_output: per-image feature list -> duplicate_list
            #   - other list fields (e.g., deepstack_features): per-layer features -> repeat_elements
            new_fields = {}
            for k, v in obj.items():
                if k == "pooler_output" and isinstance(v, list):
                    new_fields[k] = self._deep_repeat(v, n, list_mode="duplicate_list")
                else:
                    new_fields[k] = self._deep_repeat(v, n, list_mode="repeat_elements")
            new_obj = obj.__class__(**new_fields)
            return new_obj
        else:
            return obj

    def _compute_vit_cache(self, model, pixel_values, image_grid_thw):
        """Compute ViT output in one pass and return cached result (matching get_image_features return format).
        
        Generic processing: recursively detach all nested tensors, compatible with different return formats of Qwen2VL/Qwen2.5VL/Qwen3VL.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        inner_model = unwrapped.model if hasattr(unwrapped, 'model') else unwrapped
        with torch.no_grad():
            result = inner_model.get_image_features(pixel_values, image_grid_thw)
        
        return self._deep_detach(result)


    def _repeat_vit_cache(self, vit_cache, n):
        """Repeat all tensors in vit_cache n times along dim=0.
        
        Generic processing: recursively repeat all nested tensors, compatible with any return format.
        """
        return self._deep_repeat(vit_cache, n)

    @contextmanager
    def _use_cached_image_features(self, model, vit_cache):
        """Context manager: temporarily replace the model's get_image_features method to return cached results.
        
        This keeps the model's full forward path intact (including get_placeholder_mask, get_rope_index,
        language_model, lm_head, etc.), only skipping the ViT computation.
        Fully compatible with DeepSpeed ZeRO-3 (all hooks fire as normal).
        
        Args:
            model: wrapped model (DeepSpeed engine) or unwrapped model
            vit_cache: cached result from _compute_vit_cache
        """
        unwrapped = self.accelerator.unwrap_model(model)
        # Get inner model (Qwen3VLModel / Qwen2VLModel)
        inner_model = unwrapped.model if hasattr(unwrapped, 'model') else unwrapped
        
        # Save original method
        original_get_image_features = inner_model.get_image_features
        
        # Define replacement method: ignore pixel_values and image_grid_thw, return cache directly
        # Use **kwargs for compatibility with Qwen3.5 extra params like return_dict
        def cached_get_image_features(pixel_values, image_grid_thw=None, **kwargs):
            return vit_cache
        
        try:
            inner_model.get_image_features = cached_get_image_features
            yield
        finally:
            inner_model.get_image_features = original_get_image_features

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, chunk_size=2,
                              vit_cache=None):
        """Compute per-token log probabilities via chunked forward passes to control peak memory.
        
        [EasyR1 Optimization] Uses flash_attn Triton cross entropy kernel instead of row-wise log_softmax+gather,
        ~2x speedup with inplace backward support for reduced memory.
        
        Args:
            input_ids: (B*G, seq_len) token ids for all generations
            attention_mask: (B*G, seq_len) 
            pixel_values: original single-copy image patches (not repeated), shape (total_patches, hidden_dim)
            image_grid_thw: original single-copy image grid info (not repeated), shape (num_images, 3)
            chunk_size: number of samples per forward pass, default 2 (memory-speed tradeoff)
            vit_cache: cached ViT output (from _compute_vit_cache),
                       if provided, skips ViT computation via monkey-patch
        """
        total_batch = input_ids.size(0)  # B * G
        all_per_token_logps = []
        
        for chunk_start in range(0, total_batch, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_batch)
            actual_chunk = chunk_end - chunk_start
            
            chunk_input_ids = input_ids[chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[chunk_start:chunk_end]
            
            # Repeat pixel_values and image_grid_thw within each chunk
            chunk_pixel_values = pixel_values.repeat(actual_chunk, 1)
            chunk_image_grid_thw = image_grid_thw.repeat(actual_chunk, 1)
            
            # Reset DeepSpeed parameter coordinator trace to allow multiple forward passes
            self._reset_ds_param_coordinator(model)
            
            try:
                if vit_cache is not None:
                    if actual_chunk > 1:
                        repeated_cache = self._repeat_vit_cache(vit_cache, actual_chunk)
                    else:
                        repeated_cache = vit_cache
                    with self._use_cached_image_features(model, repeated_cache):
                        logits = model(
                            chunk_input_ids, attention_mask=chunk_attention_mask,
                            pixel_values=chunk_pixel_values, image_grid_thw=chunk_image_grid_thw
                        ).logits
                    if actual_chunk > 1:
                        del repeated_cache
                else:
                    logits = model(
                        chunk_input_ids, attention_mask=chunk_attention_mask,
                        pixel_values=chunk_pixel_values, image_grid_thw=chunk_image_grid_thw
                    ).logits
            except ValueError as e:
                print(f"ValueError encountered in _get_per_token_logps chunk [{chunk_start}:{chunk_end}]: {e}")
                raise
            
            del chunk_pixel_values, chunk_image_grid_thw
            
            # [EasyR1 Optimization] Vectorized computation via log_probs_from_logits (Triton CE kernel or F.cross_entropy)
            # ~2x faster than row-wise log_softmax + gather, Triton version supports inplace backward for memory savings
            logits = logits[:, :-1, :]  # (actual_chunk, L-1, V)
            chunk_ids_shifted = chunk_input_ids[:, 1:]  # (actual_chunk, L-1)
            
            chunk_logps = self._log_probs_from_logits(logits, chunk_ids_shifted)  # (actual_chunk, L-1)
            all_per_token_logps.append(chunk_logps)
            
            del logits, chunk_input_ids, chunk_attention_mask, chunk_ids_shifted, chunk_logps
        
        return torch.cat(all_per_token_logps, dim=0)
    
    def get_local_batch_samples(self, epoch_iterator, num_batches):
        batch_samples = []
        num_items_in_batch = None
        end = False
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                end = True
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if self.args.average_tokens_across_devices and num_items_in_batch is not None:
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.item()

        return batch_samples, num_items_in_batch, end

    def get_train_dataloader(self):
        dataloader = super().get_train_dataloader()
        self._current_epoch_dataloader = dataloader
        try:
            self._current_epoch_iterator = iter(dataloader)
            logger.info("Captured current epoch dataloader and iterator for GPG-style resampling.")
        except Exception:
            self._current_epoch_iterator = None
            logger.warning("Failed to capture current epoch dataloader and iterator for GPG-style resampling.")
        return dataloader

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    def _create_empty_batch_result(self, batch_size, device):
        """Create an empty batch result for skipping problematic data."""
        # Create minimal dummy data to avoid training interruption
        dummy_seq_len = 10
        dummy_pixel_len = 1
        return {
            "rewards": torch.zeros(batch_size, device=device),
            "advantages": torch.zeros(batch_size, device=device),
            "rewards_per_func": torch.zeros(batch_size, len(self.reward_funcs), device=device),
            "std_grouped_rewards": torch.ones(batch_size, device=device),  # Avoid division by zero
            "n_valid_samples": 0,
            "num_reward_samples": batch_size,
            "prompt_mask": torch.ones(batch_size, dummy_seq_len, device=device, dtype=torch.long),
            "prompt_ids": torch.full((batch_size, dummy_seq_len), self.processing_class.pad_token_id, device=device, dtype=torch.long),
            "completion_mask": torch.ones(batch_size, dummy_seq_len, device=device, dtype=torch.long),
            "completion_ids": torch.full((batch_size, dummy_seq_len), self.processing_class.pad_token_id, device=device, dtype=torch.long),
            "pixel_values": torch.zeros(dummy_pixel_len, self._get_pixel_feature_dim(), device=device),  # Dynamically get pixel feature dim
            "image_grid_thw": torch.ones(batch_size, 3, device=device, dtype=torch.long),
            "inverse_alpha": 1.0,
        }

    def _get_pixel_feature_dim(self):
        """Dynamically get model pixel feature dimension, compatible with Qwen2-VL/Qwen2.5-VL/Qwen3-VL/Qwen3.5."""
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            config = unwrapped.config
            # Vision config for Qwen3.5/Qwen3-VL/Qwen2.5-VL/Qwen2-VL
            if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'hidden_size'):
                # Typically the last dim of pixel_values equals vision hidden_size * temporal_patch_size
                return config.vision_config.hidden_size * getattr(config.vision_config, 'temporal_patch_size', 2)
        except Exception:
            pass
        return 1176  # fallback: Qwen2-VL default

    def url2resize_pic(self, path, image_len, text_len=0):
        # Try to open image; on failure, use blank image and print path
        try:
            image = Image.open(path)
            original_width, original_height = image.size
            
            # Step 1: Calculate minimum scale factor (ensure at least one side >= 32 after scaling)
            min_scale = 1.0  # Default: no scaling
            min_width = 384
            if original_width > min_width or original_height > min_width:
                width_scale = min_width / original_width if original_width > min_width else 1.0
                height_scale = min_width / original_height if original_height > min_width else 1.0
                min_scale = max(width_scale, height_scale)
            
            # Step 2: Calculate max scale based on total pixels, considering text length
            # Longer text means less pixel budget for images
            # Set token budget based on model size: 2B/4B use 12000, 7B/8B use 8000
            model_name = self.model.config._name_or_path.lower() if hasattr(self.model.config, '_name_or_path') else ''
            if '2b' in model_name or '4b' in model_name:
                total_token_budget = 22000  # for ZeRO-3
            else:  # 7B/8B or other cases
                total_token_budget = 22000

            pixel_per_token = 32 * 32  # pixels per token
            # Text token budget (assuming ~1-2 tokens per character)
            text_token_budget = text_len / 2 if text_len > 0 else 0
            # Remaining token budget for images
            image_token_budget = total_token_budget - text_token_budget  # Reserve at least 30% budget for images
            # Adjusted pixel budget = remaining tokens * pixels per token
            adjusted_pixels = image_token_budget * pixel_per_token
            
            max_pixels = adjusted_pixels / image_len
            current_pixels = original_width * original_height
            if current_pixels > max_pixels:
                max_scale = (adjusted_pixels / image_len / current_pixels) ** 0.5
            else:
                max_scale = 1.0
            max_scale = min(max_scale, 1.0)  # Ensure no upscaling
            
            # Step 3: Generate random scale within [min_scale, max_scale]
            if min_scale > max_scale:
                scale_factor = max_scale  
            else:
                scale_factor = random.uniform(min_scale, max_scale) if min_scale < 1.0 else max_scale
            
            new_width = max(int(original_width * scale_factor), 32)
            new_height = max(int(original_height * scale_factor), 32)
            # Reduce print frequency
            if random.random() < 0.01:  # 1% chance to print
                print(f"min_scale, max_scale, image_len: {min_scale:.3f}, {max_scale:.3f}, {image_len} Resizing image from ({original_width}, {original_height}) to ({new_width}, {new_height}) with scale factor {scale_factor:.2f}")
            # Apply scaling
            if scale_factor != 1.0 or new_width == 32:
                image = image.resize((new_width, new_height), Image.BILINEAR)
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            return Image.new('RGB', (32, 32), color='white')
        return image
    def _generate_and_score_completions(
        self, model, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        max_resample_attempts = 20
        n_valid_samples = 0
        device = self.accelerator.device
        new_rewards_per_func = None
        new_rewards = None
        new_advantages = None
        new_prompt_ids = None
        new_prompt_mask = None
        new_completion_ids = None
        new_completion_mask = None

        # Pre-check whether to skip this batch
        if False:
            logger.warning("Pre-emptively skipping batch due to estimated high memory usage")
            return self._create_empty_batch_result(len(inputs), device)

        try:
            prompts = [x["prompt"] for x in inputs]
            prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
            if random.random() < 0.1:
                print("prompts_text:",prompts_text)
            # images = [x["image"] for x in inputs]
            for index_t in range(len(inputs)):
                if type(inputs[index_t]['image']) is list:
                    # Calculate text length (question + expected answer)
                    # Prompt may be a list structure, need to extract text content
                    prompt_data = inputs[index_t].get("prompt", "")
                    prompt_text_len = 0
                    if isinstance(prompt_data, list):
                        # Iterate through message list, extract all text-type content
                        for msg in prompt_data:
                            if isinstance(msg, dict) and 'content' in msg:
                                content = msg['content']
                                if isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and item.get('type') == 'text' and item.get('text'):
                                            prompt_text_len += len(str(item['text']))
                                elif isinstance(content, str):
                                    prompt_text_len += len(content)
                    else:
                        prompt_text_len = len(str(prompt_data))
                    
                    # Process solution similarly
                    solution_data = inputs[index_t].get("solution", "")
                    answer_text_len = 0
                    if isinstance(solution_data, list):
                        for msg in solution_data:
                            if isinstance(msg, dict) and 'content' in msg:
                                content = msg['content']
                                if isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and item.get('type') == 'text' and item.get('text'):
                                            answer_text_len += len(str(item['text']))
                                elif isinstance(content, str):
                                    answer_text_len += len(content)
                    else:
                        answer_text_len = len(str(solution_data)) if solution_data else 0
                    
                    text_len = prompt_text_len + answer_text_len
                    
                    inputs[index_t]['image']= [self.url2resize_pic(item['path'], len(inputs[index_t]["image"]), text_len) for item in inputs[index_t]["image"]]
                    # inputs[index_t]['image'] = inputs[index_t]['image'][:min(6, len(inputs[index_t]['image']))]
                    
                    # Calculate and output total token count
                    text_token_count = text_len / 2  # Assuming ~2 tokens per character
                    image_token_count = 0
                    for img in inputs[index_t]['image']:
                        if img is not None:
                            img_width, img_height = img.size
                            image_token_count += (img_width * img_height) // (32 * 32)
                    total_token_count = text_token_count + image_token_count

                    print(f"Token stats: text_tokens={text_token_count}, image_tokens={image_token_count}, total_tokens={total_token_count}, num_images={len(inputs[index_t]['image'])}")
            
            images = [x["image"] for x in inputs]  
            # print('len(inputs), len(images)', len(prompts_text), len(images[0]))
            if len(images[0]) ==0:
                images=[]

            # [<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1280x720 at 0x7F5C056BE9B0>]
            # Image.open([x["image"] for x in inputs][0])
            # [Image.open(item['path']) for item in [x["image"] for x in inputs][0]]
            # self.processing_class(text=prompts_text,images=[x["image"][0:1] for x in inputs],return_tensors="pt",padding=True,padding_side="left",                add_special_tokens=False,            )

            prompt_inputs = self.processing_class(
                text=prompts_text,
                images=images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
                # max_length=200000,
                max_prompt_length=self.max_prompt_length
            )
            
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids , prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
            pixel_values = prompt_inputs["pixel_values"]
            image_grid_thw = prompt_inputs["image_grid_thw"]
            
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                unwrapped_model.eval()
                prompt_length = prompt_ids.size(1)
                
                # Mini-batch generation: generate in multiple rounds, mini_gen_batch_size sequences per round
                num_mini_batches = self.num_generations // self.mini_gen_batch_size
                all_prompt_completion_ids = []
                
                # Create logits processor to sanitize inf/nan, preventing CUDA device-side assert under bf16
                train_sanitize_processor = LogitsProcessorList([SanitizeLogitsProcessor()])
                
                for mini_idx in range(num_mini_batches):
                    mini_completion_ids = unwrapped_model.generate(
                        **prompt_inputs, generation_config=self.generation_config,
                        logits_processor=train_sanitize_processor,
                    )
                    all_prompt_completion_ids.append(mini_completion_ids)
                
                # Concatenate all mini-batch results
                # Each generate output shape: (B * mini_gen_batch_size, P+C)
                # Need to rearrange so all generations for the same prompt are adjacent
                # Generate output order: [prompt0_gen0, prompt0_gen1, ..., prompt0_genM, prompt1_gen0, ...]
                # After multiple mini-batches, need to interleave-merge
                B = len(prompts)  # original batch size
                M = self.mini_gen_batch_size
                
                # Pad to same length first (different mini-batches may generate different sequence lengths)
                max_seq_len = max(ids.size(1) for ids in all_prompt_completion_ids)
                padded_ids = []
                for ids in all_prompt_completion_ids:
                    if ids.size(1) < max_seq_len:
                        pad_size = max_seq_len - ids.size(1)
                        ids = F.pad(ids, (0, pad_size), value=self.processing_class.pad_token_id)
                    padded_ids.append(ids)
                
                # Each mini-batch shape: (B*M, seq_len) -> reshape to (B, M, seq_len)
                # Then concatenate along M dim -> (B, G, seq_len) -> reshape to (B*G, seq_len)
                reshaped = [ids.view(B, M, -1) for ids in padded_ids]
                prompt_completion_ids = torch.cat(reshaped, dim=1).view(B * self.num_generations, -1)
                # Release intermediate variables from generation concatenation
                del all_prompt_completion_ids, padded_ids, reshaped
                
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
                prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
                unwrapped_model.train()

            # [EasyR1 Optimization] gc.collect + empty_cache to release KV cache fragments
            gc.collect()
            torch.cuda.empty_cache()
            
            # Mask everything after the first EOS token
            is_eos = completion_ids == self.processing_class.eos_token_id
            device = self.accelerator.device
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

            # Concatenate prompt_mask with completion_mask for logit computation
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
            # Keep original single-copy pixel_values/image_grid_thw (not repeated to B*G) to save memory
            # Each micro-batch in compute_loss repeats chunk_size copies
            pixel_values = prompt_inputs["pixel_values"]          # original B copies
            image_grid_thw = prompt_inputs["image_grid_thw"]      # original B copies
            
            # Release temporary variables from EOS mask computation + unneeded fields from prompt_inputs
            del is_eos, eos_idx, sequence_indices
            del prompt_inputs  # Release entire dict reference (pixel_values/image_grid_thw already extracted to independent vars)
            
            # Decode the generated completions
            completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                completions = [[{"role": "assistant", "content": completion}] for completion in completions]

            # Compute the rewards
            prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

            rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                if isinstance(reward_func, PreTrainedModel):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                    for key in reward_kwargs:
                        for example in inputs:
                            # Repeat each value in the column for `num_generations` times
                            reward_kwargs[key].extend([example[key]] * self.num_generations)
        
                    output_reward_func = reward_func(prompts=prompts, completions=completions,global_step=self.state.global_step,max_steps=self.state.max_steps,input_raw_data=inputs, **reward_kwargs)
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            # Sum the rewards from all reward functions
            rewards = rewards_per_func.sum(dim=1)

            print(f"[DEBUG rewards] rewards_per_func shape={rewards_per_func.shape}")
            for i, reward_func in enumerate(self.reward_funcs):
                rf_name = reward_func.__name__ if hasattr(reward_func, '__name__') else type(reward_func).__name__
                col = rewards_per_func[:, i]
                print(f"[DEBUG rewards] reward_func[{i}]={rf_name}: "
                      f"values={col.tolist()}, mean={col.mean().item():.4f}, std={col.std().item():.4f}")
            print(f"[DEBUG rewards] total rewards={rewards.tolist()}")
            print(f"[DEBUG rewards] rewards grouped (num_generations={self.num_generations}):")
            for g_idx, group in enumerate(rewards.view(-1, self.num_generations).tolist()):
                print(f"  group[{g_idx}]: {group}  mean={sum(group)/len(group):.4f}")

            # Release CPU variables no longer needed from reward computation
            del completions

            with torch.inference_mode():
                gathered_rewards_per_func = self.accelerator.gather(rewards_per_func)
                gathered_rewards = gathered_rewards_per_func.sum(dim=1)

                gathered_mean_grouped_rewards = gathered_rewards.view(-1, self.num_generations).mean(dim=1)
                gathered_std_grouped_rewards = gathered_rewards.view(-1, self.num_generations).std(dim=1)

                # Find groups with zero standard deviation
                identical_value_mask = gathered_std_grouped_rewards <= 1e-5
                easy_mask = gathered_mean_grouped_rewards >= len(self.reward_funcs) - 1e-5
                hard_mask = gathered_mean_grouped_rewards <= 1e-5

                # Count number of groups with zero standard deviation
                num_identical_reward_groups = identical_value_mask.sum().item()
                num_easy_problem = easy_mask.sum().item()
                num_hard_problem = hard_mask.sum().item()
                num_reward_samples = gathered_std_grouped_rewards.numel()

                n_valid_samples += num_reward_samples - num_identical_reward_groups

                print(f"[DEBUG n_valid] num_reward_samples={num_reward_samples}, "
                      f"num_identical_reward_groups={num_identical_reward_groups}, "
                      f"num_easy={num_easy_problem}, num_hard={num_hard_problem}, "
                      f"n_valid_samples={n_valid_samples}")
                print(f"[DEBUG n_valid] gathered_std_grouped_rewards={gathered_std_grouped_rewards.tolist()}")
                print(f"[DEBUG n_valid] gathered_mean_grouped_rewards={gathered_mean_grouped_rewards.tolist()}")
                print(f"[DEBUG n_valid] gathered_rewards (first 32)={gathered_rewards[:32].tolist()}")

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA OOM or RuntimeError in reward computation, skipping this batch: {e}")
            print(f"CUDA OOM or RuntimeError in reward computation, skipping this batch: {e}")
            # Clean GPU memory
            torch.cuda.empty_cache()
            
            # In distributed environment, need to notify other processes to skip this batch
            skip_flag = torch.tensor(1.0, device=device)
            try:
                # Try to synchronize skip status across all processes
                gathered_skip_flags = self.accelerator.gather(skip_flag)
            except Exception as sync_e:
                logger.warning(f"Failed to sync skip flag: {sync_e}")
            
            # Return an empty result for the caller to skip this batch
            return self._create_empty_batch_result(len(inputs), device)
        # ====== Compute advantages in original group order (ensure correct group structure) ======
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        if self.args.adjust_gd:
            advantages = (rewards - mean_grouped_rewards)
        else:
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        my_rewards_stds = rewards.view(-1, self.num_generations).std(dim=1)

        my_valid_value_mask = torch.where(my_rewards_stds > 0)[0]
        my_identical_value_mask = torch.where(my_rewards_stds == 0)[0]

        num_questions = len(prompts) // self.num_generations
        _b_valid = my_valid_value_mask.shape[0] * self.num_generations
        _b_ident = my_identical_value_mask.shape[0] * self.num_generations
        assert _b_valid + _b_ident == len(prompts)

        # prompt_completion_ids, attention_mask are constructed via concatenation
        # pixel_values, image_grid_thw are original B copies (not repeated to B*G)
        if _b_valid > 0:
            valid_rewards = rewards.reshape(num_questions, self.num_generations)[my_valid_value_mask].reshape(_b_valid)
            valid_advantages = advantages.reshape(num_questions, self.num_generations)[my_valid_value_mask].reshape(_b_valid)
            valid_prompt_ids = prompt_ids.reshape(num_questions, self.num_generations, -1)[my_valid_value_mask].reshape(_b_valid, -1)
            valid_prompt_mask = prompt_mask.reshape(num_questions, self.num_generations, -1)[my_valid_value_mask].reshape(_b_valid, -1)
            valid_completion_ids = completion_ids.reshape(num_questions, self.num_generations, -1)[
                my_valid_value_mask].reshape(_b_valid, -1)
            valid_completion_mask = completion_mask.reshape(num_questions, self.num_generations, -1)[
                my_valid_value_mask].reshape(_b_valid, -1)

            # rewards_per_func
            valid_rewards_per_func = rewards_per_func.reshape(num_questions, self.num_generations, -1)[
                my_valid_value_mask].reshape(_b_valid, -1)
        else:
            valid_rewards, valid_advantages, valid_prompt_ids, valid_prompt_mask, valid_completion_ids, valid_completion_mask, \
                valid_rewards_per_func = None, None, None, None, None, None, None
            
        if _b_ident > 0:
            identical_rewards = rewards.reshape(num_questions, self.num_generations)[
                my_identical_value_mask].reshape(_b_ident)
            identical_advantages = advantages.reshape(num_questions, self.num_generations)[
                my_identical_value_mask].reshape(_b_ident)
            identical_prompt_ids = prompt_ids.reshape(num_questions, self.num_generations, -1)[
                my_identical_value_mask].reshape(_b_ident, -1)
            identical_prompt_mask = prompt_mask.reshape(num_questions, self.num_generations, -1)[
                my_identical_value_mask].reshape(_b_ident, -1)
            identical_completion_ids = completion_ids.reshape(num_questions, self.num_generations, -1)[
                my_identical_value_mask].reshape(_b_ident, -1)
            identical_completion_mask = completion_mask.reshape(num_questions, self.num_generations, -1)[
                my_identical_value_mask].reshape(_b_ident, -1)
            
            identical_rewards_per_func = rewards_per_func.reshape(num_questions, self.num_generations, -1)[
                my_identical_value_mask].reshape(_b_ident, -1)
        else:
            identical_rewards, identical_advantages, identical_prompt_ids, identical_prompt_mask, identical_completion_ids, identical_completion_mask, \
                identical_rewards_per_func = None, None, None, None, None, None, None

        new_rewards = d_concat(new_rewards, valid_rewards)
        new_advantages = d_concat(new_advantages, valid_advantages)
        new_prompt_mask = d_concat_with_padding(new_prompt_mask, valid_prompt_mask, 0, left_pad=True)
        new_prompt_ids = d_concat_with_padding(new_prompt_ids, valid_prompt_ids, self.processing_class.pad_token_id, left_pad=True)
        new_completion_mask = d_concat_with_padding(new_completion_mask, valid_completion_mask, 0, left_pad=False)
        new_completion_ids = d_concat_with_padding(new_completion_ids, valid_completion_ids, self.processing_class.pad_token_id, left_pad=False)

        new_rewards_per_func = d_concat(new_rewards_per_func, valid_rewards_per_func)

        # Reassemble sample batch -- all fields concatenated in [valid, identical] order, maintaining alignment
        rewards = d_concat(new_rewards, identical_rewards)[:len(prompts)]
        advantages = d_concat(new_advantages, identical_advantages)[:len(prompts)]
        rewards_per_func = d_concat(new_rewards_per_func, identical_rewards_per_func)[:len(prompts)]
        prompt_ids = d_concat_with_padding(new_prompt_ids, identical_prompt_ids, self.processing_class.pad_token_id, left_pad=True)[:len(prompts)]
        prompt_mask = d_concat_with_padding(new_prompt_mask, identical_prompt_mask, 0, left_pad=True)[:len(prompts)]
        completion_ids = d_concat_with_padding(new_completion_ids, identical_completion_ids, self.processing_class.pad_token_id, left_pad=False)[:len(prompts)]
        completion_mask = d_concat_with_padding(new_completion_mask, identical_completion_mask, 0, left_pad=False)[:len(prompts)]

        # Recompute std from reassembled rewards (only for monitoring metrics)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        inverse_alpha = n_valid_samples / num_reward_samples
        inverse_alpha = min(1.0, inverse_alpha)

        # [EasyR1 Optimization] gc.collect + empty_cache combination
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "rewards": rewards,
            "advantages": advantages,
            "rewards_per_func": rewards_per_func,
            "std_grouped_rewards": std_grouped_rewards,
            "n_valid_samples": n_valid_samples,
            "num_reward_samples": num_reward_samples,
            "prompt_mask": prompt_mask,
            "prompt_ids": prompt_ids,
            "completion_mask": completion_mask,
            "completion_ids": completion_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "inverse_alpha": inverse_alpha,
        }


    # 
    def compute_loss(self, model, inputs_raw, return_outputs=False, num_items_in_batch=None):
        import copy
        for _ in range(2):
                if _ != 0:
                    # Use a dummy input for retry to avoid reusing corrupted data
                    inputs = copy.deepcopy(inputs_raw)
                else:
                    inputs=copy.deepcopy(inputs_raw)
                if return_outputs:
                    raise ValueError("The GRPOTrainer does not support returning outputs")
                # try:
                inputs = self._generate_and_score_completions(model, inputs)
                
                # Check if empty result was returned (batch skipped due to memory issues)
                if inputs["n_valid_samples"] == 0:
                    logger.warning("Skipping batch due to memory issues, returning zero loss")
                    # In distributed environment, need to ensure all processes skip synchronously
                    # Create a flag tensor to synchronize all processes
                    skip_flag = torch.tensor(1.0, device=self.accelerator.device)
                    # Synchronize skip status across all processes
                    gathered_skip_flags = self.accelerator.gather(skip_flag)
                    
                    # Return a minimal loss value to avoid training interruption
                    return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
                        
                # except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                #     logger.warning(f"CUDA OOM or RuntimeError in compute_loss, skipping this batch: {e}")
                #     # Clean GPU memory
                #     torch.cuda.empty_cache()
                    
                #     # In distributed environment, need to ensure all processes skip synchronously
                #     # Create a flag tensor to synchronize all processes
                #     skip_flag = torch.tensor(1.0, device=self.accelerator.device)
                #     # Synchronize skip status across all processes
                #     gathered_skip_flags = self.accelerator.gather(skip_flag)
                    
                #     # Return a minimal loss value to avoid training interruption
                #     return torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)
                
                prompt_completion_ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], dim=1)
                completion_mask = inputs["completion_mask"]
                attention_mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], dim=1)
                pixel_values = inputs["pixel_values"]
                image_grid_thw = inputs["image_grid_thw"]
                prompt_length = inputs["prompt_ids"].size(1)

                rewards = inputs["rewards"]
                advantages = inputs["advantages"]
                rewards_per_func = inputs["rewards_per_func"]
                std_grouped_rewards = inputs["std_grouped_rewards"]

                inverse_alpha = inputs["inverse_alpha"]

                n_valid_samples = inputs["n_valid_samples"]
                num_reward_samples = inputs["num_reward_samples"]

                # current_resample_attempt = inputs["current_resample_attempt"]

                # ======== [EasyR1-style] Micro-batch forward+backward gradient accumulation ========
                # Reference: EasyR1 update_policy: each micro-batch does forward -> loss -> backward
                # training_step then skips default backward (gradients already accumulated)
                # Memory safe: only forward chunk_size samples at a time, peak logits = chunk_size x L x V
                
                train_chunk_size = self._forward_chunk_size  # Configurable, default 4 (halves forward count vs hardcoded 2)
                total_samples = prompt_completion_ids.size(0)  # B * G
                accumulated_loss = torch.tensor(0.0, device=prompt_completion_ids.device)
                
                # ======== Ref model forward (chunked, ViT cache accelerated) ========
                if not self._need_ref_model:
                    _need_ref_forward = False
                elif self.args.pg_name == 'gpg':
                    _need_ref_forward = self.state.global_step % self._monitor_kl_interval == 0
                else:
                    _need_ref_forward = True
                
                # [EasyR1 Optimization] gc.collect + empty_cache combination, more thorough than empty_cache alone
                gc.collect()
                torch.cuda.empty_cache()
                
                # [EasyR1 Optimization] Ref model and Train model share ViT cache
                # When ref_model is an independent model, image features do not depend on trainable params,
                # only 1 ViT forward (computed with model) is needed, reused by both ref and train
                with torch.no_grad():
                    shared_vit_cache = self._compute_vit_cache(model, pixel_values, image_grid_thw)
                
                if _need_ref_forward:
                    with torch.inference_mode():
                        # Chunked forward for ref model (reusing shared ViT cache)
                        # ref forward and train forward use same chunk_size (logits are the memory bottleneck)
                        if self.ref_model is not None:
                            ref_per_token_logps = self._get_per_token_logps(
                                self.ref_model, prompt_completion_ids, attention_mask, 
                                pixel_values, image_grid_thw, chunk_size=train_chunk_size,
                                vit_cache=shared_vit_cache)
                        else:
                            with self.accelerator.unwrap_model(model).disable_adapter():
                                ref_per_token_logps = self._get_per_token_logps(
                                    model, prompt_completion_ids, attention_mask, 
                                    pixel_values, image_grid_thw, chunk_size=train_chunk_size,
                                    vit_cache=shared_vit_cache)
                    ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
                else:
                    ref_per_token_logps = None
                
                # model_vit_cache directly reuses shared_vit_cache (no 2nd ViT forward needed)
                model_vit_cache = shared_vit_cache
                
                # [EasyR1 Optimization] Cleanup after ref forward
                gc.collect()
                torch.cuda.empty_cache()
                
                # ======== [GRPO clipping] Cache old_per_token_logps ========
                # Only executed in GRPO mode: first do a no-grad forward to snapshot current policy logps
                # In subsequent gradient-enabled forward, ratio = pi_theta / pi_theta_old can truly deviate from 1.0
                # GPG and Pinsker are unaffected
                old_per_token_logps = None
                if self.args.pg_name == 'grpo':
                    with torch.no_grad():
                        old_per_token_logps = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask,
                            pixel_values, image_grid_thw, chunk_size=train_chunk_size,
                            vit_cache=shared_vit_cache)
                        old_per_token_logps = old_per_token_logps[:, prompt_length - 1:]
                    gc.collect()
                    torch.cuda.empty_cache()

                # ======== Micro-batch forward (with gradients) + backward ========
                all_per_token_kl = []
                all_pinsker_loss = []
                all_detached_logps = []
                
                for chunk_start in range(0, total_samples, train_chunk_size):
                    chunk_end = min(chunk_start + train_chunk_size, total_samples)
                    actual_chunk = chunk_end - chunk_start
                    
                    # Get micro-batch
                    chunk_ids = prompt_completion_ids[chunk_start:chunk_end]
                    chunk_attn = attention_mask[chunk_start:chunk_end]
                    chunk_pixel_values = pixel_values.repeat(actual_chunk, 1)
                    chunk_image_grid_thw = image_grid_thw.repeat(actual_chunk, 1)
                    
                    # Reset DeepSpeed parameter coordinator
                    self._reset_ds_param_coordinator(model)
                    
                    # [ViT cache] Forward entire micro-batch, reusing ViT cache
                    if actual_chunk > 1:
                        repeated_cache = self._repeat_vit_cache(model_vit_cache, actual_chunk)
                    else:
                        repeated_cache = model_vit_cache
                    with self._use_cached_image_features(model, repeated_cache):
                        logits = model(
                            chunk_ids, attention_mask=chunk_attn,
                            pixel_values=chunk_pixel_values, image_grid_thw=chunk_image_grid_thw
                        ).logits  # (C, L, V)
                    if actual_chunk > 1:
                        del repeated_cache
                    del chunk_pixel_values, chunk_image_grid_thw
                    
                    # [EasyR1 Optimization] logits -> per-token log probs (vectorized Triton CE kernel, ~2x faster than row-wise)
                    logits = logits[:, :-1, :]
                    shifted_ids = chunk_ids[:, 1:]
                    
                    chunk_logps = self._log_probs_from_logits(logits, shifted_ids)  # (C, L-1)
                    del logits
                    
                    # Extract completion portion
                    chunk_comp_logps = chunk_logps[:, prompt_length - 1:]  # (C, C_len)
                    chunk_comp_mask = completion_mask[chunk_start:chunk_end]
                    chunk_advs = advantages[chunk_start:chunk_end]
                    del chunk_logps
                    
                    # ======== Loss Computation ========
                    if self.args.pg_name == 'gpg':
                        # GPG: loss = -logps * advantages (vectorized)
                        all_detached_logps.append(chunk_comp_logps.detach())
                        per_token_loss = -chunk_comp_logps * chunk_advs.unsqueeze(1)
                        per_sample_loss = (per_token_loss * chunk_comp_mask).sum(dim=1) / chunk_comp_mask.sum(dim=1)
                        chunk_loss = per_sample_loss.sum() / total_samples
                        if getattr(self.args, 'gpg_alpha_rescale', False):
                            chunk_loss = chunk_loss / (inverse_alpha + 1e-5)
                        del per_token_loss, per_sample_loss
                        
                    elif self.args.pg_name == 'grpo':
                        chunk_old_logps = old_per_token_logps[chunk_start:chunk_end]
                        if ref_per_token_logps is not None:
                            chunk_ref_logps = ref_per_token_logps[chunk_start:chunk_end]
                            chunk_kl = torch.exp(chunk_ref_logps - chunk_comp_logps) - (chunk_ref_logps - chunk_comp_logps) - 1
                        else:  # beta == 0: no reference model is loaded and the KL term vanishes
                            chunk_ref_logps = None
                            chunk_kl = torch.zeros_like(chunk_comp_logps)
                        # importance sampling ratio: rho = pi_theta / pi_theta_old (old from cached no-grad forward)
                        log_ratio = chunk_comp_logps - chunk_old_logps
                        ratio = torch.exp(log_ratio)
                        # PPO clipping: min(rho*A, clip(rho, 1-eps, 1+eps)*A)
                        clipped_ratio = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon_high)
                        per_token_loss1 = ratio * chunk_advs.unsqueeze(1)
                        per_token_loss2 = clipped_ratio * chunk_advs.unsqueeze(1)
                        per_token_loss = -torch.min(per_token_loss1, per_token_loss2) + self.beta * chunk_kl
                        per_sample_loss = (per_token_loss * chunk_comp_mask).sum(dim=1) / chunk_comp_mask.sum(dim=1)
                        chunk_loss = per_sample_loss.sum() / total_samples
                        if getattr(self.args, 'gpg_alpha_rescale', False):
                            chunk_loss = chunk_loss / (inverse_alpha + 1e-5)
                        all_per_token_kl.append((chunk_kl * chunk_comp_mask).sum(dim=1) / chunk_comp_mask.sum(dim=1))
                        del chunk_ref_logps, chunk_old_logps, chunk_kl, log_ratio, ratio, clipped_ratio, per_token_loss1, per_token_loss2, per_token_loss, per_sample_loss
                        
                    elif self.args.pg_name == 'pinsker':
                        chunk_ref_logps = ref_per_token_logps[chunk_start:chunk_end]
                        chunk_pinsker = torch.abs(chunk_ref_logps - chunk_comp_logps) * chunk_comp_mask
                        chunk_pinsker_loss = (chunk_pinsker.sum(dim=1) ** 2.0) / 2.0 / chunk_comp_mask.sum(dim=1)
                        all_pinsker_loss.append(chunk_pinsker_loss.detach())
                        reward_loss = torch.exp(chunk_comp_logps - chunk_comp_logps.detach()) * chunk_advs.unsqueeze(1)
                        reward_loss = (reward_loss * chunk_comp_mask).sum(dim=1) / chunk_comp_mask.sum(dim=1)
                        per_sample_loss = -(reward_loss - self.beta * chunk_pinsker_loss)
                        chunk_loss = per_sample_loss.sum() / total_samples
                        if getattr(self.args, 'gpg_alpha_rescale', False):
                            chunk_loss = chunk_loss / (inverse_alpha + 1e-5)
                        del chunk_ref_logps, chunk_pinsker, reward_loss, per_sample_loss
                    else:
                        raise ValueError(f"Unknown policy gradient name: {self.args.pg_name}")
                    
                    del chunk_comp_logps, chunk_comp_mask, chunk_advs
                    
                    # [EasyR1-style] Immediate backward within micro-batch, gradient accumulation
                    self.accelerator.backward(chunk_loss)
                    accumulated_loss += chunk_loss.detach()
                    
                    del chunk_loss, chunk_ids, chunk_attn, shifted_ids
                
                # Release ViT cache and old_logps cache
                del shared_vit_cache, model_vit_cache
                if old_per_token_logps is not None:
                    del old_per_token_logps
                gc.collect()
                torch.cuda.empty_cache()
                
                # ======== Metrics Logging ========
                completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
                self._metrics["completion_length"].append(completion_length)

                reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
                for i, reward_func in enumerate(self.reward_funcs):
                    if isinstance(reward_func, PreTrainedModel):
                        reward_func_name = reward_func.config._name_or_path.split("/")[-1]
                    else:
                        reward_func_name = reward_func.__name__
                    self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

                self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
                self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
                self._metrics['n_valid_samples'].append(n_valid_samples)
                self._metrics['n_reward_samples'].append(num_reward_samples)
                self._metrics['inverse_alpha'].append(inverse_alpha)

                if self.args.pg_name == 'grpo':
                    mean_kl = torch.cat(all_per_token_kl).mean()
                    self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
                elif self.args.pg_name == 'pinsker':
                    mean_pinsker = torch.cat(all_pinsker_loss).mean()
                    self._metrics["pinsker"].append(self.accelerator.gather_for_metrics(mean_pinsker).mean().item())
                elif self.args.pg_name == 'gpg':
                    if ref_per_token_logps is not None:
                        no_grad_logps = torch.cat(all_detached_logps, dim=0)
                        per_token_kl = torch.exp(ref_per_token_logps - no_grad_logps) - (ref_per_token_logps - no_grad_logps) - 1
                        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                        self._metrics["kl_gpg"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

                # Gradients already accumulated in micro-batch loop, return detached loss
                return accumulated_loss.detach().requires_grad_(True)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training_step to skip the default backward call.
        
        compute_loss already completes gradient accumulation via micro-batch forward + immediate backward,
        so the trainer does not need to call accelerator.backward() on the returned loss.
        """
        model.train()
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        if loss is not None:
            del inputs
            return loss.detach()
        return torch.tensor(0.0, device=self.accelerator.device)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


def d_concat(old_d, new_d):
    if new_d is None:
        return old_d
    if old_d is None:
        return new_d
    
    return torch.concat([old_d, new_d])


def d_concat_with_padding(old_d, new_d, pad_token_id, left_pad=False):
    if new_d is None:
        return old_d
    if old_d is None:
        return new_d
    
    if new_d.shape[1] < old_d.shape[1]:
        new_d = pad_sequence_to_length(new_d, old_d.shape[1], pad_token_id, left_pad)
    else:
        old_d = pad_sequence_to_length(old_d, new_d.shape[1], pad_token_id, left_pad)
    
    return torch.concat([old_d, new_d])

def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)
