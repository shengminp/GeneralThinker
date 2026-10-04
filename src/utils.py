import re
import os
import torch
import argparse
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
from latex2sympy2_extended.latex2sympy2 import NormalizationConfig

#### utils for DPP setting ####
def is_dist_env():
    return os.environ.get("WORLD_SIZE", "1") != "1"

def get_rank():
    if is_dist_env() and dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def set_dist_env():
    use_dist = is_dist_env()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif torch.cuda.is_available():
        device = torch.device("cuda", 0)
        backend = "nccl"
    
    if use_dist:
        dist.init_process_group(backend=backend)
#### utils for DPP setting ####

#### utils for parser ####
def get_base_parser():
    parser = argparse.ArgumentParser(description="Configuration for Finetuning")

    parser.add_argument('--base_model', type=str, required=True, help='Base model name')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='Checkpoint name')
    parser.add_argument('--data_name', type=str, required=True, help='Data name')
    parser.add_argument('--reward_type', type=str, required=False, help='Reward type')
    
    parser.add_argument('--modulation', action='store_true', help='Modulation or not')
    parser.add_argument('--controlled_modulation', action='store_true', help='Controlled modulation or not')
    parser.add_argument('--modulation_lambda', type=float, help='Modulation shaping value')

    parser.add_argument('--perturbation', type=str, choices=['shuffle', 'reverse'], help='Token-signal perturbation analysis')
    parser.add_argument('--top_entropy_quantile', type=float, default=1.0, help='top entropy quantile value')
    parser.add_argument('--per_gpu_batch_size', type=int, required=True, help='Batch size per GPU')
    return parser

def add_training_args(parser):
    parser.add_argument('--lora_r', type=int, required=True, help='LoRA rank')
    parser.add_argument('--learning_rate', type=float, required=True, help='Learning rate')
    parser.add_argument('--num_train_epochs', type=float, required=True, help='Number of epochs for training')
    parser.add_argument('--grad_accum', type=int, default=1, help='Steps of gradient accumulation')
    return parser

def add_generation_args(parser):
    parser.add_argument('--baseline', type=str, default=None, help='Name of baseline')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Baseline path')
    parser.add_argument('--analysis', action='store_true', help='Analysis or not')
    parser.add_argument('--with_prompt', type=str, required=True, help='If generate with prompt')
    parser.add_argument('--generation_type', type=str, required=True, help='Generation type')
    parser.add_argument('--generation_times', type=int, default=1, help='Generation times')
    return parser

def parse_args(use_training_args=False):
    parser = get_base_parser()
    if use_training_args:
        parser = add_training_args(parser)
    else:
        parser = add_generation_args(parser)
    return parser.parse_args()
#### utils for parser ####

#### utils for likelihood reward ####
def likelihood_reward(
    prompts,
    completions,
    is_truncated,
    processing_class,
    prepare_inputs_func,
    model,
    **kwargs
):
    likelihood_rewards = [-30.0] * len(prompts)
    valid_indices = []
    prompt_reasoning_texts = []
    prompt_reasoning_gt_texts = []

    for i, (p, c, gt) in enumerate(zip(prompts, completions, kwargs['answer'], strict=True)):
        if is_truncated[i] is True:
            continue

        marker = "\\boxed{"
        start = c.rfind(marker)
        if start == -1:
            continue
        
        depth = 1
        end = None

        for index in range(start + len(marker), len(c)):
            if c[index] == "{":
                depth += 1
            elif c[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index
                    break
        
        if end is None:
            continue

        generated_answer = c[start + len(marker):end].strip()
        if generated_answer == "":
            continue
    
        reasoning_text = c[:start].strip()
        if reasoning_text == "":
            continue

        prompt_reasoning_text = p + c[:start] + marker
        prompt_reasoning_gt_text = prompt_reasoning_text + gt + "}"

        valid_indices.append(i)
        prompt_reasoning_texts.append(prompt_reasoning_text)
        prompt_reasoning_gt_texts.append(prompt_reasoning_gt_text)
    
    if not valid_indices:
        return {
            "rewards": likelihood_rewards,
            "nll_input_ids": None,
            "nll_attention_mask": None,
            "nll_answer_mask": None,
            "nll_original_indices": [],
        }
    
    with torch.inference_mode():
        tmp_inputs = processing_class(
            text=prompt_reasoning_gt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        input_ids = tmp_inputs["input_ids"]
        attention_mask = tmp_inputs["attention_mask"]
        
        full_len = input_ids.shape[1]
        answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, prompt_reasoning in enumerate(prompt_reasoning_texts):
            prompt_reasoning_ids = processing_class(
                text=prompt_reasoning,
                return_tensors="pt",
                padding=False,
                add_special_tokens=False,
            )["input_ids"][0]

            actual_len = int(attention_mask[i].sum().item())
            len_prompt_reasoning = prompt_reasoning_ids.shape[0]

            pad_len = full_len - actual_len
            start_gt = pad_len + len_prompt_reasoning
            end_gt = pad_len + actual_len
            answer_mask[i, start_gt:end_gt] = True

        # Forward and calculate the log-likelihood reward
        reward_inputs = prepare_inputs_func({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        logits = model(**reward_inputs).logits

        shift_logits = logits[:, :-1, :]
        shift_labels = reward_inputs["input_ids"][:, 1:]
        shift_answer_mask = answer_mask[:, 1:].to(shift_labels.device)

        logprobs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = logprobs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        masked_token_logprobs = token_logprobs.masked_fill(~shift_answer_mask, 0.0)
        gt_token_logprobs = masked_token_logprobs.sum(dim=-1)

        for row_idx, original_idx in enumerate(valid_indices):
            likelihood_rewards[original_idx] = gt_token_logprobs[row_idx].item()

    return {
        "rewards": likelihood_rewards,
        "nll_input_ids": input_ids,
        "nll_attention_mask": attention_mask,
        "nll_answer_mask": answer_mask,
        "nll_original_indices": valid_indices,
    }
#### utils for likelihood reward ####

#### utils for compare ####

def extract_boxed(text: str) -> str | None:
    if text is None:
        return None

    marker = "\\boxed{"
    start = text.lower().rfind(marker)
    if start == -1:
        return None

    depth = 1
    result = []

    for index in range(start + len(marker), len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return "".join(result).strip()
        result.append(text[index])

    return None
    
def compare_pred_label(pred, label, dataset_type, label_context=None):
    label = label.strip()
    label_context = str(label_context).strip() if label_context is not None else None
    pred_segment = extract_boxed(pred)
    if pred_segment is None:
        return False

    if dataset_type == "math":
        label_expr = f"${label}$"
        pred_expr = f"${pred_segment}$"

        strict_latex_config = LatexExtractionConfig(
            boxed_match_priority=0,
            normalization_config=NormalizationConfig(
                basic_latex=True,
                units=True,
                malformed_operators=False,
                nits=False,
                boxed="all",
                equations=False,
            ),
        )
        
        label_parsed = parse(
            label_expr,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            fallback_mode="no_fallback",
        )

        pred_parsed = parse(
            pred_expr,
            extraction_config=[strict_latex_config, ExprExtractionConfig()],
            fallback_mode="no_fallback",
        )

        if not label_parsed or not pred_parsed:
            return False
           
        return verify(
            label_parsed,
            pred_parsed,
            float_rounding=6,
            numeric_precision=15,
            strict=True,
        )
                
    elif dataset_type == "mcq":
        pred_letter = None
        label_letter = None

        pred_letter_match = (
            re.match(r"^\(?\s*([A-J])\s*\)?\s*$", pred_segment, flags=re.IGNORECASE)
            or re.match(r"^\(?\s*([A-J])\s*[\.\):：\-]\s+.+$", pred_segment, flags=re.IGNORECASE)
            or re.match(
                r"^\s*(?:option|answer|final answer|the answer is|correct answer is)\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?:\s*[\.\):：\-]\s*.*)?$",
                pred_segment,
                flags=re.IGNORECASE,
            )
        )
        if pred_letter_match:
            pred_letter = pred_letter_match.group(1).upper()

        label_letter_match = (
            re.match(r"^\(?\s*([A-J])\s*\)?\s*$", label, flags=re.IGNORECASE)
            or re.match(r"^\(?\s*([A-J])\s*[\.\):：\-]\s+.+$", label, flags=re.IGNORECASE)
            or re.match(
                r"^\s*(?:option|answer|final answer|the answer is|correct answer is)\s*[:：]?\s*\(?\s*([A-J])\s*\)?(?:\s*[\.\):：\-]\s*.*)?$",
                label,
                flags=re.IGNORECASE,
            )
        )
        if label_letter_match:
            label_letter = label_letter_match.group(1).upper()

        if pred_letter is not None and label_letter is not None:
            return pred_letter == label_letter

        pred_text = re.sub(
            r"^\s*(?:option|answer|final answer|the answer is|correct answer is)?\s*[:：]?\s*\(?\s*[A-J]\s*\)?\s*[\.\):：\-]?\s*",
            "",
            pred_segment,
            flags=re.IGNORECASE,
        ).strip()

        label_text = re.sub(
            r"^\s*(?:option|answer|final answer|the answer is|correct answer is)?\s*[:：]?\s*\(?\s*[A-J]\s*\)?\s*[\.\):：\-]?\s*",
            "",
            label,
            flags=re.IGNORECASE,
        ).strip()

        pred_text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", pred_text)
        label_text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", label_text)

        pred_text = pred_text.replace("$", "").replace("{", "").replace("}", "")
        label_text = label_text.replace("$", "").replace("{", "").replace("}", "")

        pred_text = re.sub(r"\s+", " ", pred_text).strip().strip(" \n\t\r.:;!?'\"`~").lower()
        label_text = re.sub(r"\s+", " ", label_text).strip().strip(" \n\t\r.:;!?'\"`~").lower()

        context_text = None
        if label_context is not None:
            context_text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", label_context)
            context_text = context_text.replace("$", "").replace("{", "").replace("}", "")
            context_text = re.sub(r"\s+", " ", context_text).strip().strip(" \n\t\r.:;!?'\"`~").lower()

        if context_text is not None and pred_text == context_text:
            return True
        if label_text and pred_text == label_text:
            return True

        return False
    return False
#### utils for compare ####

#### utils for binary reward ####
def binary_reward(
    prompts,
    completions,
    is_truncated,
    processing_class,
    prepare_inputs_func,
    model,
    **kwargs
):
    binary_rewards = [0.0] * len(prompts)

    for i, (p, c, gt) in enumerate(zip(prompts, completions, kwargs['answer'], strict=True)):
        if is_truncated[i] is True:
            continue
        
        marker = "\\boxed{"
        start = c.rfind(marker)
        if start == -1:
            continue
        
        reasoning_text = c[:start].strip()
        if reasoning_text == "":
            continue

        question_type = kwargs['type'][i]
        answer_context = kwargs['answer_context'][i] if question_type == "mcq" else None
        compare_same = compare_pred_label(c, gt, question_type, answer_context)
        if compare_same:
            binary_rewards[i] = 1.0

    return binary_rewards
#### utils for binary reward ####

#### utils for compute metric ####
def compute_metric_text(total_preds, dataset, tokenizer):
    pred_ids = np.where(total_preds.predictions != -100, total_preds.predictions, tokenizer.pad_token_id)
    label_ids = np.where(total_preds.label_ids != -100, total_preds.label_ids, tokenizer.pad_token_id)

    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    results = []
    for i, (pred, label) in enumerate(zip(preds, labels)):
        pred = str(pred).strip()
        label = str(label).strip()
        question_type = dataset[i]['type']
        answer_context = dataset[i]['answer_context'] if question_type == "mcq" else None
        compare_same = compare_pred_label(pred, label, question_type, answer_context)
        if compare_same:
            results.append(1.0)
        else:
            results.append(0.0)
        
    return {'accuracy': sum(results) / len(results)}
#### utils for compute metric ####


def pred2text(total_preds, config, tokenizer):
    pred_ids = np.where(total_preds.predictions != -100, total_preds.predictions, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    predictions = []
    for pred in preds:
        predictions.append(pred)

    return {'prediction': predictions}