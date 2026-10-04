import re
import pandas as pd
from datasets import load_dataset

USER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
USER_SELF_INSTRUCTION = "Please reason step by step given the final answer within \\boxed{}."

def format_query(
    model_type,
    tokenizer,
    system_context,
    prompt_context,
    query,
    user_context,
    assis_context
):
    if system_context:
        message = [{"role": "system", "content": system_context}] + prompt_context + [{"role": "user", "content": query}]
    else:
        if user_context:
            message = prompt_context + [{"role": "user", "content": f"{query}\n{user_context}"}]
        else:
            message = prompt_context + [{"role": "user", "content": query}]

    if "Qwen3" in model_type:
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )

    if assis_context:
        return prompt + assis_context
    else:
        return prompt


def remove_system(model_type, prompt):
    if "Qwen" in model_type:
        removed_prompt = re.sub(
            r'<\|im_start\|>system[\s\S]*?(?=<\|im_start\|>user)',
            '',
            prompt
        )
    elif "Llama" in model_type:
        removed_prompt = re.sub(
            r'<\|begin_of_text\|><\|start_header_id\|>system[\s\S]*?(?=<\|eot_id\|><\|start_header_id\|>user)',
            '',
            prompt
        )
        removed_prompt = re.sub(r"<\|eot_id\|>", "<|begin_of_text|>", removed_prompt, count=1)
    return removed_prompt


def prepare_generation_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for generation.
    """
    def _preprocess_func(dataset):
        formatted_reasoning_prompts = []
        formatted_self_teacher_prompts = []
        
        for question, answer in zip(dataset['question'], dataset['answer']):
            reasoning_prompt = format_query(
                model_type=config.base_model, 
                tokenizer=tokenizer,
                system_context=None,
                prompt_context=prompt_context, 
                query=question,
                user_context=USER_INSTRUCTION,
                assis_context=None
            )
            reasoning_prompt = remove_system(config.base_model, reasoning_prompt)
            formatted_reasoning_prompts.append(f"{reasoning_prompt}")

            self_teacher_prompt = format_query(
                model_type=config.base_model, 
                tokenizer=tokenizer,
                system_context=None,
                prompt_context=prompt_context, 
                query=question,
                user_context=USER_SELF_INSTRUCTION,
                assis_context="\\boxed{" + f"{answer}" + "}\n"
            )
            self_teacher_prompt = remove_system(config.base_model, self_teacher_prompt)
            formatted_self_teacher_prompts.append(f"{self_teacher_prompt}")


        dataset['prompt'] = formatted_reasoning_prompts
        dataset['self_teacher_prompt'] = formatted_self_teacher_prompts
        return dataset
    
    raw_datasets = {
        split: load_dataset("json", data_files=path, split='train')
        for split, path in config.dataset_path.items()
    }

    prompt_context = []
    if config.with_prompt == "true":
        prompt_df = pd.read_json(config.prompt_path)
        for _, data in prompt_df.iterrows():
            prompt_context.append({"role": "user", "content": data['question']})
            prompt_context.append({"role": "assistant", "content": data['answer']})

    with training_args.main_process_first():
        tokenized_datasets = {}
        for split, dataset in raw_datasets.items():
            tokenized_dataset = dataset.map(_preprocess_func, batched=True, num_proc=4)
            tokenized_datasets[split] = tokenized_dataset

    return tokenized_datasets


def prepare_training_data(config, tokenizer, training_args):
    """
    Load and prepare dataset for training.
    """ 
    def _preprocess_func(dataset):
        formatted_reasoning_prompts = []
        formatted_self_teacher_prompts = []
        
        for question, answer in zip(dataset['question'], dataset['answer']):
            reasoning_prompt = format_query(
                model_type=config.base_model, 
                tokenizer=tokenizer,
                system_context=None,
                prompt_context=[], 
                query=question,
                user_context=USER_INSTRUCTION,
                assis_context=None
            )
            reasoning_prompt = remove_system(config.base_model, reasoning_prompt)
            formatted_reasoning_prompts.append(f"{reasoning_prompt}")

            self_teacher_prompt = format_query(
                model_type=config.base_model, 
                tokenizer=tokenizer,
                system_context=None,
                prompt_context=[], 
                query=question,
                user_context=USER_SELF_INSTRUCTION,
                assis_context="\\boxed{" + f"{answer}" + "}\n"
            )
            self_teacher_prompt = remove_system(config.base_model, self_teacher_prompt)
            formatted_self_teacher_prompts.append(f"{self_teacher_prompt}")

        dataset['prompt'] = formatted_reasoning_prompts
        dataset['self_teacher_prompt'] = formatted_self_teacher_prompts
        return dataset

    raw_datasets = {
        split: load_dataset("json", data_files=path, split='train')
        for split, path in config.dataset_path.items()
    }

    with training_args.main_process_first():
        tokenized_datasets = {}
        for split, dataset in raw_datasets.items():
            tokenized_dataset = dataset.map(_preprocess_func, batched=True, num_proc=4)
            tokenized_datasets[split] = tokenized_dataset

    return tokenized_datasets
