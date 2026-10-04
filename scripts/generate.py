import os
import sys
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
sys.path.insert(0, ROOT_PATH)

import json
import logging
import datasets
import transformers
import pandas as pd
import torch.distributed as dist
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
)

from src import *

logger = logging.getLogger(__name__)

def setup_generation_args(config):
    generation_args = GRMTrainingArguments(
        output_dir=config.output_dir,
        # Evaluation
        per_device_eval_batch_size=config.per_gpu_batch_size,
        #Logging & Monitoring Training
        logging_strategy=config.all_strategy,
        # Logging
        log_level=config.log_level,
        disable_tqdm=False,
        # Best Model Tracking
        metric_for_best_model=config.metric_name,
        # GRPO
        num_generations_eval = config.num_generations_eval,
        # External Script Flags (not used by Trainer)
        modulation=config.modulation,
        controlled_modulation=config.controlled_modulation,
        modulation_lambda=config.modulation_lambda,
        top_entropy_quantile=config.top_entropy_quantile,
        analysis=config.analysis,
        # Generation
        use_vllm=True,
        vllm_max_model_length=config.max_new_tokens+config.max_prompt,
        vllm_importance_sampling_correction=False,
        repetition_penalty=config.repetition_penalty,
        temperature=config.temperature,
        top_p=config.top_p,
        max_completion_length=config.max_new_tokens
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if generation_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = generation_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {generation_args.local_process_index}, device: {generation_args.device}, n_gpu: {generation_args.n_gpu}"
    )
    
    return generation_args


def setup_model_and_tokenizer(config, world_rank):
    """
    Setup model and tokenizer based on generation configuration.
    """
    if config.checkpoint_path == config.base_model:
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            use_fast=True,
            padding_side="left"
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype="auto"
        )

        model.eval()
        return model, tokenizer
    
    # For loading SPO
    if os.path.isfile(
        os.path.join(config.checkpoint_path, "pytorch_model.bin")
    ):
        logger.info(
            f"Loading merged HuggingFace checkpoint from "
            f"{config.checkpoint_path}"
        )

        # SPO uses exactly the same Qwen tokenizer as the base model.
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            use_fast=True,
            padding_side="left"
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            torch_dtype="auto"
        )

        model.config.use_cache = True
        model.eval()

        return model, tokenizer

    merge_dir = os.path.join(config.checkpoint_path, 'merge')
    if world_rank == 0 and not os.path.isdir(merge_dir):
        peft_config = PeftConfig.from_pretrained(config.checkpoint_path)

        tokenizer = AutoTokenizer.from_pretrained(
            peft_config.base_model_name_or_path,
            use_fast=True,
            padding_side="left"
        )

        base_config = AutoConfig.from_pretrained(
            peft_config.base_model_name_or_path
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            config=base_config,
            torch_dtype="auto"
        )
        
        peft_model = PeftModel.from_pretrained(
            model=base_model,
            model_id=config.checkpoint_path,
            config=peft_config,
            torch_dtype="auto"
        )
        peft_model = peft_model.merge_and_unload(progressbar=True)

        peft_model.save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)

        del peft_model, base_model
    
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        merge_dir,
        use_fast=True,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        merge_dir,
        torch_dtype="auto",
    )
    model.config.use_cache = True

    model.eval()
    return model, tokenizer


def run_generation(generation_args, config, model, tokenizer, test_dataset):
    """
    Main generation loop.
    """

    logger.info(f'============ CHECK IMPORTANT INFO =================')
    logger.info(f">> Load model from {config.checkpoint_path} <<")
    logger.info(f">> Save result to {config.generate_path}.json <<")
    
    trainer = GRMTrainer(
        model=model,
        reward_funcs=likelihood_reward,
        args=generation_args,
        train_dataset=test_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        only_generate=True,
    )

    analysis_results = []

    for num in range(config.generation_times):
        logger.info(f"============ Generate for {num+1} time ============")
        if config.analysis:
            trainer.analysis_records = []

        test_preds = trainer.predict(test_dataset)
        
        if trainer.is_world_process_zero():
            total_results = pred2text(test_preds, config, tokenizer)
            result_df = pd.DataFrame({
                'question': test_dataset['question'],
                'answer': test_dataset['answer'],
                'pred': total_results['prediction']
            })

            if config.analysis:
                for record in trainer.analysis_records:
                    record["generation_round"] = num
                analysis_results.extend(trainer.analysis_records)

    if trainer.is_world_process_zero():
        with open(f"{config.generate_path}.json", 'w') as f:
            json.dump(result_df.to_dict(orient='records'), f, indent=4)
            
        if config.analysis:
            with open(f"{config.generate_path}_analysis.json", 'w') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    set_dist_env()
    world_rank = get_rank()

    args = parse_args(use_training_args=False)
    config = GenerationConfiguration(**vars(args))
    generation_args = setup_generation_args(config)
    model, tokenizer = setup_model_and_tokenizer(config, world_rank)
    data = prepare_generation_data(config, tokenizer, generation_args)
    run_generation(generation_args, config, model, tokenizer, data['test'])

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()