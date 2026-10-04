import os
import sys
SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))
sys.path.insert(0, ROOT_PATH)

import logging
import datasets
import transformers
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.trainer_utils import get_last_checkpoint

from src import *

logger = logging.getLogger(__name__)

def setup_training_args(config):
    training_args = GRMTrainingArguments(
        output_dir=config.output_dir,
        # Training Duration and Batch Size
        per_device_train_batch_size=config.per_gpu_batch_size,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        # Learning Rate & Scheduler
        learning_rate=config.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,
        # Regularization & Training Stability
        gradient_accumulation_steps=config.grad_accum,
        # Mixed Precision Training
        bf16=True,
        # Gradient Checkpointing
        gradient_checkpointing=False,
        #Logging & Monitoring Training
        logging_strategy=config.all_strategy,
        logging_steps=10,
        logging_first_step=True,
        # Logging
        log_level=config.log_level,
        disable_tqdm=False,
        # Experiment Tracking
        report_to=config.report_to_where,
        run_name=config.run_name,
        # Evaluation
        eval_strategy=config.all_strategy,
        eval_steps=config.all_strategy_steps,
        per_device_eval_batch_size=16,
        eval_on_start=False,
        # Metrics Computation
        include_for_metrics=['loss'],
        # Checkpointing & Saving
        save_strategy=config.all_strategy,
        save_steps=config.all_strategy_steps,
        # Best Model Tracking
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_name,
        # Dataloader
        label_names=["labels"],
        # DDP (DistributedDataParallel)
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        # GRPO
        loss_type="grpo",
        num_generations=config.num_generations,
        num_generations_eval=config.num_generations_eval,
        max_completion_length=config.max_completion_length,
        temperature=config.temperature,
        disable_dropout=True,
        beta=config.beta,
        mask_truncated_completions=True,
        top_entropy_quantile=config.top_entropy_quantile,
        # vLLM
        use_vllm=True,
        vllm_enable_sleep_mode=True,
        vllm_max_model_length=config.max_completion_length+400,
        vllm_importance_sampling_correction=False,
        # GRM
        reward_type=config.reward_type,
        bad_word_list=config.bad_word_list if "Qwen3" in config.base_model else None,
        modulation=config.modulation,
        controlled_modulation=config.controlled_modulation,
        modulation_lambda=config.modulation_lambda,
        perturbation=config.perturbation
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_process_index}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    if config.report_to_where is not None:
        os.environ["WANDB_PROJECT"]=config.report_project_name
    
    return training_args


def setup_model_and_tokenizer(config):
    """
    Setup model and tokenizer based on training configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        config.checkpoint_path,
        use_fast=True,
        padding_side="left"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint_path,
        torch_dtype="auto"
    )
    
    common_lora_kwargs = dict(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=config.lora_inference_mode,
        target_modules=config.lora_target_modules,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias
    )

    lora_config = LoraConfig(**common_lora_kwargs)
    model = get_peft_model(model, lora_config)

    return model, tokenizer


def run_tuning(training_args, model, tokenizer, tokenized_datasets):
    """
    Main training loop.
    """

    logger.info(f'============ CHECK IMPORTANT INFO =================')
    logger.info(f">> Load model from {config.checkpoint_path} <<")
    logger.info(f">> If use cache: {model.model.config.use_cache}")
    if training_args.reward_type == "likelihood":
        reward_function = likelihood_reward
    elif training_args.reward_type == "binary":
        reward_function = binary_reward
    trainer = GRMTrainer(
        model=model,
        reward_funcs=reward_function,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['valid'],
        processing_class=tokenizer,
        compute_metrics=compute_metric_text
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.save_state()

    if trainer.is_world_process_zero():
        peft_model = trainer.model
        peft_model.eval()
        merged_model  = peft_model.merge_and_unload(progressbar=True)
        merge_dir = os.path.join(trainer.args.output_dir, "merge")
        merged_model .save_pretrained(merge_dir)
        tokenizer.save_pretrained(merge_dir)

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    set_dist_env()

    args = parse_args(use_training_args=True)
    config = TuningConfiguration(**vars(args))
    training_args = setup_training_args(config)
    model, tokenizer = setup_model_and_tokenizer(config)
    data = prepare_training_data(config, tokenizer, training_args)
    run_tuning(training_args, model, tokenizer, data)

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()