import os
from typing import List, Any
from dataclasses import dataclass, field
from transformers import TrainingArguments, Seq2SeqTrainingArguments

SCRIPT_PATH = os.path.abspath(__file__)
ROOT_PATH = os.path.abspath(os.path.join(SCRIPT_PATH, '../../'))


FILE_MAPPING = {
    "general": {"train": "train.json", "valid": "valid.json"},
    # math benchmark
    "gsm8k": {"train": "train.json", "valid": "valid.json", "test": "test.json"},
    "math": {"train": "train_sample.json", "valid": "valid_sample.json", "test": "test.json", "analysis": "analysis.json"},
    "minerva": {"test": "test.json"},
    "olympiadbench": {"test": "test.json"},
    "amc23": {"test": "test.json"},
    "aime24": {"test": "test.json"},
    "aime25": {"test": "test.json"},
    "aime26": {"test": "test.json"},
    # STEM benchmark
    "gpqa": {"test": "test.json"},
    "gpqa_diamond": {"test": "test.json"},
    # general benchmark
    "mmlu": {"test": "test.json"},
    "mmlu_pro": {"test": "test.json", "analysis": "analysis.json"},
    "mmlu_redux": {"test": "test.json"},
    "supergpqa": {"test": "test.json"},
}

MAX_PROMPT = {
    # math benchmark
    "gsm8k": 260,
    "math": 1024,
    "minerva": 512,
    "olympiadbench": 1500,
    "amc23": 260,
    "aime24": 512,
    "aime25": 1024,
    "aime26": 512,
    # STEM benchmark
    "gpqa": 2800,
    "gpqa_diamond": 2800,
    # general benchmark
    "mmlu": 1024,
    "mmlu_pro": 2048,
    "mmlu_redux": 1024,
    "supergpqa": 4500,
}


@dataclass
class BaseConfiguration:
    base_model: str
    data_name: str
    per_gpu_batch_size: int = 0
    checkpoint_name: str = None

    learning_rate: float = 1e-6
    num_train_epochs: float = 0.0
    max_steps: int = -1
    grad_accum: int = 1
    
    all_strategy_steps: int = 300
    all_strategy: str = "steps"
    metric_name: str = "accuracy"
    log_level: str = "info"
    output_dir: str = "models/"
    dataset_dir: str = "datasets/"
    resume_from_checkpoint: str = None
    
    reward_type: str = None
    modulation: bool = False
    controlled_modulation: bool = False
    modulation_lambda: float = None

    top_entropy_quantile: float = 1.0

    perturbation: str = None


    def __post_init__(self):
        self.dataset_dir = os.path.abspath(os.path.join(ROOT_PATH, self.dataset_dir, self.data_name))
        self.output_dir = os.path.abspath(os.path.join(ROOT_PATH, self.output_dir))


@dataclass
class TuningConfiguration(BaseConfiguration):
    # LoRA Configuration
    lora_r: int = 32
    lora_dropout: float = 0.0
    lora_inference_mode: bool = False
    lora_bias: str = "none"

    # GRPO Configuration
    num_generations: int = 8
    num_generations_eval: int = 1
    max_completion_length: int = 1024
    temperature: float = 1.0
    beta: float = 0.001

    # Eval Configuration
    max_new_tokens: int = 1024

    # Logging Reporting
    report_to_where: str = "wandb"
    
    def __post_init__(self):
        super().__post_init__()
        self.output_dir = os.path.abspath(os.path.join(self.output_dir, self.data_name, self.base_model))

        self.lora_alpha = self.lora_r * 2
        self.lora_target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        if "Qwen3" in self.base_model:
            self.bad_word_list = ["<tool_call>", "<tool_response>"]

        if self.checkpoint_name is not None:
            self.checkpoint_path = os.path.join(self.output_dir, "merge")
        else:
            self.checkpoint_path = self.base_model
        
        if self.report_to_where is not None:
            self.report_project_name = f"{self.data_name}_train"

        self.dataset_path = {
            'train': os.path.join(self.dataset_dir, FILE_MAPPING[self.data_name]['train']), 
            'valid': os.path.join(self.dataset_dir, FILE_MAPPING[self.data_name]['valid'])
        }
        
        prefix_parts = [self.reward_type]
        if self.modulation:
            prefix_parts.append(f"modulation_{self.modulation_lambda}")

            if self.controlled_modulation:
                prefix_parts.append("dp")

            if self.top_entropy_quantile != 1.0:
                prefix_parts.append(f"top_{self.top_entropy_quantile}_entropy")

            if self.perturbation is not None:
                prefix_parts.append(self.perturbation)


        prefix = "_".join(prefix_parts)
        self.run_name = f"dc_{prefix}"
        self.output_dir = os.path.abspath(os.path.join(self.output_dir, prefix))


@dataclass
class GenerationConfiguration(BaseConfiguration):
    baseline: str = None
    checkpoint_path: str = None
    analysis: bool = False
    generate_dir: str = "results/"
    with_prompt: str = "false"
    generation_type: str = None
    generation_times: int = 1
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.0
    temperature: float = 0.6
    top_p: float = 0.9
    num_generations_eval: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.output_dir = os.path.abspath(os.path.join(self.output_dir, "general", self.base_model))
        self.max_prompt = MAX_PROMPT[self.data_name]
        if self.analysis:
            self.dataset_path = {
                'test': os.path.join(self.dataset_dir, FILE_MAPPING[self.data_name]['analysis'])
            }
        else:
            self.dataset_path = {
                'test': os.path.join(self.dataset_dir, FILE_MAPPING[self.data_name]['test'])
            }
        self.prompt_path = os.path.join(self.dataset_dir, "prompt.json")
        self.generate_dir = os.path.abspath(os.path.join(ROOT_PATH, self.generate_dir, self.data_name, self.base_model))

        if self.checkpoint_path is None:
            if self.reward_type is None:
                generate_suffix = f'{self.generation_type}'
                self.checkpoint_path = self.base_model
            else:
                prefix_parts = [self.reward_type]
                if self.modulation is True:
                    prefix_parts.append(f"modulation_{self.modulation_lambda}")
                    
                    if self.controlled_modulation:
                        prefix_parts.append("dp")
                        
                    if self.top_entropy_quantile != 1.0:
                        prefix_parts.append(f"top_{self.top_entropy_quantile}_entropy")

                    if self.perturbation is not None:
                        prefix_parts.append(self.perturbation)

                prefix = "_".join(prefix_parts)
                self.output_dir = os.path.abspath(os.path.join(self.output_dir, prefix))
                
                if self.checkpoint_name:
                    generate_suffix = f'{prefix}_{self.checkpoint_name}_{self.generation_type}'
                    self.checkpoint_path = os.path.join(self.output_dir, self.checkpoint_name)
                else:
                    generate_suffix = f'{prefix}_{self.generation_type}'
                    self.checkpoint_path = os.path.join(self.output_dir)
        else:
            generate_suffix = f"{self.baseline}_{self.generation_type}"
        
        self.generate_path = os.path.join(self.generate_dir, f'{generate_suffix}')
        
        # Prepare generation
        if self.generation_type == "greedy":
            self.repetition_penalty = 1.0
            self.temperature = 0.0
            self.top_p = 1.0
        elif self.generation_type == "sample":
            self.repetition_penalty = 1.0

        self._create_generate_dir()
    
    def _create_generate_dir(self):
        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir, exist_ok=True)
    

@dataclass
class GRMTrainingArguments(Seq2SeqTrainingArguments):
    _VALID_DICT_FIELDS = TrainingArguments._VALID_DICT_FIELDS + ["model_init_kwargs"]

    ###### GRM Config ######
    reward_type: str | None = field(default=None)

    bad_word_list: List[str] | None = field(
        default=None,
        metadata={
            "help": "List of bad words (strings) to filter during generation."
        },
    )

    stop_strings: str | None = field(default=None)

    modulation: bool | None = field(default=False)
    controlled_modulation: bool | None = field(default=False)
    modulation_lambda: float | None = field(default=None)

    perturbation: str | None = field(default=None)
    analysis: bool | None = field(default=False)
    ###### GRM Config ######

    # Override fields from TrainingArguments whose help strings contain unescaped "%" characters.
    # argparse interprets "%" as a format specifier, raising TypeError when rendering --help output.
    # Fixed upstream in transformers v5.3.0, but overridden here to support older versions.
    # - Introduced in v5.2.0; fixed in v5.3.0
    use_liger_kernel: bool = field(
        default=False,
        metadata={
            "help": "Enable Liger Kernel optimizations. Increases throughput by ~20%% and reduces memory by ~60%%."
        },
    )

    # Parameters that control the model and reference model
    model_init_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )
    disable_dropout: bool = field(
        default=False,
        metadata={
            "help": "Whether to disable dropout in the model. This is useful for training with a reference model, as "
            "it prevents the model from generating different logprobs for the same input."
        },
    )
    cast_lm_head_to_fp32: bool = field(
        default=False,
        metadata={
            "help": "Whether to cast the language modeling head of the policy and reference, models to float32."
            "As recommended by the [ScaleRL](https://huggingface.co/papers/2510.13786) recipe. This flag is only "
            "supported when the model has untied word embedding and language modeling head layers i.e. "
            "`tie_word_embeddings` in the model config is False."
        },
    )
    
    # Parameters that control the data preprocessing
    # The default value remove_unused_columns is overwritten from the parent class, because in GRPO we usually rely on
    # additional columns to compute the reward
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    num_generations: int | None = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The effective batch size (num_processes * per_device_batch_size "
            "* gradient_accumulation_steps) must be evenly divisible by this value."
        },
    )
    num_generations_eval: int | None = field(
        default=None,
        metadata={
            "help": "Number of generations to sample during evaluation. This allows using fewer generations during "
            "evaluation to save computation. If `None`, uses the value of `num_generations`."
        },
    )
    max_completion_length: int | None = field(
        default=512,
        metadata={"help": "Maximum length of the generated completion."},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_dataset: bool | None = field(
        default=True,
        metadata={"help": "Whether to shuffle the training dataset."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, the prompts ids and completions ids will be padded to a multiple of this value."},
    )

    # Parameters that control generation
    generation_batch_size: int | None = field(
        default=None,
        metadata={
            "help": "Batch size to use for generation. If `None`, it defaults to the effective training batch size: "
            "`per_device_train_batch_size * num_processes * steps_per_generation`."
        },
    )
    steps_per_generation: int | None = field(
        default=None,
        metadata={"help": "Number of steps per generation. If `None`, it defaults to `gradient_accumulation_steps`."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: int = field(
        default=0,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `0`, "
            "top-k-filtering is disabled and all tokens are considered."
        },
    )
    min_p: float | None = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to `GenerationConfig` (if using transformers) or "
            "`SamplingParams` (if using vLLM) when sampling completions. This can be used to further customize the "
            "generation behavior, such as setting `suppress_tokens`, `num_beams`, etc. If it contains keys that "
            "conflict with the other generation parameters (like `min_p`, `top_p`, etc.), they will override them."
        },
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional keyword arguments to pass to the `apply_chat_template` function when generating "
            "completions."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    cache_implementation: str | None = field(
        default=None,
        metadata={"help": "Implementation of the cache method for faster generation when use_vllm is set to False."},
    )

    # Parameters that control generation acceleration powered by vLLM
    use_vllm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed."
        },
    )
    vllm_mode: str = field(
        default="colocate",
        metadata={
            "help": "Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `'server'` or "
            "`'colocate'`. `'server'`: The trainer will send generation requests to a separate vLLM server. Make sure "
            "a TRL vLLM server is running (start with `trl vllm-serve`). `'colocate'`: vLLM will run in the same "
            "process and share the training GPUs. This avoids the need for a separate server but may cause resource "
            "contention with training."
        },
    )
    vllm_model_impl: str = field(
        default="vllm",
        metadata={
            "help": "Model implementation to use for vLLM. Must be one of `transformers` or `vllm`. `transformers`: "
            "Use the `transformers` backend for model implementation. `vllm`: Use the `vllm` library for "
            "model implementation."
        },
    )
    vllm_enable_sleep_mode: bool = field(
        default=False,
        metadata={
            "help": "Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory "
            "usage low, but waking the engine adds host–device transfer latency."
        },
    )
    vllm_structured_outputs_regex: str | None = field(
        default=None,
        metadata={"help": "Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled."},
    )

    # Parameters that control the vLLM server (only used when `vllm_mode` is `"server"`)
    vllm_server_base_url: str | None = field(
        default=None,
        metadata={
            "help": "Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` "
            "and `vllm_server_port` are ignored."
        },
    )
    vllm_server_host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_port: int = field(
        default=8000,
        metadata={"help": "Port of the vLLM server to connect to. Ignored if vllm_server_base_url is provided."},
    )
    vllm_server_timeout: float = field(
        default=240.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    vllm_group_port: int = field(
        default=51216,
        metadata={
            "help": "Port number for the weight update group. This is used to communicate with the vLLM server. "
            "Unless the port is occupied, there is no need to change it.",
        },
    )

    # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
    vllm_gpu_memory_utilization: float = field(
        default=0.3,
        metadata={
            "help": "Control the GPU memory utilization for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_gpu_memory_utilization` flag."
        },
    )
    vllm_max_model_length: int | None = field(
        default=None,
        metadata={
            "help": "Context window for vLLM. Set it to at least the maximum prompt length in the dataset plus "
            "`max_completion_length`; if omitted, it is inferred from the model config."
        },
    )
    vllm_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Control the tensor parallel size for vLLM. This setting only applies when `vllm_mode` is set "
            "to `'colocate'`. If you are using `vllm_mode='server'`, this parameter must be passed separately when "
            "launching the vLLM server via the `--vllm_tensor_parallel_size` flag."
        },
    )

    # Parameters that control the training
    beta: float = field(
        default=0.0,
        metadata={
            "help": "KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and "
            "improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement "
            "learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`."
        },
    )
    num_iterations: int = field(
        default=1,
        metadata={"help": "Number of iterations per batch (denoted as μ in the algorithm)."},
    )
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Epsilon value for clipping."},
    )
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Enables the upper clipping bound in two-sided GRPO loss when set to a float. If `None` "
            "(default), standard GRPO clipping is used. Recommended to be greater than `1 + ε` when enabled. This "
            "method is introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)."
        },
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={
            "help": "Upper-bound epsilon value for clipping. If not specified, it defaults to the same value as the "
            "lower-bound specified in argument `epsilon`. Paper DAPO recommends `0.28`. "
            "When used with `loss_type='cispo'`, this corresponds to the ε_max param specified in the"
            "[ScaleRL paper]https://huggingface.co/papers/2510.13786) and the recommended value is `5.0`."
        },
    )
    sapo_temperature_neg: float = field(
        default=1.05,
        metadata={
            "help": "Temperature for tokens with non-positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )
    sapo_temperature_pos: float = field(
        default=1.0,
        metadata={
            "help": "Temperature for tokens with positive advantage scores used in the `sapo` loss function. "
            "This parameter is introduced in the [Soft Adaptive Policy Optimization "
            "paper](https://huggingface.co/papers/2511.20347)."
        },
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={
            "help": "Controls whether importance sampling ratios are computed at the `'token'` or `'sequence'` level. "
            "`'token'` keeps the raw per-token log-probability ratios (one weight per token).  `'sequence'` averages "
            "the log-probability ratios across valid tokens to produce a single ratio per sequence. The GSPO paper "
            "shows that sequence-level sampling often yields more stable training and better alignment with "
            "sequence-level rewards."
        },
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    multi_objective_aggregation: str = field(
        default="sum_then_normalize",
        metadata={
            "help": "Method to aggregate multiple reward functions. Supported values are: "
            "`'sum_then_normalize'` (default): First sums the weighted rewards from each reward function, then "
            "applies reward scaling/normalization as specified by `scale_rewards` (see `scale_rewards` for details). "
            "`'normalize_then_sum'`: First normalizes/scales each reward function across generations (within each "
            "group), then sums the normalized rewards using the specified weights. The aggregated reward is then "
            "normalized at the batch level when forming advantages. This is the suggested approach from the paper "
            "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization."
        },
    )
    scale_rewards: str = field(
        default="group",
        metadata={
            "help": "Specifies the scaling strategy for rewards. Supported values are: "
            "`True` or `group'` (default): rewards are scaled by the standard deviation within each group, ensuring "
            "unit variance within a group. "
            "`'batch'`: rewards are scaled by the standard deviation across the entire batch, as recommended in the "
            "PPO Lite paper. "
            "`False` or `'none'`: no scaling is applied. The Dr. GRPO paper recommends not scaling rewards, as "
            "scaling by the standard deviation introduces a question-level difficulty bias."
        },
    )
    loss_type: str = field(
        default="dapo",
        metadata={
            "help": "Specifies the loss formulation to use. Supported values are 'grpo', 'dapo', 'bnpo', and "
            "'dr_grpo'. "
            "'grpo': Aggregates token-level losses by normalizing over sequence length. Not recommended due to length "
            "bias—this approach tends to prefer shorter completions with positive advantages and longer ones with "
            "negative advantages. "
            "'dapo' (default): Aggregates token-level losses by normalizing with the number of active token in the "
            "global accumulated batch. This method was introduced in the DAPO paper to eliminate length bias. "
            "'dr_grpo': Aggregates token-level losses by normalizing with a global constant. This method was "
            "introduced in the Dr. GRPO paper to eliminate length bias. The value of the constant corresponds to "
            "`max_completion_length`. "
            "'bnpo': Aggregates token-level losses by normalizing with the number of active token in the local batch. "
            "Note that normalization is performed over the local batch only, so results may slightly vary depending "
            "on the local batch size, despite a constant effective batch size. When using "
            "`per_device_train_batch_size==1`, the loss is equivalent to the GRPO loss."
            "'cispo': Clips the importance sampling weights instead of the advantage scaled importance weights. "
            "The clipped weights are then multiplied with the advantages and policy model's log probs. "
            "Individual token losses are aggregated by normalizing with the number of active tokens in "
            "the global accumulated batch. This method was introduced in the "
            "[MiniMax-M1 paper](https://huggingface.co/papers/2506.13585). "
            "'sapo': Soft Adaptive Policy Optimization loss, as introduced in the "
            "[Soft Adaptive Policy Optimization paper](https://huggingface.co/papers/2511.20347). "
            "Replaces hard clipping with a smooth, temperature-controlled gate that adaptively attenuates "
            "off-policy updates while preserving useful learning signals."
            "'luspo': Length-Unbiased Sequence Policy Optimization loss. A sequence-level loss that scales each "
            "sequence's loss by its length. This is a modification of GSPO and requires "
            "`importance_sampling_level='sequence'`. Introduced in the [LUSPO "
            "paper](https://huggingface.co/papers/2602.05261)."
        },
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={
            "help": "When enabled, truncated completions are excluded from the loss calculation, preventing them from "
            "being incorrectly penalized and introducing noise during training. According to the DAPO paper, this is "
            "a good practice for training stability."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.6,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=512,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={
            "help": "ρ parameter from Beyond the 80/20 Rule. Keeps in the policy loss term only the top-ρ quantile of "
            "tokens by entropy of the probability distribution at each sequence position, improving results. Range: "
            "[0.0-1.0]. A value of `0.0` masks all but the highest entropy token; `1.0` keeps all tokens. The paper "
            "recommends a value of `0.2`. If used with `mask_truncated_completions=True`, only tokens from "
            "non-truncated completions are considered."
        },
    )
    vllm_importance_sampling_correction: bool = field(
        default=True,
        metadata={
            "help": "Whether to apply Importance Sampling (IS) to correct for the mismatch between vLLM "
            "completion logprobs and recomputed training logprobs. If set to `False`, no IS is applied "
            "regardless of `vllm_importance_sampling_mode`. When `True`, the selected mode determines how "
            "IS ratios are computed and constrained."
        },
    )

    vllm_importance_sampling_mode: str = field(
        default="sequence_mask",
        metadata={
            "help": "Specifies how Importance Sampling (IS) is performed when "
            "vllm_importance_sampling_correction=True. Modes are defined along two orthogonal "
            "dimensions: (1) constraint, which determines how to handle ratios above "
            "vllm_importance_sampling_cap (C)—either truncation (clip from above, ρ ← min(ρ, C)) or "
            "masking (set ratios above C to zero); and (2) granularity, which determines whether "
            "ratios are computed per token or as a single sequence-level ratio applied to all tokens. "
            "Supported options are: 'token_truncate', 'token_mask', 'sequence_truncate', and "
            "'sequence_mask'."
        },
    )

    vllm_importance_sampling_cap: float = field(
        default=3.0,
        metadata={
            "help": "Importance sampling cap C used by `vllm_importance_sampling_mode`. For '*_truncate' modes, "
            "ratios are clipped from above at C. For '*_mask' modes, ratios larger than C are set to zero."
        },
    )
    off_policy_mask_threshold: float | None = field(
        default=None,
        metadata={
            "help": "Threshold for off-policy sequence masking. If `None`, off-policy sequence masking is disabled. "
            "When set, sequences with negative advantages and high KL divergence are masked out to stabilize "
            "training. This parameter corresponds to the `delta` threshold in Equation 9 of the [DeepSeek-V3.2 "
            "paper](https://huggingface.co/papers/2512.02556). It expects a positive value (e.g., 0.5)."
        },
    )
    use_bias_correction_kl: bool = field(
        default=False,
        metadata={
            "help": "Whether to use the unbiased KL divergence estimator with importance sampling correction. This "
            "corrects the KL divergence estimate by multiplying it with the importance sampling ratio. "
            "This is described in the [DeepSeek-V3.2 paper](https://huggingface.co/papers/2512.02556)."
        },
    )

    # Parameters that control the logging
    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log a sample of (prompt, completion) pairs every `logging_steps` steps. If `rich` is "
            "installed, it prints the sample. If `wandb` logging is enabled, it logs it to `wandb`."
        },
    )
    num_completions_to_print: int | None = field(
        default=None,
        metadata={"help": "Number of completions to print with `rich`. If `None`, all completions are logged."},
    )
    log_unique_prompts: bool = field(
        default=False,
        metadata={
            "help": "Whether to log unique prompts. If `True`, only unique prompts are logged. If `False`, all "
            "prompts are logged."
        },
    )
    log_completions_hub_repo: str | None = field(
        default=None,
        metadata={
            "help": "Hugging Face Hub repository to save the completions. Should be a complete repository name like "
            "`'username/reponame'` or `'orgname/reponame'`, or just `'reponame'` in which case the repository will "
            "be created in the currently-logged-in Hugging Face user's namespace. Note that this repository will be "
            "public unless you set `hub_private_repo=True` or your organization's default is to create private "
            "repositories."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        if self.log_completions_hub_repo is not None and not self.log_completions:
            raise ValueError(
                "log_completions_hub_repo is set, but log_completions is False. Enable log_completions to upload "
                "completions to the Hub, or unset log_completions_hub_repo."
            )

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Determine the number of generations to use for evaluation
            num_generations = self.num_generations_eval or self.num_generations

            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by the number of generations used for evaluation ({num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.delta is not None and self.use_liger_kernel:
            raise ValueError("Liger kernel does not support two-sided GRPO loss yet.")