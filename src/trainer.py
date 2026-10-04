from transformers.trainer import *
from transformers.trainer_utils import *
from transformers.trainer_utils import _is_peft_model
from transformers.trainer_seq2seq import *
from transformers.training_args_seq2seq import *
from trl.trainer.grpo_trainer import *
from trl.generation.vllm_generation import *
from trl.models.utils import _ForwardRedirection
from transformers.utils import logging
from transformers.trainer_optimizer import (
    _OPTIMIZER_HANDLERS,
    _parse_optim_args
)
from collections.abc import Mapping, Sized
logger = logging.get_logger(__name__)

class VLLMGenerationGRM:
    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        is_fsdp_enabled: bool,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # vLLM configuration
        mode: str = "colocate",
        structured_outputs_regex: str | None = None,
        # Server mode configuration
        server_base_url: str | None = None,
        server_host: str = "0.0.0.0",
        server_port: int = 8000,
        server_timeout: float = 240.0,
        group_port: int = 51216,
        # Colocate mode configuration
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        max_num_seqs: int | None = None,
        enable_sleep_mode: bool = False,
        model_impl: str = "auto",
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
    ):
        self.model = model
        self.accelerator = accelerator
        self.is_fsdp_enabled = is_fsdp_enabled
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.structured_outputs_regex = structured_outputs_regex

        # Server mode configuration
        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.group_port = group_port
        self.server_timeout = server_timeout

        # Colocate mode configuration
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.max_num_seqs = max_num_seqs
        self.enable_sleep_mode = enable_sleep_mode
        self.model_impl = model_impl

        # Generation configuration
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.logprobs = logprobs
        self.generation_kwargs = generation_kwargs or {}

        self._weights_synced = False

        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM in server or colocate mode."""
        model = self.model
        accelerator = self.accelerator

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install trl[vllm]` to use it."
            )

        if self.mode == "server":
            if accelerator.is_main_process:
                if self.server_base_url is not None:
                    base_url = self.server_base_url
                else:
                    base_url = f"http://{self.server_host}:{self.server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=self.group_port, connection_timeout=self.server_timeout
                )
                self.vllm_client.init_communicator(device=torch.cuda.current_device())

        elif self.mode == "colocate":
            # Make sure tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % self.tensor_parallel_size == 0:
                raise ValueError(
                    f"tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if self.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `tensor_parallel_size` ranks.
                # For example, if world_size=8 and tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
                        for i in range(accelerator.num_processes // self.tensor_parallel_size)
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
            # Ensure distributed rendezvous variables are set without colliding across concurrent runs
            ensure_master_addr_port()

            quantization = None
            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        quantization = "bitsandbytes"
                        break
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

            # Build LLM initialization kwargs
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_length,
                max_num_seqs=self.max_num_seqs,
                enable_sleep_mode=self.enable_sleep_mode,
                model_impl=self.model_impl,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=accelerator.process_index // self.tensor_parallel_size,
                # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768) - thinking there's not enough memory
                max_num_batched_tokens=4096,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                logprobs_mode="processed_logprobs",
                quantization=quantization,
            )
            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.mode}'.")

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        accelerator.wait_for_everyone()

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Fix parameter name for vLLM compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _sync_fsdp1_params_to_vllm(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        accelerator = self.accelerator

        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp1_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.mode == "server" and accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    def _sync_fsdp2_params_to_vllm(self, module: nn.Module):
        """FSDP2-specific parameter synchronization."""
        accelerator = self.accelerator

        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            param = param.full_tensor()

            if self.mode == "server" and accelerator.is_main_process:
                self.vllm_client.update_named_param(name, param)
            elif self.mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(name, param)])

    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
        # Wake up vLLM weights before loading to ensure device memory is mapped. Without this, load_weights() writes to
        # freed/unmapped memory when sleep mode is active, which crashes on backends with strict physical memory
        # management (e.g., Ascend NPU). See https://github.com/huggingface/trl/issues/5142
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])

        model = self.model
        accelerator = self.accelerator
        is_fsdp_enabled = self.is_fsdp_enabled

        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if is_peft_model(model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with gather_if_zero3(list(model.parameters())):
                model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(model)  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
                        if model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                        if self.mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if is_fsdp_enabled:
                fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
                fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                if fsdp_version == 1:
                    self._sync_fsdp1_params_to_vllm(model)  # use memory-efficient post-order traversal for FSDP
                elif fsdp_version == 2:
                    self._sync_fsdp2_params_to_vllm(model)
            else:
                for name, param in model.named_parameters():
                    name = self._fix_param_name_to_vllm(name)
                    with gather_if_zero3([param]):
                        if self.mode == "server" and accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()
        
        self._weights_synced = True

    def generate(
        self,
        prompts: list[list[int]],
        images: list[list | None] | None,
        num_generations: int,
        profiler: ProfilingContext | None = None,
    ) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of token ID lists, one per prompt (already tokenized).
            images: Optional list of image lists for VLM support. Each element is a list of PIL images for the
                corresponding prompt, or `None` if no images for that prompt. `None` if no images at all.
            num_generations: Number of generations per prompt.
            profiler: Optional profiler for performance tracking.

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, logprob_token_ids).

            - `prompt_ids`: `list[list[int]]` of shape `(batch_size, prompt_len)`.
            - `completion_ids`: `list[list[int]]` of shape `(batch_size, completion_len)`.
            - `logprobs`: `list[list[list[float | None]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `logprob_token_ids`: `list[list[list[int]]]` of shape `(batch_size, completion_len, num_logprobs)`.

            `num_logprobs` is 1 when `logprobs=0`, or up to N+1 when `logprobs=N` (the sampled token is always included
            and may fall outside the top-N).
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length

        # Wake up colocated vLLM weights if needed (idempotent if already awake from sync_weights)
        if self.mode == "colocate" and self.enable_sleep_mode:
            empty_cache()  # required to avoid OOM in some cases

            if not self._weights_synced:
                raise RuntimeError(
                    "vLLM weights are not synced. In colocate + sleep mode, call "
                    "`sync_weights()` before `generate()`. Do not rely on `reload_weights()` "
                    "because it may reload the initial checkpoint instead of the current trainer weights."
                )

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            all_prompts = gather_object(prompts)
            # Always gather images (even when None) to avoid deadlock: images may be None on some ranks
            # and non-None on others in mixed datasets, and gather_object is a collective operation.
            all_images = gather_object(images if images is not None else [None] * len(prompts))
            if all(img is None for img in all_images):
                all_images = None

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and
                # generate num_generations outputs for each one. This is faster than generating outputs for each
                # duplicate prompt individually.
                ordered_set_of_prompt_ids = all_prompts[::num_generations]
                ordered_set_of_images = all_images[::num_generations] if all_images is not None else None

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                }
                with profiler:
                    output = self.vllm_client.generate(
                        prompts=ordered_set_of_prompt_ids,
                        images=ordered_set_of_images,
                        **sampling_params,
                    )
                    payload = (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output.get("logprob_token_ids"),
                    )
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_logprob_token_ids = obj_list[0]

            # vllm_client.generate(n=num_generations) returns num_generations completions per prompt.
            # Duplicate prompt_ids to align with per-completion entries.
            all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            process_slice = slice(
                accelerator.process_index * len(prompts),
                (accelerator.process_index + 1) * len(prompts),
            )
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
            logprob_token_ids = all_logprob_token_ids[process_slice] if all_logprob_token_ids is not None else None

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0 if min_p is None else min_p,
                "max_tokens": max_completion_length,
                "logprobs": self.logprobs,
            }
            generation_kwargs.update(self.generation_kwargs)

            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            if self.tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts)
                gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                all_prompts = [p for sublist in gathered_prompts for p in sublist]
                # Always gather images (even when None) to avoid deadlock: images may be None on some
                # ranks and non-None on others in mixed datasets, and all_gather_object is collective.
                local_images = images if images is not None else [None] * len(prompts)
                gathered_images = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_images, local_images, group=self.tp_group)
                all_images = [img for sublist in gathered_images for img in sublist]
                if all(img is None for img in all_images):
                    all_images = None
            else:
                all_prompts = prompts
                all_images = images

            if self.enable_sleep_mode:
                self.llm.wake_up(tags=["kv_cache"])

            # Build vLLM-compatible prompt inputs with token IDs and optional multi-modal data
            vllm_prompts = []
            if all_images is not None:
                for ids, img_list in zip(all_prompts, all_images, strict=True):
                    row = {"prompt_token_ids": ids}
                    if img_list is not None:
                        row["multi_modal_data"] = {"image": img_list if len(img_list) > 1 else img_list[0]}
                    vllm_prompts.append(row)
            else:
                vllm_prompts = [{"prompt_token_ids": ids} for ids in all_prompts]

            with profiler:
                all_outputs = self.llm.generate(vllm_prompts, sampling_params=sampling_params, use_tqdm=False)

            all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            all_logprobs, all_logprob_token_ids = extract_logprobs(all_outputs)

            if self.tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                prompt_ids = all_prompt_ids[tp_slice]
                completion_ids = all_completion_ids[tp_slice]
                logprobs = all_logprobs[tp_slice] if all_logprobs is not None else None
                logprob_token_ids = all_logprob_token_ids[tp_slice] if all_logprob_token_ids is not None else None
            else:
                prompt_ids = all_prompt_ids
                completion_ids = all_completion_ids
                logprobs = all_logprobs
                logprob_token_ids = all_logprob_token_ids

            if self.enable_sleep_mode:
                self.llm.sleep(level=2)
                self._weights_synced = False

        return prompt_ids, completion_ids, logprobs, logprob_token_ids
    

class BaseTrainer(Trainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module | None = None,
        args: TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: "Dataset | IterableDataset | datasets.Dataset | None" = None,
        eval_dataset: "Dataset | dict[str, Dataset] | datasets.Dataset | None" = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[..., PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        # Init flow:
        #   1. Args & seed               – defaults, determinism
        #   2. Accelerator & logging     – accelerator, memory tracker, log level, device setup
        #   3. Model resolution          – model / model_init, Liger Kernel, quantization checks
        #   4. Distributed strategy      – model-parallel, FSDP, SageMaker MP flags
        #   5. Device placement          – move model to device, model wrapping
        #   6. Model introspection       – loss kwargs, label names, label smoother
        #   7. Store init arguments      – data, callables, optimizer, scheduler, validation
        #   8. Callbacks                 – reporting integrations, JIT checkpoint, progress bar
        #   9. Hub & output              – repo init, output directory
        #  10. Training state            – TrainerState, TrainerControl, internal bookkeeping
        #  11. Finalize                  – use_cache, XLA FSDPv2 mesh, memory tracker stop

        # ---- 1. Args & seed --------------------------------------------------------
        if args is None:
            output_dir = "tmp_trainer"
            logger.info(f"No `TrainingArguments` passed, using `output_dir={output_dir}`.")
            args = TrainingArguments(output_dir=output_dir)
        self.args = args
        # Seed must be set before instantiating the model when using model_init
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

        # ---- 2. Accelerator & logging ----------------------------------------------
        # `create_accelerator_and_postprocess` reads self.model and self.args,
        # and may set self.deepspeed — store temporary refs before calling it.
        self.deepspeed = None
        self.model = model
        self.create_accelerator_and_postprocess()

        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)

        args._setup_devices  # force device and distributed setup init explicitly

        # ---- 3. Model resolution ----------------------------------------------------
        if model is None:
            if model_init is not None:
                self.model_init = model_init
                model = self.call_model_init()
            else:
                raise RuntimeError("`Trainer` requires either a `model` or `model_init` argument")
        else:
            if model_init is not None:
                raise ValueError("`Trainer` requires either a `model` or `model_init` argument, but not both.")
            self.model_init = model_init

        if model.__class__.__name__ in MODEL_MAPPING_NAMES:
            raise ValueError(
                f"The model you have picked ({model.__class__.__name__}) cannot be used as is for training: it only "
                "computes hidden states and does not accept any labels. You should choose a model with a head "
                "suitable for your task like any of the `AutoModelForXxx` listed at "
                "https://huggingface.co/docs/transformers/model_doc/auto"
            )

        validate_quantization_for_training(model)

        # ---- 4. Distributed strategy ------------------------------------------------
        self.is_model_parallel = False
        if getattr(model, "hf_device_map", None) is not None:
            devices = [device for device in set(model.hf_device_map.values()) if device not in ["cpu", "disk"]]
            if len(devices) > 1:
                self.is_model_parallel = True
            elif len(devices) == 1:
                self.is_model_parallel = self.args.device != torch.device(devices[0])

        self.is_fsdp_xla_enabled = args.fsdp_config["xla"]

        if len(args.fsdp) > 0:
            if self.is_deepspeed_enabled:
                raise ValueError(
                    "Using --fsdp xxx together with --deepspeed is not possible, deactivate one of those flags."
                )
            if not args.fsdp_config["xla"] and args.parallel_mode != ParallelMode.DISTRIBUTED:
                raise ValueError("Using fsdp only works in distributed training.")

        # Postpone switching model to cuda when MP, DeepSpeed, full bf16/fp16 eval, or FSDP
        
        if args.place_model_on_device is not None:
            self.place_model_on_device = args.place_model_on_device
        elif (
            self.is_model_parallel
            or self.is_deepspeed_enabled
            or (args.fp16_full_eval or args.bf16_full_eval)
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
            or is_sagemaker_mp_enabled()
        ):
            self.place_model_on_device = False
        else:
            self.place_model_on_device = True

        # ---- 5. Device placement ----------------------------------------------------
        # Bnb Quantized models don't support `.to` operation.
        if (
            self.place_model_on_device
            and getattr(model, "quantization_method", None) != QuantizationMethod.BITS_AND_BYTES
        ):
            self._move_model_to_device(model, args.device)

        # Force n_gpu to 1 to avoid DataParallel as MP will manage the GPUs
        if self.is_model_parallel:
            self.args._n_gpu = 1

        # `self.model is self.model_wrapped` is used later to check if it's wrapped
        self.model_wrapped = model
        self.model = model

        # ---- 6. Model introspection -------------------------------------------------
        unwrapped_model = unwrap_peft_model(self.accelerator.unwrap_model(model))

        if hasattr(unwrapped_model, "accepts_loss_kwargs"):
            self.model_accepts_loss_kwargs = unwrapped_model.accepts_loss_kwargs
        else:
            forward_params = inspect.signature(unwrapped_model.forward).parameters
            self.model_accepts_loss_kwargs = any(
                k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
            )

        # Sequence Parallelism computes its own good_tokens count
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            self.model_accepts_loss_kwargs = False

        model_to_inspect = unwrap_peft_model(self.model)
        default_label_names = find_labels(model_to_inspect.__class__)
        self.label_names = default_label_names if self.args.label_names is None else self.args.label_names
        self.can_return_loss = can_return_loss(model_to_inspect.__class__)

        if self.args.label_smoothing_factor != 0:
            if getattr(self.model.config, "problem_type", None) == "multi_label_classification":
                warnings.warn(
                    "Label smoothing is not compatible with multi-label classification. "
                    "Disabling label smoothing for this training run.",
                    UserWarning,
                )
                self.label_smoother = None
            else:
                self.label_smoother = LabelSmoother(epsilon=self.args.label_smoothing_factor)
        else:
            self.label_smoother = None

        # ---- 7. Store init arguments ------------------------------------------------
        # Data
        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor))
            else default_data_collator
        )
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.processing_class = processing_class
        self.neftune_noise_alpha = args.neftune_noise_alpha

        # Callables
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.preprocess_logits_for_metrics = preprocess_logits_for_metrics

        # Optimizer & scheduler
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs

        self._validate_args()

        # ---- 8. Callbacks -----------------------------------------------------------
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)

        if self.args.enable_jit_checkpoint:
            from .trainer_jit_checkpoint import JITCheckpointCallback

            jit_callback = JITCheckpointCallback()
            default_callbacks = default_callbacks + [jit_callback]
            jit_callback.set_trainer(self)

        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)

        # ---- 9. Hub & output ---------------------------------------------------------
        self.hub_model_id = None  # Set by init_hf_repo() when push_to_hub is enabled
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # ---- 10. Training state -----------------------------------------------------
        self.control = TrainerControl()
        self.state = TrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.is_in_train = False  # True between train() entry and exit
        self.hp_name = None  # Set by hyperparameter_search() to label the trial
        self.hp_search_backend = None  # Set by hyperparameter_search() (optuna / ray / wandb)
        # Per-process FLOP counter; accumulated into self.state.total_flos then reset
        self.current_flos = 0
        # Set True by _setup_loggers() on first call to self.log()
        self._loggers_initialized = False
        # Lazily filled by _set_signature_columns_if_needed(); caches model.forward param names
        self._signature_columns = None
        # Effective batch size; may be reduced by find_executable_batch_size
        self._train_batch_size = args.train_batch_size
        # Guards one-time LR scheduler creation in create_optimizer_and_scheduler
        self._created_lr_scheduler = False

        self.control = self.callback_handler.on_init_end(self.args, self.state, self.control)

        # ---- 11. Finalize -----------------------------------------------------------
        if getattr(self.model, "config", None) is not None:
            self.model.config.use_cache = self.args.use_cache

        self.is_fsdp_xla_v2_enabled = args.fsdp_config.get("xla_fsdp_v2", False)
        if self.is_fsdp_xla_v2_enabled:
            if not IS_XLA_FSDPV2_POST_2_2:
                raise ValueError("FSDPv2 requires `torch_xla` 2.2 or higher.")
            num_devices = xr.global_runtime_device_count()
            xs.set_global_mesh(xs.Mesh(np.array(range(num_devices)), (num_devices, 1), axis_names=("fsdp", "tensor")))
        self.is_fsdp_xla_v1_enabled = self.is_fsdp_xla_enabled and not self.is_fsdp_xla_v2_enabled

        self._memory_tracker.stop_and_update_metrics()

    def _validate_args(self) -> None:
        """Validate constructor arguments and fail fast on incompatible combinations."""
        args = self.args

        # --- SageMaker Model Parallel mixed-precision validation ---
        if is_sagemaker_mp_enabled():
            if args.bf16:
                raise ValueError("SageMaker Model Parallelism does not support BF16 yet. Please use FP16 instead ")
            if args.fp16 != smp.state.cfg.fp16:
                logger.warning(
                    f"FP16 provided in SM_HP_MP_PARAMETERS is {smp.state.cfg.fp16}, "
                    f"but FP16 provided in trainer argument is {args.fp16}, "
                    f"setting to {smp.state.cfg.fp16}"
                )
                args.fp16 = smp.state.cfg.fp16

        # --- Training-argument validations ---
        if args.batch_eval_metrics and self.compute_metrics is not None:
            if "compute_result" not in inspect.signature(self.compute_metrics).parameters:
                raise ValueError(
                    "When using `batch_eval_metrics`, your `compute_metrics` function must take a `compute_result`"
                    " boolean argument which will be triggered after the last batch of the eval set to signal that the"
                    " summary statistics should be returned by the function."
                )
        if args.eval_strategy is not None and args.eval_strategy != "no" and self.eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but you didn't pass an `eval_dataset` to `Trainer`. Either set `args.eval_strategy` to `no` or pass an `eval_dataset`. "
            )
        if args.save_strategy == SaveStrategy.BEST or args.load_best_model_at_end:
            if args.metric_for_best_model is None:
                raise ValueError(
                    "`args.metric_for_best_model` must be provided when using 'best' save_strategy or if `args.load_best_model_at_end` is set to `True`."
                )

        # --- Optimizer validations ---
        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError("Passing both `optimizers` and `optimizer_cls_and_kwargs` arguments is incompatible.")
        if self.model_init is not None and (self.optimizer is not None or self.lr_scheduler is not None):
            raise RuntimeError(
                "Passing a `model_init` is incompatible with providing the `optimizers` argument. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )
        if is_torch_xla_available() and self.optimizer is not None:
            for param in self.model.parameters():
                model_device = param.device
                break
            for param_group in self.optimizer.param_groups:
                if len(param_group["params"]) > 0:
                    optimizer_device = param_group["params"][0].device
                    break
            if model_device != optimizer_device:
                raise ValueError(
                    "The model and the optimizer parameters are not on the same device, which probably means you"
                    " created an optimizer around your model **before** putting on the device and passing it to the"
                    " `Trainer`. Make sure the lines `import torch_xla.core.xla_model as xm` and"
                    " `model.to(xm.xla_device())` is performed before the optimizer creation in your script."
                )
        if (self.is_fsdp_xla_enabled or self.is_fsdp_enabled) and (
            self.optimizer is not None or self.lr_scheduler is not None
        ):
            raise RuntimeError(
                "Passing `optimizers` is not allowed if PyTorch FSDP is enabled. "
                "You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method."
            )

        # --- Dataset validations ---
        if not callable(self.data_collator) and callable(getattr(self.data_collator, "collate_batch", None)):
            raise TypeError("The `data_collator` should be a simple callable (function, class with `__call__`).")
        if args.max_steps > 0 and args.num_train_epochs > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")
        if self.train_dataset is not None and not has_length(self.train_dataset) and args.max_steps <= 0:
            raise ValueError(
                "The train_dataset does not implement __len__, max_steps has to be specified. "
                "The number of steps needs to be known in advance for the learning rate scheduler."
            )

        if self.train_dataset is not None and isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            logger.info(
                f"The `train_sampling_strategy='{args.train_sampling_strategy}'` option is ignored when using an `IterableDataset`. "
                "Samplers cannot be used with IterableDataset as they require indexed access to the dataset."
            )

    def _build_accelerator_args(self, **kwargs) -> dict[str, Any]:
        """Helper method to build accelerator-specific keyword arguments."""
        args = {
            "mixed_precision": self.args.mixed_precision,
            "deepspeed_plugin": self.args.deepspeed_plugin,
        }
        args.update(kwargs)

        if self.args.ddp_find_unused_parameters is not None:
            find_unused = self.args.ddp_find_unused_parameters
        elif isinstance(self.model, PreTrainedModel):
            # find_unused_parameters breaks checkpointing as per
            # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            find_unused = not (self.model.is_gradient_checkpointing or self.args.gradient_checkpointing)
        else:
            find_unused = True

        ddp_kwargs = {"find_unused_parameters": find_unused}
        if self.args.ddp_bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
        if self.args.ddp_broadcast_buffers is not None:
            ddp_kwargs["broadcast_buffers"] = self.args.ddp_broadcast_buffers

        args["kwargs_handlers"] = [DistributedDataParallelKwargs(**ddp_kwargs)]

        # We defer compatibility checks to accelerator
        if self.args.parallelism_config is not None:
            min_accelerate_version = "1.12.0"
            if not is_accelerate_available(min_accelerate_version):
                raise ImportError(
                    f"ParallelismConfig requires accelerate>={min_accelerate_version}). Please upgrade accelerate to use this feature."
                )
            args["parallelism_config"] = self.args.parallelism_config

        if getattr(self.model, "tp_size", None) is not None and self.model.tp_size > 1:
            if self.args.parallelism_config is None:
                if is_accelerate_available("1.12.0"):
                    if self.args.parallelism_config is None:
                        from accelerate import ParallelismConfig

                        args["parallelism_config"] = ParallelismConfig(tp_size=self.model.tp_size)
                else:
                    raise ValueError("Requires accelerate>1.12.0 to use Tensor Parallelism.")
            elif args["parallelism_config"].tp_size != self.model.tp_size:
                args["parallelism_config"].tp_size = self.model.tp_size

        if is_accelerate_available("1.2.0"):
            # it we don't have the correct version, we will rely on env var instead that were set in TrainingArguments
            from accelerate.utils import TorchDynamoPlugin

            dynamo_plugin = TorchDynamoPlugin(
                backend=self.args.torch_compile_backend, mode=self.args.torch_compile_mode
            )
            args["dynamo_plugin"] = dynamo_plugin

        return args

    def create_accelerator_and_postprocess(self) -> None:
        """Create the accelerator and perform post-creation setup (FSDP, DeepSpeed, etc.)."""
        # We explicitly don't rely on the `Accelerator` to do gradient accumulation
        grad_acc_kwargs = {}
        if self.args.accelerator_config.gradient_accumulation_kwargs is not None:
            grad_acc_kwargs = self.args.accelerator_config.gradient_accumulation_kwargs

        # check if num_steps is attempted to be passed in gradient_accumulation_kwargs
        if "num_steps" in grad_acc_kwargs:
            if self.args.gradient_accumulation_steps > 1:
                # raise because we do not know which setting is intended.
                raise ValueError(
                    "The `AcceleratorConfig`'s `num_steps` is set but `gradient_accumulation_steps` is greater than 1 in the passed `TrainingArguments`"
                    "If using the passed `AcceleratorConfig` is desired, do not set the `TrainingArguments` `gradient_accumulation_steps`."
                )
            else:
                self.args.gradient_accumulation_steps = grad_acc_kwargs["num_steps"]
        else:
            grad_acc_kwargs["num_steps"] = self.args.gradient_accumulation_steps

        # Just making sure that gradient_state have the correct values passed.
        # We don't rely on `accumulate` from accelerate to set sync_gradients in gradient_state.
        # Rather, we do it ourselves by setting self.accelerator.gradient_state._set_sync_gradients.
        gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

        accelerator_config = self.args.accelerator_config.to_dict()

        # Extract dataloader config params from accelerator config
        dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params}
        )
        dataloader_config.data_seed = self.args.data_seed

        non_blocking = accelerator_config.pop("non_blocking")

        if non_blocking and not self.args.dataloader_pin_memory:
            logger.warning(
                "`non_blocking` is enabled but `dataloader_pin_memory` is not. For the best performance, it's recommended to enable both."
            )
        dataloader_config.non_blocking = non_blocking
        # this would have been updated above, no need for it anymore
        accelerator_config.pop("gradient_accumulation_kwargs")

        fsdp_plugin = None
        if self.args.fsdp_plugin_args is not None:
            from accelerate.utils import FullyShardedDataParallelPlugin

            fsdp_plugin = FullyShardedDataParallelPlugin(**self.args.fsdp_plugin_args)

        args = self._build_accelerator_args(
            dataloader_config=dataloader_config,
            fsdp_plugin=fsdp_plugin,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        # create accelerator object
        self.accelerator = Accelerator(**args)
        # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
        self.gather_function = self.accelerator.gather_for_metrics

        if "use_gather_object" in inspect.signature(self.gather_function).parameters:
            self.gather_function = functools.partial(
                self.gather_function, use_gather_object=self.args.eval_use_gather_object
            )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            for param in ["limit_all_gathers", "activation_checkpointing"]:
                setattr(fsdp_plugin, param, self.args.fsdp_config.get(param, getattr(fsdp_plugin, param)))
            if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
                raise ValueError(
                    "The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg "
                    "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic "
                    "when using FSDP."
                )

        if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
            propagate_args_to_deepspeed(self.accelerator, self.args)

        # `save_only_model` can't be used with DeepSpeed/FSDP along with `load_best_model_at_end`
        if (
            self.args.save_only_model
            and (self.is_deepspeed_enabled or self.is_fsdp_enabled)
            and self.args.load_best_model_at_end
        ):
            wrapper = "DeepSpeed" if self.is_deepspeed_enabled else "FSDP"
            raise ValueError(f"{wrapper} can't be used with `save_only_model` along with `load_best_model_at_end`.")

        # `auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3
        if (
            self.is_deepspeed_enabled
            and self.accelerator.state.deepspeed_plugin.zero_stage == 3
            and self.args.auto_find_batch_size
        ):
            raise ValueError(
                "`auto_find_batch_size` isn't supported yet with DeepSpeed Zero-3. Please consider using Zero-2, Zero-1, or FSDP"
            )
        if (
            self.args.save_only_model
            and self.is_fsdp_enabled
            and "SHARDED_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type)
        ):
            raise ValueError("save_only_model option is not compatible with FSDP state dict type 'SHARDED_STATE_DICT'")

    # ---- Data Loading ----

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=dataloader_key,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        return self._get_dataloader(
            dataset=test_dataset,
            description="test",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
        )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a [`~torch.utils.data.DataLoader`] by accessing its dataset. When
        dataloader.dataset does not exist or has no length, estimates as best it can
        """
        try:
            dataset = dataloader.dataset
            # Special case for IterableDatasetShard, we need to dig deeper
            if isinstance(dataset, IterableDatasetShard):
                return len(dataloader.dataset.dataset)
            return len(dataloader.dataset)
        except (NameError, AttributeError, TypeError):  # no dataset or length, estimate by length of dataloader
            return len(dataloader) * self.args.per_device_train_batch_size

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Callable[[Dataset], torch.utils.data.Sampler] | None = None,
        is_training: bool = False,
        dataloader_key: str | None = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        data_collator = self.data_collator
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

        # MPS requrires forking if multiple workers are specified
        should_fork = torch.backends.mps.is_available() and self.args.dataloader_num_workers > 1

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "multiprocessing_context": "fork" if should_fork else None,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: str | None = None
    ) -> "datasets.Dataset":
        """Remove dataset columns not accepted by the model's forward method."""
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, "
                " you can safely ignore this message."
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if len(columns) == 0:
            raise ValueError(
                f"No columns in the dataset match the model's forward method signature: ({', '.join(signature_columns)}). "
                f"The following columns have been ignored: [{', '.join(ignored_columns)}]. "
                "Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`."
            )

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    # ---- Optimizer & Scheduler & Learning rate ----

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method (or `create_optimizer` and/or
        `create_scheduler`) in a subclass.
        """
        self.create_optimizer()
        self.create_scheduler(num_training_steps=num_training_steps)

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.

        Returns:
            `torch.optim.Optimizer`: The optimizer instance.
        """
        opt_model = self.model if model is None else model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Check if this is a factory (for complex optimizers like Muon, Dion)
            # Factories are instantiated first, then called with (opt_model, **kwargs)
            if is_optimizer_factory(optimizer_cls):
                self.optimizer = optimizer_cls()(opt_model, **optimizer_kwargs)
            else:
                # Standard optimizer class instantiation
                # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for GaLore optimizer.
                if "params" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("params")

                # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
                # e.g. for LOMO optimizer.
                if "model" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("model")

                # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
                # to avoid arguments conflicts.
                if "optimizer_dict" in optimizer_kwargs:
                    optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if "bitsandbytes" in str(optimizer_cls) and optimizer_kwargs.get("optim_bits", None) == 8:
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.

        Returns:
            `torch.optim.lr_scheduler.LRScheduler`: The learning rate scheduler instance.
        """
        if self.lr_scheduler is None:
            if optimizer is None:
                if is_sagemaker_mp_enabled() and smp.state.cfg.fp16:
                    # If fp16 is enabled, we unwrap the optimizer
                    optimizer = self.optimizer.optimizer
                else:
                    optimizer = self.optimizer
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments, model: PreTrainedModel | None = None) -> tuple[Any, Any]:
        """
        Returns the optimizer class and optimizer parameters based on the training arguments.

        Args:
            args (`transformers.training_args.TrainingArguments`):
                The training arguments for the training session.
            model (`PreTrainedModel`, *optional*):
                The model being trained. Required for some optimizers (GaLore, Apollo, LOMO).

        Returns:
            A tuple containing the optimizer class and a dictionary of optimizer keyword arguments.
        """
        ctx = OptimizerContext(
            args=args,
            model=model,
            optimizer_kwargs={"lr": args.learning_rate},
            adam_kwargs={
                "betas": (args.adam_beta1, args.adam_beta2),
                "eps": args.adam_epsilon,
            },
            optim_args=_parse_optim_args(args.optim_args),
        )

        handler = _OPTIMIZER_HANDLERS.get(args.optim)
        if handler is None:
            raise ValueError(f"Trainer cannot instantiate unsupported optimizer: {args.optim}")

        return handler(ctx)

    def get_decay_parameter_names(self, model: nn.Module) -> list[str]:
        """
        Get all parameter names that weight decay will be applied to.

        This function filters out parameters in two ways:
        1. By layer type (instances of layers specified in ALL_LAYERNORM_LAYERS)
        2. By parameter name patterns (containing 'bias', or variation of 'norm')
        """
        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(model, [nn.LayerNorm], forbidden_name_patterns)
        return decay_parameters

    def _get_learning_rate(self) -> float:
        """
        Returns the current learning rate from the scheduler.

        Handles DeepSpeed's dynamic loss scaling warmup period where `get_last_lr` may fail.
        """
        if self.is_deepspeed_enabled:
            # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
            # not run for the first few dozen steps while loss scale is too large, and thus during
            # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
            try:
                last_lr = self.lr_scheduler.get_last_lr()[0]
            except AssertionError as e:
                if "need to call step" in str(e):
                    logger.warning("tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                    last_lr = 0
                else:
                    raise
        else:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                last_lr = self.optimizer.param_groups[0]["lr"]
            else:
                last_lr = self.lr_scheduler.get_last_lr()[0]

        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
        return last_lr

    # ---- Training ----

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        trial: "optuna.Trial | dict[str, Any] | None" = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`list[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.

        Returns:
            [`~trainer_utils.TrainOutput`]: Object containing the global step count, training loss, and metrics.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(args.seed) if args.full_determinism else set_seed(args.seed)
            self.model = self.call_model_init(trial)
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        if self.args.use_liger_kernel:
            apply_liger_kernel(self.model, self.args.liger_kernel_config)

        # When fp16/bf16 full eval is enabled, __init__ skips device placement so that
        # evaluation_loop can cast dtype and move in one step. Move the model now for training.
        if (args.fp16_full_eval or args.bf16_full_eval) and not self.is_model_parallel and self.model_init is None:
            self._move_model_to_device(self.model, args.device)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        # If the model uses a tokenizer, it may have a new tokens for fine-tuning purposes.
        if isinstance(self.processing_class, (PreTrainedTokenizerBase, ProcessorMixin)) and hasattr(
            self.model, "config"
        ):
            align_special_tokens(self.model, self.processing_class)

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.neftune_hook_handle = activate_neftune(self.model, self.neftune_noise_alpha, self.accelerator)

        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)

        if DebugOption.UNDERFLOW_OVERFLOW in args.debug:
            if args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP with torchrun"
                )
            else:
                DebugUnderflowOverflow(self.model)

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({args.output_dir})")

        if resume_from_checkpoint is not None:
            # Load model checkpoint before accelerator.prepare() for regular models,
            # so that buffers and parameters are on the right device after prepare.
            # Deepspeed/FSDP models are loaded after prepare in _prepare_for_training.
            if not is_sagemaker_mp_enabled() and not self.is_deepspeed_enabled and not self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint)
            state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            if state.train_batch_size is not None and args.auto_find_batch_size:
                # Only restore the checkpoint's train_batch_size when using auto_find_batch_size,
                self._train_batch_size = state.train_batch_size

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
        ctx = suppress_progress_bars() if args.push_to_hub else contextlib.nullcontext()
        with ctx:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _inner_training_loop(
        self,
        batch_size: int | None = None,
        args: TrainingArguments | None = None,
        resume_from_checkpoint: str | None = None,
        trial: "optuna.Trial | dict[str, Any] | None" = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Run the actual training loop: forward, backward, optimizer step, logging, and checkpointing."""
        # reset everything
        self.accelerator.free_memory()
        if args.auto_find_batch_size:
            self._update_auto_batch_size(batch_size)
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader)

        epochs_trained, steps_trained_in_current_epoch = self._init_training_state(
            max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
        )
        model, train_dataloader = self._prepare_for_training(max_steps, train_dataloader, resume_from_checkpoint)

        # Train!
        base_logger = getattr(logger, "_logger", logger)
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Num update steps per epoch = {num_update_steps_per_epoch:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        if resume_from_checkpoint is not None:
            logger.info(
                f"  Resuming training from checkpoint with epoch {epochs_trained} and global step {self.state.global_step}"
            )
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Fast-forwarding the dataloader past {epochs_trained} epochs and"
                    f" {steps_trained_in_current_epoch} batches to resume from the exact training state."
                )

        start_time = time.time()
        # needed to calculate tokens/s
        self._initial_num_input_tokens_seen = self.state.num_input_tokens_seen
        # Logging state: _tr_loss accumulates on-device between logging steps (avoiding costly .item() syncs
        # on TPUs), then gets drained into _total_loss_scalar at each logging step.
        self._tr_loss = torch.tensor(0.0, device=args.device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step

        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        for epoch in range(epochs_trained, num_train_epochs):
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)
            self._run_epoch(
                model=model,
                epoch=epoch,
                train_dataloader=train_dataloader,
                steps_in_epoch=steps_in_epoch,
                num_update_steps_per_epoch=num_update_steps_per_epoch,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
                start_time=start_time,
                resume_from_checkpoint=resume_from_checkpoint,
                epochs_trained=epochs_trained,
                steps_trained_in_current_epoch=steps_trained_in_current_epoch,
            )
            if self.control.should_training_stop:
                break

        return self._finalize_training(trial, num_train_samples, start_time)

    def _init_training_state(
        self, max_steps, num_update_steps_per_epoch, num_train_epochs, resume_from_checkpoint, trial
    ) -> tuple[int, int]:
        """Initialize TrainerState, optionally restoring from checkpoint. Returns (epochs_trained, steps_trained_in_current_epoch)."""
        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size
        self.state.compute_steps(self.args, max_steps)

        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        return epochs_trained, steps_trained_in_current_epoch

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        """Wrap model, create optimizer and scheduler, and run accelerator.prepare. Returns (model, train_dataloader)."""
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2)
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer()

        # Pass `self.model_wrapped` so that `_wrap_model` can detect if the model is already
        # wrapped (e.g. in DataParallel) on subsequent `train()` calls and avoid double wrapping.
        model = self._wrap_model(self.model_wrapped)

        # If the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases in accelerate such as FSDP-XLA, SageMaker MP/DP, DataParallel
        use_accelerator_prepare = model is self.model

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            if delay_optimizer_creation:
                # TODO: check if we can move this somewhere else
                if self.is_fsdp_enabled and _is_peft_model(self.model):
                    update_fsdp_plugin_peft(self.model, self.accelerator)
                # we only prepare the model as we don't have an optimizer
                model = self.accelerator.prepare(self.model)
                # using the model we prepared to create the optimizer
                self.create_optimizer(model)
                self.optimizer = self.accelerator.prepare(self.optimizer)
            elif self.is_deepspeed_enabled and type(self.lr_scheduler).__name__ == "DummyScheduler":
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        # Create scheduler now that the optimizer won't change anymore
        self.create_scheduler(num_training_steps=max_steps)

        # updating self.model_wrapped
        self.model_wrapped = model

        if self.is_fsdp_enabled or self.is_fsdp_xla_enabled:
            # breaking convention for FSDP model
            # TODO: check if this is really needed
            self.model = self.model_wrapped = model

        # backward compatibility
        # TODO: check if we really need this
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # Important: at this point:
        # self.model         is the Transformers Model except when we are using FSDP
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        if self.is_fsdp_enabled:
            # Fix `got mixed torch.Tensor and DTensor` error in model.generate() for FSDP2 with LoRA
            if hasattr(self.model, "generate"):
                dist.fsdp.register_fsdp_forward_method(self.model, "generate")

        # since DataLoader was Accelerate prepared w/o a model arg in the same call, we now have to complete the DL wrapping for ALST/UlyssesSP, after model has been prepared
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled:
            train_dataloader = self.accelerator.deepspeed_ulysses_dl_adapter(train_dataloader, model)

        # load checkpoint
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            self._load_optimizer_and_scheduler(resume_from_checkpoint)
            self._load_scaler(resume_from_checkpoint)

        # Update the references for the callback_handler
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        return model, train_dataloader

    def _run_epoch(
        self,
        model,
        epoch,
        train_dataloader,
        steps_in_epoch,
        num_update_steps_per_epoch,
        trial,
        ignore_keys_for_eval,
        start_time,
        resume_from_checkpoint,
        epochs_trained,
        steps_trained_in_current_epoch,
    ):
        """Run one full pass over the dataloader."""

        step = -1
        grad_norm = None
        learning_rate = None
        rng_to_sync = False

        # Handle resumption from checkpoint: skip already-trained batches in the resumed epoch
        num_update_steps_trained = 0
        if epoch == epochs_trained and resume_from_checkpoint is not None:
            if steps_trained_in_current_epoch > 0 and not self.args.ignore_data_skip:
                train_dataloader = skip_first_batches(train_dataloader, steps_trained_in_current_epoch)
                step = steps_trained_in_current_epoch - 1
                num_update_steps_trained = steps_trained_in_current_epoch // self.args.gradient_accumulation_steps
                rng_to_sync = True
            elif steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        epoch_iterator = iter(train_dataloader)

        # We chunkify the epoch iterator into gradient accumulation steps `n` batches
        remainder = steps_in_epoch % self.args.gradient_accumulation_steps
        if remainder == 0:
            remainder = self.args.gradient_accumulation_steps

        # Outer loop: one iteration per optimizer step. Each iteration prefetches
        # `gradient_accumulation_steps` batches (fewer for the last step if the epoch
        # doesn't divide evenly).
        for update_step in range(num_update_steps_trained, num_update_steps_per_epoch):
            num_batches = (
                self.args.gradient_accumulation_steps if update_step != (num_update_steps_per_epoch - 1) else remainder
            )
            batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, self.args.device)

            # This is used to correctly scale the loss when the last accumulation step has fewer batches.
            # Not used if `num_items_in_batch` is not None.
            self.current_gradient_accumulation_steps = len(batch_samples)

            # need to sync after if we skipped the batches in `get_batch_samples` for shuffle order reason
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False

            # Inner loop: forward + backward for each micro-batch. Gradients are
            # accumulated without syncing until the last micro-batch, then we clip,
            # step the optimizer, and log/save/evaluate.
            for i, inputs in enumerate(batch_samples):
                step += 1
                do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                # Since we perform prefetching, we need to manually set sync_gradients
                self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                if step % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                # We sync the gradients in the following cases: 1. sync_each_batch set to True 2. Using deepspeed 3. when we are at the last batch sample
                if (
                    self.accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False)
                    or self.accelerator.distributed_type == DistributedType.DEEPSPEED
                    or i == len(batch_samples) - 1
                ):
                    sync_context = contextlib.nullcontext
                else:
                    sync_context = functools.partial(self.accelerator.no_sync, model=model)
                
                with sync_context():
                    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)

                if (
                    self.args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    self._tr_loss += self._tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if self._tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {self._tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    self._tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))
                self._track_num_input_tokens(inputs)

                if do_sync_step:
                    grad_norm = None
                    if self.args.max_grad_norm > 0:
                        grad_norm = self._clip_grad_norm(model)
                    grad_norm = self._get_grad_norm(model, grad_norm=grad_norm)

                    self.control = self.callback_handler.on_pre_optimizer_step(self.args, self.state, self.control)
                    self.optimizer.step()
                    self.control = self.callback_handler.on_optimizer_step(self.args, self.state, self.control)

                    # get leaning rate before update
                    learning_rate = self._get_learning_rate()

                    if not self.accelerator.optimizer_step_was_skipped:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)
                    self._maybe_log_save_evaluate(
                        self._tr_loss,
                        grad_norm,
                        model,
                        trial,
                        epoch,
                        ignore_keys_for_eval,
                        start_time,
                        learning_rate=learning_rate,
                    )
                else:
                    self.control = self.callback_handler.on_substep_end(self.args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # PyTorch/XLA relies on the dataloader to insert mark_step each iteration.
        # When we break out of the loop early, we flush the pending graph manually.
        if is_torch_xla_available():
            xm.mark_step()

        if step < 0:
            logger.warning(
                "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                f" num_steps ({self.state.max_steps}) higher than the number of available samples."
            )
            self.control.should_training_stop = True

        self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
        self._maybe_log_save_evaluate(
            self._tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate=learning_rate,
        )

    def _finalize_training(self, trial, num_train_samples, start_time):
        """Finalize training: metrics, best-model loading, cleanup. Returns TrainOutput."""
        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")

        # add remaining tr_loss
        self._total_loss_scalar += self._tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self._memory_tracker.stop_and_update_metrics(metrics)
        self.log(metrics)

        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            self._load_best_model()

        checkpoints_sorted = sort_checkpoints(
            output_dir=self._get_output_dir(trial), best_model_checkpoint=self.state.best_model_checkpoint
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            deactivate_neftune(self.model, self.neftune_hook_handle, self.accelerator)
        self.is_in_train = False

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # Prepare buffers for context parallelism
        cp_context, inputs = self._prepare_context_parallel_inputs(model, inputs)

        # Context manager is no-op if CP isn't enabled
        with cp_context():
            model.train()
            if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
                self.optimizer.train()

            inputs = self._prepare_inputs(inputs)
            
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            del inputs
            if (
                self.args.torch_empty_cache_steps is not None
                and self.state.global_step % self.args.torch_empty_cache_steps == 0
            ):
                clear_device_cache()

            kwargs = {}

            # For LOMO optimizers you need to explicitly use the learning rate
            if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
                kwargs["learning_rate"] = self._get_learning_rate()

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
                # If the model does not accept loss kwargs, we need to normalize the loss by the number of gradient accumulation steps
                loss = loss / self.current_gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()

    def compute_loss_context_manager(self) -> contextlib.ExitStack:
        """
        A helper wrapper to group together context managers.
        """
        ctx_stack = contextlib.ExitStack()

        autocast_ctx = self.autocast_smart_context_manager()
        if not isinstance(autocast_ctx, contextlib.nullcontext):
            ctx_stack.enter_context(autocast_ctx)

        return ctx_stack

    def autocast_smart_context_manager(self, cache_enabled: bool | None = True) -> contextlib.AbstractContextManager:
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here.
        """
        return contextlib.nullcontext()

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: nn.Module,
        trial: "optuna.Trial | dict[str, Any] | None",
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        """Log metrics, run evaluation, and save checkpoints if the current training state requires it."""
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = nested_gather(tr_loss, self.args.parallel_mode).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # ---- Training Utilites ----
    def get_batch_samples(
        self, epoch_iterator: Iterator, num_batches: int, device: torch.device
    ) -> tuple[list, torch.Tensor | int | None]:
        """
        Collects a specified number of batches from the epoch iterator and optionally counts the number of items in the batches to properly scale the loss.
        """
        batch_samples = []

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        num_items_in_batch = self._get_num_items_in_batch(batch_samples, device)
        return batch_samples, num_items_in_batch

    def _get_num_items_in_batch(self, batch_samples: list, device: torch.device) -> torch.Tensor | int | None:
        """
        Counts the number of items in the batches to properly scale the loss.
        Args:
            batch_samples (`list`): List of batches
            device (`torch.device`): The device on which the number of items in the batch should be.
        Returns:
            None if the number of items in the batch doesn't need to be computed else the number of items in the batch
        """
        num_items_in_batch = None
        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )
        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum((batch["labels"].ne(-100)).sum() for batch in batch_samples)
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                if self.args.world_size > 1:
                    num_items_in_batch = self.accelerator.gather(num_items_in_batch.to(device)).sum()
            elif self.args.n_gpu > 1:
                # In DP case, if we don't average, we need to divide by the number of gpu. This is the simplest approximation.
                # Otherwise, we would have to scatter labels and calculate num_items_in_batch for each gpu.
                num_items_in_batch = num_items_in_batch // self.args.n_gpu

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 2-dim tensor with the same value repeated
                    num_items_in_batch = num_items_in_batch.unsqueeze(0).expand(self.args.n_gpu, -1)
                # Divide by number of devices with the same batch
                if pc := getattr(self.accelerator, "parallelism_config", None):
                    num_items_in_batch = num_items_in_batch // pc.non_data_parallel_size

        return num_items_in_batch

    def _prepare_input(self, data: torch.Tensor | Any) -> torch.Tensor | Any:
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
                # NLP models inputs are int/uint and those get adjusted to the right dtype of the
                # embedding. Other models such as wav2vec2's inputs are already float and thus
                # may need special handling to match the dtypes of the model
                kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )

        return inputs

    def _prepare_context_parallel_inputs(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any]
    ) -> tuple[Callable, dict[str, torch.Tensor | Any]]:
        """
        Prepare inputs for context parallelism by setting up buffers and validation.

        Args:
            model: The model being trained
            inputs: Input tensors to prepare

        Returns:
            tuple: (context_manager, prepared_inputs) where context_manager is either
                   the context parallelism wrapper or a no-op context
        """
        if (
            getattr(self.accelerator, "parallelism_config", None) is not None
            and self.accelerator.parallelism_config.cp_enabled
        ):
            if self.accelerator.parallelism_config.cp_backend == "torch":
                if hasattr(model, "config"):
                    if model.config._attn_implementation != "sdpa":
                        raise ValueError(
                            f"Context parallelism is supported only with SDPA attention, you are using {model.config._attn_implementation}."
                        )

                if "shift_labels" not in inputs:
                    logger.warning_once("Shift labels not found in the inputs, shifting manually")
                    if "labels" in inputs:
                        _ignore_index = -100
                        labels = nn.functional.pad(inputs["labels"], (0, 1), value=_ignore_index)
                        inputs["shift_labels"] = labels[:, 1:].contiguous()

            # note: we don't do anything for accelerator.parallelism_config.sp_backend == "deepspeed" since:
            # - accelerator.parallelism_config performs the `model.config._attn_implementation` checks already and it supports more than `dspa`
            # - UlyssesSPDataLoaderAdapter called from Accelerate performs the `shift_label` creation - must not interfere
            # - position_ids generation should be done by HF Trainer if it wasn't done by the user

            if "position_ids" not in inputs:
                logger.warning_once("Position IDs not found in the inputs, generating manually")
                inputs["position_ids"] = torch.arange(
                    inputs["input_ids"].size(1), device=inputs["input_ids"].device
                ).expand(inputs["input_ids"].size(0), -1)

            buffers = []
            buffer_seq_dims = []

            if "input_ids" in inputs:
                buffers.append(inputs["input_ids"])
                buffer_seq_dims.append(1)  # Sequence dimension
            if "labels" in inputs:
                buffers.append(inputs["labels"])
                buffer_seq_dims.append(1)
            if "shift_labels" in inputs:
                buffers.append(inputs["shift_labels"])
                buffer_seq_dims.append(1)
            # Add attention_mask to buffers for context parallel splitting (only if causal)
            if "attention_mask" in inputs:
                # Only validate causal mask once for performance
                if not getattr(self, "_attn_mask_causal_checked", False):
                    # Context parallel currently doesn't support other masks than causal
                    # Accelerate applies hooks to replace mask with is_causal arg in SDPA
                    # Check if the mask is really causal and if not throw an error
                    attention_mask = inputs["attention_mask"]
                    if not is_attention_mask_causal(attention_mask):
                        raise ValueError(
                            "Context parallelism only supports causal attention masks. "
                            "The provided attention_mask is not causal. "
                            "Please ensure your data uses causal masking (lower triangular) "
                            "or remove the attention_mask to use the model's default causal masking."
                        )
                    self._attn_mask_causal_checked = True
                if self._attn_mask_causal_checked:
                    # Add to buffers only after validation (or if validation already passed)
                    attention_mask = inputs["attention_mask"]
                    if attention_mask.dim() == 2:
                        buffers.append(attention_mask)
                        buffer_seq_dims.append(1)
                    else:
                        # Other dimensionality; keep as-is without sharding to avoid incorrect splits
                        pass
            # Include position_ids in context parallelism splitting
            if "position_ids" in inputs and inputs["position_ids"] is not None:
                buffers.append(inputs["position_ids"])
                buffer_seq_dims.append(1)

            return partial(
                self.accelerator.maybe_context_parallel,
                buffers=buffers,
                buffer_seq_dims=buffer_seq_dims,
                no_restore_buffers=set(buffers),
            ), inputs

        return contextlib.nullcontext, inputs

    def set_initial_training_values(
        self, args: TrainingArguments, dataloader: DataLoader
    ) -> tuple[int, int, int, int, int, int | None, int]:
        """
        Calculates and returns the following values:
        - `num_train_epochs`
        - `num_update_steps_per_epoch`
        - `num_examples`
        - `num_train_samples`
        - `total_train_batch_size`
        - `steps_in_epoch` (total batches per epoch)
        - `max_steps`
        """
        # Case 1: we rely on `args.max_steps` first
        max_steps = args.max_steps
        # If max_steps is negative, we use the number of epochs to determine the number of total steps later
        epoch_based = max_steps < 0
        len_dataloader = len(dataloader) if has_length(dataloader) else None
        total_train_batch_size = self.get_total_train_batch_size(args)

        # Account for Sequence Parallelism (SP) dataloader adapter's effect
        sp_size = self.get_sp_size()
        if sp_size > 1 and len_dataloader is not None:
            len_dataloader = len_dataloader * sp_size

        # Case 2: We have a dataloader length and can extrapolate
        if len_dataloader is not None:
            num_update_steps_per_epoch = max(
                len_dataloader // args.gradient_accumulation_steps
                + int(len_dataloader % args.gradient_accumulation_steps > 0),
                1,
            )
            # Case 3: We have a length but are using epochs, we can extrapolate the number of steps
            if epoch_based:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
        # Now we figure out `num_examples`, `num_train_epochs`, and `train_samples`
        if len_dataloader:
            num_examples = self.num_examples(dataloader)
            if args.max_steps > 0:
                num_train_epochs = max_steps // num_update_steps_per_epoch + int(
                    max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = max_steps * total_train_batch_size
            else:
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )
        steps_in_epoch = len_dataloader if len_dataloader is not None else max_steps * args.gradient_accumulation_steps
        return (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            total_train_batch_size,
            steps_in_epoch,
            max_steps,
        )

    def get_total_train_batch_size(self, args: TrainingArguments) -> int:
        """Calculates total batch size (micro_batch * grad_accum * dp_world_size).

        Accounts for all parallelism dimensions: TP, CP, and SP.

        Formula: dp_world_size = world_size // (tp_size * cp_size * sp_size)

        Where:
        - TP (Tensor Parallelism): Model layers split across GPUs
        - CP (Context Parallelism): Sequences split using Ring Attention (FSDP2)
        - SP (Sequence Parallelism): Sequences split using ALST/Ulysses (DeepSpeed)

        All dimensions are separate and multiplicative: world_size = dp_size * tp_size * cp_size * sp_size
        """
        dp_world_size = args.world_size // self.get_tp_size() // self.get_cp_size() // self.get_sp_size()
        return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size

    def get_sp_size(self) -> int:
        """Get the sequence parallel size"""
        if getattr(self.accelerator, "parallelism_config", None) is None:
            return 1
        else:
            pc = self.accelerator.parallelism_config
            return pc.sp_size

    def get_cp_size(self) -> int:
        """Get the context parallel size"""
        if getattr(self.accelerator, "parallelism_config", None) is None:
            return 1
        else:
            pc = self.accelerator.parallelism_config
            return pc.cp_size

    def get_tp_size(self) -> int:
        """Get the tensor parallel size from either the model or DeepSpeed config."""

        # 1. Check model.tp_size first
        if (model_tp := getattr(self.model, "_tp_size", None)) is not None:
            return model_tp

        # 2. Fall back to DeepSpeed config if enabled
        if self.is_deepspeed_enabled and (deepspeed_config := getattr(self.args, "hf_deepspeed_config", None)):
            return deepspeed_config.config.get("tensor_parallel", {}).get("autotp_size", 1)

        # 3. Default fallback
        return 1

    def _wrap_model(self, model: nn.Module, training: bool = True, dataloader: DataLoader | None = None) -> nn.Module:
        """Wrap `model` for distributed training if needed (DDP, FSDP, SageMaker, etc.)."""
        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if self.accelerator.unwrap_model(model, keep_torch_compile=False) is not model:
            return model

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(model, smp.model.DistributedModel):
                return model
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # Multi-gpu training, 8bit models does not support DP
        if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
            model = nn.DataParallel(model)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training using PyTorch FSDP
        if self.is_fsdp_xla_enabled:
            model = wrap_model_xla_fsdp(model, self.args, self.is_fsdp_xla_v2_enabled)
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        return model

    def _update_auto_batch_size(self, batch_size):
        """Free memory, reset model wrapping, and update DeepSpeed config for the new batch size when using `auto_find_batch_size`"""
        # `_train_batch_size` value might have changed to `auto_find_batch_size`
        self._train_batch_size = batch_size
        # frees the wrapped model and resets it back to the unwrapped base model
        release_memory(self.model_wrapped)

        if self.is_fsdp_enabled:
            # Remove FSDP wrapping from sub-models because self.model points to the wrapped model in FSDP case
            self.model = unwrap_model(self.model, recursive=True)

        self.model_wrapped = self.model

        # Check for DeepSpeed *after* the initial pass and modify the config
        if self.is_deepspeed_enabled:
            # Temporarily unset `self.args.train_batch_size`
            original_bs = self.args.per_device_train_batch_size
            self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
            propagate_args_to_deepspeed(self.accelerator, self.args, auto_find_batch_size=True)
            self.args.per_device_train_batch_size = original_bs

    def _track_num_input_tokens(self, inputs):
        """Count input tokens seen (all or non-padding) and update state."""
        if self.args.include_num_input_tokens_seen == "no":
            return
        main_input_name = getattr(self.model, "main_input_name", "input_ids")
        if main_input_name not in inputs:
            logger.warning(
                "Tried to track the number of tokens seen, however the current model is "
                "not configured properly to know what item is the input. To fix this, add "
                "a `main_input_name` attribute to the model class you are using."
            )
            return

        if self.args.include_num_input_tokens_seen == "non_padding":
            if "attention_mask" in inputs:
                input_tokens = inputs["attention_mask"].sum()
            elif (
                self.processing_class is not None
                and hasattr(self.processing_class, "pad_token_id")
                and self.processing_class.pad_token_id is not None
            ):
                input_tokens = (inputs[main_input_name] != self.processing_class.pad_token_id).sum()
            else:
                logger.warning(
                    "Could not determine method to count non-padding tokens, falling back to counting all tokens."
                )
                input_tokens = inputs[main_input_name].numel()
        else:
            input_tokens = inputs[main_input_name].numel()

        input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
        self.state.num_input_tokens_seen += self.accelerator.gather(input_tokens).sum().item()

    def _clip_grad_norm(self, model):
        """Clip gradients to max_grad_norm. Returns the pre-clip gradient norm."""
        if is_sagemaker_mp_enabled() and self.args.fp16:
            return self.optimizer.clip_master_grads(self.args.max_grad_norm)
        return self.accelerator.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

    def _get_grad_norm(self, model, grad_norm=None):
        """Return the gradient norm as a Python float."""
        if grad_norm is None:
            # Compute norm without clipping (inf means no actual clipping happens)
            grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), float("inf"))

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            if hasattr(grad_norm, "item"):
                grad_norm = grad_norm.item()
        return grad_norm

    # ---- Evaluation & Prediction ----

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset` | dict[str, `Dataset`], *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                `__len__` method.

                <Tip>

                If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                separate evaluations on each dataset. This can be useful to monitor how training affects other
                datasets or simply to get a more fine-grained evaluation.
                When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                loss on `data1` and `metric_for_best_model="eval_data2_loss"` for the loss on `data2`.

                </Tip>

            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        output = self.evaluation_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs)
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    def predict(
        self, test_dataset: Dataset, ignore_keys: list[str] | None = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        output = self.evaluation_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, label_ids=output.label_ids, metrics=output.metrics)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`list[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss")
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = len(self.label_names) == 0 and return_loss

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", ["past_key_values"])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                        )
                    loss = loss.detach().mean()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    def _evaluate(
        self,
        trial: "optuna.Trial | dict[str, Any] | None",
        ignore_keys_for_eval: list[str] | None,
        skip_scheduler: bool = False,
    ) -> dict[str, float]:
        """Run evaluation, report to HP search, and step ReduceLROnPlateau if needed."""
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', "
                    f"which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. "
                    f"Please ensure that the `compute_metrics` function returns a dictionary that includes '{metric_to_check}' or "
                    f"consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics

    # ---- Checkpoint Saving ----

    def _get_output_dir(self, trial: "optuna.Trial | dict[str, Any] | None") -> str:
        """Return the output directory, accounting for hyperparameter search trials."""
        if self.hp_search_backend is not None and trial is not None:
            if self.hp_search_backend == HPSearchBackend.OPTUNA:
                run_id = trial.number
            elif self.hp_search_backend == HPSearchBackend.RAY:
                import ray.tune

                run_id = ray.tune.get_context().get_trial_id()
            elif self.hp_search_backend == HPSearchBackend.WANDB:
                import wandb

                run_id = wandb.run.id
            run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
            run_dir = os.path.join(self.args.output_dir, run_name)
        else:
            run_dir = self.args.output_dir
        return run_dir

    def _save_checkpoint(self, model: nn.Module, trial: "optuna.Trial | dict[str, Any] | None") -> None:
        """Save model checkpoint, optimizer, scheduler, scaler, RNG states, and trainer state."""
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)

        if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
            # Wait for everyone to get here so we are sure the model has been saved by process 0
            # before we check if the best_checkpoint_dir exists
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
            best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

            if os.path.exists(best_checkpoint_dir):
                self.state.best_model_checkpoint = best_checkpoint_dir

        if not self.args.save_only_model:
            # Save optimizer and scheduler
            self._save_optimizer_and_scheduler(output_dir)
            self._save_scaler(output_dir)
            # Save RNG state
            self._save_rng_state(output_dir)

        # Save the Trainer state
        if self.args.should_save:
            # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
            for cb in [
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]:
                cb_name = cb.__class__.__name__
                cb_state = cb.state()
                if isinstance(self.state.stateful_callbacks[cb_name], list):
                    self.state.stateful_callbacks[cb_name].append(cb_state)
                else:
                    self.state.stateful_callbacks[cb_name] = cb_state
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            # we use mtime as default, filesystems without mtime support will be detected in `sort_checkpoints`
            rotate_checkpoints(
                output_dir=run_dir,
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
                use_mtime=True,
            )

    def _determine_best_metric(self, metrics: dict[str, float], trial: "optuna.Trial | dict[str, Any] | None") -> bool:
        """
        Determine if the model should be saved based on the evaluation metrics.

        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model

            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

            if operator(metric_value, self.state.best_metric):
                self.state.best_metric = metric_value

                if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]:
                    self.state.best_global_step = self.state.global_step

                is_new_best_metric = True

        return is_new_best_metric

    def _save_rng_state(self, output_dir: str) -> None:
        """Save random number generator states for reproducible resumption."""
        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_xla_available():
            rng_states["xla"] = xm.get_rng_state()

        if is_torch_npu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["npu"] = torch.npu.random.get_rng_state_all()
            else:
                rng_states["npu"] = torch.npu.random.get_rng_state()

        if is_torch_hpu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["hpu"] = torch.hpu.random.get_rng_state_all()
            else:
                rng_states["hpu"] = torch.hpu.random.get_rng_state()

        if is_torch_mlu_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["mlu"] = torch.mlu.random.get_rng_state_all()
            else:
                rng_states["mlu"] = torch.mlu.random.get_rng_state()

        if is_torch_musa_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                rng_states["musa"] = torch.musa.get_rng_state_all()
            else:
                rng_states["musa"] = torch.musa.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        """Save optimizer and learning rate scheduler states to `output_dir`."""
        if is_torch_xla_available():
            xm.rendezvous("saving_optimizer_states")
            if self.is_fsdp_xla_v1_enabled:
                optm = {
                    "optimizer": self.optimizer.state_dict(),
                    "shard_metadata": self.model.get_shard_metadata(),
                }
                xm.save(
                    optm,
                    os.path.join(
                        output_dir, f"rank{self.args.process_index}-of-{self.args.world_size}-{OPTIMIZER_NAME}"
                    ),
                    master_only=False,
                )
            else:
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
        elif self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
            )
            if accept_exclude_frozen_parameters and _is_peft_model(self.model):
                self.model_wrapped.save_checkpoint(output_dir, exclude_frozen_parameters=True)
            else:
                self.model_wrapped.save_checkpoint(output_dir)
        elif self.is_fsdp_enabled:
            # save fsdp specific ckpt for resuming from ckpt
            save_fsdp_model(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.model, output_dir, **get_fsdp_ckpt_kwargs()
            )
            save_fsdp_optimizer(
                self.accelerator.state.fsdp_plugin, self.accelerator, self.optimizer, self.model, output_dir
            )
        elif self.args.should_save:
            # deepspeed.save_checkpoint above saves model/optim/sched
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        # Save SCHEDULER & SCALER
        is_deepspeed_custom_scheduler = self.is_deepspeed_enabled and not isinstance(
            self.lr_scheduler, DeepSpeedSchedulerWrapper
        )
        if (
            self.args.should_save
            and (not self.is_deepspeed_enabled or is_deepspeed_custom_scheduler)
            and not is_torch_xla_available()
        ):
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

    def _save_scaler(self, output_dir: str) -> None:
        """Save the gradient scaler state if one exists."""
        # See if there is a scaler attribute
        try:
            scaler = self.accelerator.scaler
        except AttributeError:
            return
        if scaler is None:
            return
        if is_torch_xla_available():
            xm.rendezvous("saving_scaler_state")
            with warnings.catch_warnings(record=True) as caught_warnings:
                xm.save(self.accelerator.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
                reissue_pt_warnings(caught_warnings)

        # Save SCALER
        if self.args.should_save and not is_torch_xla_available():
            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.accelerator.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
            reissue_pt_warnings(caught_warnings)

    # ---- Checkpoint Resuming ----

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: nn.Module | None = None) -> None:
        """Load model weights from a checkpoint directory."""
        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PreTrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                smp.resume_from_checkpoint(
                    path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                )
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if os.path.isfile(safe_weights_file):
                    state_dict = safetensors.torch.load_file(safe_weights_file, device="cpu")
                else:
                    check_torch_load_is_safe()
                    state_dict = torch.load(weights_file, map_location="cpu", weights_only=True)

                # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                # which takes *args instead of **kwargs
                load_result = model.load_state_dict(state_dict, False)
                # release memory
                del state_dict
                self._issue_warnings_after_load(load_result)

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If training a model using PEFT, assume that adapter have been saved properly.
            if hasattr(model, "active_adapters") and hasattr(model, "load_adapter"):
                if os.path.exists(resume_from_checkpoint):
                    active_adapters = model.active_adapters
                    if len(active_adapters) > 1:
                        logger.warning("Multiple active adapters detected will only consider the first adapter")
                    active_adapter = active_adapters[0]

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning(f"Could not load adapter model, make sure to have PEFT >= {MIN_PEFT_VERSION} installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled())
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _load_best_model(self) -> None:
        """Load the best model found during training based on the tracked metric."""
        logger.info(f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        best_safe_model_path = os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_NAME)
        best_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_WEIGHTS_NAME)
        best_safe_adapter_model_path = os.path.join(self.state.best_model_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)

        model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(
                self.model_wrapped,
                self.state.best_model_checkpoint,
                load_module_strict=not _is_peft_model(self.model),
            )
        elif self.is_fsdp_enabled:
            load_result = load_fsdp_model(
                self.accelerator.state.fsdp_plugin,
                self.accelerator,
                model,
                self.state.best_model_checkpoint,
                **get_fsdp_ckpt_kwargs(),
            )
        elif (
            os.path.exists(best_model_path)
            or os.path.exists(best_safe_model_path)
            or os.path.exists(best_adapter_model_path)
            or os.path.exists(best_safe_adapter_model_path)
        ):
            has_been_loaded = True
            if is_sagemaker_mp_enabled():
                smp.resume_from_checkpoint(
                    path=self.state.best_model_checkpoint,
                    tag=WEIGHTS_NAME,
                    partial=False,
                    load_optimizer=False,
                )
            else:
                if _is_peft_model(model):
                    # If training a model using PEFT, assume that adapter have been saved properly.
                    if hasattr(model, "active_adapters") and hasattr(model, "load_adapter"):
                        active_adapter = model.active_adapters[0]
                        if len(model.active_adapters) > 1:
                            logger.warning("Detected multiple active adapters, will only consider the first one")

                        if os.path.exists(best_adapter_model_path) or os.path.exists(best_safe_adapter_model_path):
                            try:
                                model.load_adapter(self.state.best_model_checkpoint, active_adapter)
                            except RuntimeError as exc:
                                if model.peft_config[active_adapter].is_prompt_learning:
                                    # for context: https://github.com/huggingface/peft/issues/2256
                                    msg = (
                                        "When using prompt learning PEFT methods such as "
                                        f"{model.peft_config[active_adapter].peft_type.value}, setting "
                                        "load_best_model_at_end=True can lead to errors, it is recommended "
                                        "to set this to False and to load the model manually from the checkpoint "
                                        "directory using PeftModel.from_pretrained(base_model, <path>) after training "
                                        "has finished."
                                    )
                                    raise RuntimeError(msg) from exc
                                else:
                                    raise
                            # Load_adapter has no return value present, modify it when appropriate.
                            from torch.nn.modules.module import _IncompatibleKeys

                            load_result = _IncompatibleKeys([], [])
                        else:
                            logger.warning(
                                "The intermediate checkpoints of PEFT may not be saved correctly, "
                                f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                                "Check some examples here: https://github.com/huggingface/peft/issues/96"
                            )
                            has_been_loaded = False
                    else:
                        logger.warning(
                            f"Could not load adapter model, make sure to have PEFT >= {MIN_PEFT_VERSION} installed"
                        )
                        has_been_loaded = False
                else:
                    # We load the model state dict on the CPU to avoid an OOM error.
                    if os.path.isfile(best_safe_model_path):
                        state_dict = safetensors.torch.load_file(best_safe_model_path, device="cpu")
                    else:
                        check_torch_load_is_safe()
                        state_dict = torch.load(best_model_path, map_location="cpu", weights_only=True)

                    # If the model is on the GPU, it still works!
                    # workaround for FSDP bug https://github.com/pytorch/pytorch/issues/82963
                    # which takes *args instead of **kwargs
                    load_result = model.load_state_dict(state_dict, False)
                if not is_sagemaker_mp_enabled() and has_been_loaded:
                    self._issue_warnings_after_load(load_result)
        elif os.path.exists(os.path.join(self.state.best_model_checkpoint, SAFE_WEIGHTS_INDEX_NAME)) or os.path.exists(
            os.path.join(self.state.best_model_checkpoint, WEIGHTS_INDEX_NAME)
        ):
            load_result = load_sharded_checkpoint(
                model, self.state.best_model_checkpoint, strict=is_sagemaker_mp_enabled()
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)
        else:
            logger.warning(
                f"Could not locate the best model at {best_model_path}, if you are running a distributed training "
                "on multiple nodes, you should activate `--save_on_each_node`."
            )

    def _load_rng_state(self, checkpoint: str | None) -> None:
        """Restore random number generator states from a checkpoint."""
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            check_torch_load_is_safe()
            checkpoint_rng_state = torch.load(rng_file, weights_only=True)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        if is_torch_xla_available():
            xm.set_rng_state(checkpoint_rng_state["xla"])

        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)
        if is_torch_npu_available():
            set_rng_state_for_device("NPU", torch.npu, checkpoint_rng_state, is_distributed)
        if is_torch_hpu_available():
            set_rng_state_for_device("HPU", torch.hpu, checkpoint_rng_state, is_distributed)
        if is_torch_mlu_available():
            set_rng_state_for_device("MLU", torch.mlu, checkpoint_rng_state, is_distributed)
        if is_torch_musa_available():
            set_rng_state_for_device("MUSA", torch.musa, checkpoint_rng_state, is_distributed)

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, f"rank*-of-{self.args.world_size}-{OPTIMIZER_NAME}"))
            if self.is_fsdp_xla_v1_enabled
            else checkpoint_file_exists
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_xla_available():
                # On TPU we have to take some extra precautions to properly load the states on the right device.
                if self.is_fsdp_xla_v1_enabled:
                    check_torch_load_is_safe()
                    optimizer_state = torch.load(
                        os.path.join(
                            checkpoint, f"rank{self.args.process_index}-of-{self.args.world_size}-{OPTIMIZER_NAME}"
                        ),
                        map_location="cpu",
                        weights_only=True,
                    )
                    # We only need `optimizer` when resuming from checkpoint
                    optimizer_state = optimizer_state["optimizer"]
                else:
                    check_torch_load_is_safe()
                    optimizer_state = torch.load(
                        os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu", weights_only=True
                    )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    lr_scheduler_state = torch.load(
                        os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():

                    def opt_load_hook(mod, opt):
                        opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.is_fsdp_enabled:
                        load_fsdp_optimizer(
                            self.accelerator.state.fsdp_plugin,
                            self.accelerator,
                            self.optimizer,
                            self.model,
                            checkpoint,
                            **get_fsdp_ckpt_kwargs(),
                        )
                    else:
                        check_torch_load_is_safe()
                        self.optimizer.load_state_dict(
                            torch.load(
                                os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location, weights_only=True
                            )
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.lr_scheduler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCHEDULER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)

    def _load_scaler(self, checkpoint: str | None) -> None:
        """If scaler state exists, load it."""
        if checkpoint is None:
            return

        checkpoint_file_exists = os.path.isfile(os.path.join(checkpoint, SCALER_NAME))

        if checkpoint_file_exists:
            # On TPU we have to take some extra precautions to properly load the states on the right device.
            # Load in scaler states
            if is_torch_xla_available():
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    scaler_state = torch.load(
                        os.path.join(checkpoint, SCALER_NAME), map_location="cpu", weights_only=True
                    )
                reissue_pt_warnings(caught_warnings)
                xm.send_cpu_data_to_device(scaler_state, self.args.device)
                self.accelerator.scaler.load_state_dict(scaler_state)
            else:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    check_torch_load_is_safe()
                    self.accelerator.scaler.load_state_dict(
                        torch.load(os.path.join(checkpoint, SCALER_NAME), weights_only=True)
                    )
                reissue_pt_warnings(caught_warnings)

    def _load_callback_state(self) -> None:
        """If callback states exist and were passed in, restore their states if enabled"""
        if not self.args.restore_callback_states_from_checkpoint:
            return
        # Callback states are stored in stateful_callbacks
        not_found = []
        new_callbacks = []
        original_callbacks = self.callback_handler.callbacks + [self.control]
        for stored_callback, data in self.state.stateful_callbacks.items():
            if not isinstance(data, list):
                data = [data]
            if any(callback.__class__.__name__ == stored_callback for callback in original_callbacks):
                # We can load/restore from multiple callbacks of the same type.
                duplicates = [
                    callback for callback in original_callbacks if callback.__class__.__name__ == stored_callback
                ]
                for callback, callback_data in zip(duplicates, data):
                    args = callback_data.get("args", {})
                    attributes = callback_data.get("attributes", {})
                    new_callback = type(callback)(**args)
                    for attribute, value in attributes.items():
                        setattr(new_callback, attribute, value)
                    if isinstance(callback, TrainerControl):
                        # Specifically for restoring the `control` state
                        self.control = new_callback
                    else:
                        new_callbacks.append(new_callback)
                    # We remove the existing callback and add it to the list of new callbacks
                    self.callback_handler.remove_callback(type(new_callback))
                logger.info("Continuing training from checkpoint, restoring any callbacks that were passed in")
            else:
                not_found.append(stored_callback)
        if len(not_found) > 0:
            logger.warning(
                f"Checkpoint included callbacks not included in current configuration. Ignoring. ({', '.join(not_found)})"
            )
        for callback in new_callbacks:
            self.callback_handler.add_callback(callback)

    def _issue_warnings_after_load(self, load_result: Any) -> None:
        """Log warnings for missing or unexpected keys after loading a checkpoint."""
        if len(load_result.missing_keys) != 0:
            if self.model._keys_to_ignore_on_save is not None and set(load_result.missing_keys) == set(
                self.model._keys_to_ignore_on_save
            ):
                self.model.tie_weights()
            else:
                logger.warning(f"There were missing keys in the checkpoint model loaded: {load_result.missing_keys}.")
        if len(load_result.unexpected_keys) != 0:
            logger.warning(
                f"There were unexpected keys in the checkpoint model loaded: {load_result.unexpected_keys}."
            )

    # ---- Saving & Serialization ----

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """
        Will save the model, so you can reload it using `from_pretrained()`.

        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_xla_available():
            save_tpu_checkpoint(
                self.model, self.args, self.accelerator, self.processing_class, self.is_fsdp_xla_v1_enabled, output_dir
            )
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif self.is_fsdp_enabled:
            if "FULL_STATE_DICT" in str(self.accelerator.state.fsdp_plugin.state_dict_type):
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
        elif self.is_deepspeed_enabled:
            try:
                accept_exclude_frozen_parameters = "exclude_frozen_parameters" in set(
                    inspect.signature(self.model_wrapped.save_checkpoint).parameters.keys()
                )
                zero3_sharding = self.deepspeed.config.get("zero_optimization", {}).get("stage", None) == 3
                if accept_exclude_frozen_parameters and _is_peft_model(self.model) and zero3_sharding:
                    # When using PEFT with DeepSpeed ZeRO Stage 3,
                    # we do not need to load the frozen parameters
                    state_dict = self.deepspeed._zero3_consolidated_16bit_state_dict(exclude_frozen_parameters=True)
                else:
                    state_dict = self.accelerator.get_state_dict(self.deepspeed)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                    " zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                # remove the dummy state_dict
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model_wrapped.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)

    def _save(self, output_dir: str | None = None, state_dict: dict | None = None) -> None:
        """Save model weights, configuration, and processing class to `output_dir`."""
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model, keep_torch_compile=False), supported_classes):
                self.accelerator.unwrap_model(self.model, keep_torch_compile=False).save_pretrained(
                    output_dir, state_dict=state_dict
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                safetensors.torch.save_file(
                    state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                )
        else:
            self.model.save_pretrained(output_dir, state_dict=state_dict)

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info("Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`")
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    # ---- Logging & Metrics ----

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen != "no":
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                current_session_num_tokens = self.state.num_input_tokens_seen - self._initial_num_input_tokens_seen
                logs.update(speed_metrics("train", start_time, num_tokens=current_session_num_tokens))

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def store_flos(self) -> None:
        """Store the number of floating-point operations that went into the model."""
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            self.state.total_flos += (
                distributed_broadcast_scalars([self.current_flos], device=self.args.device).sum().item()
            )
            self.current_flos = 0
        else:
            self.state.total_flos += self.current_flos
            self.current_flos = 0

    def floating_point_ops(self, inputs: dict[str, torch.Tensor | Any]) -> int:
        """
        For models that inherit from [`PreTrainedModel`], uses that method to compute the number of floating point
        operations for every backward + forward pass. If using another model, either implement such a method in the
        model or subclass and override this method.

        Args:
            inputs (`dict[str, torch.Tensor | Any]`):
                The inputs and targets of the model.

        Returns:
            `int`: The number of floating-point operations.
        """
        if (main_input := getattr(self.model, "main_input_name", "input_ids")) in inputs and hasattr(
            self.model, "num_parameters"
        ):
            return 6 * inputs[main_input].numel() * self.model.num_parameters(exclude_embeddings=True)
        return 0

    # ---- Hub Integration ----

    def init_hf_repo(self, token: str | None = None) -> None:
        """
        Initializes a git repo in `self.args.hub_model_id`.
        """
        # Only on process zero
        if not self.is_world_process_zero():
            return

        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id

        token = token if token is not None else self.args.hub_token
        repo_url = create_repo(repo_name, token=token, private=self.args.hub_private_repo, exist_ok=True)
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None

    def create_model_card(
        self,
        language: str | None = None,
        license: str | None = None,
        tags: str | list[str] | None = None,
        model_name: str | None = None,
        finetuned_from: str | None = None,
        tasks: str | list[str] | None = None,
        dataset_tags: str | list[str] | None = None,
        dataset: str | list[str] | None = None,
        dataset_args: str | list[str] | None = None,
    ) -> None:
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            language (`str`, *optional*):
                The language of the model (if applicable)
            license (`str`, *optional*):
                The license of the model. Will default to the license of the pretrained model used, if the original
                model given to the `Trainer` comes from a repo on the Hub.
            tags (`str` or `list[str]`, *optional*):
                Some tags to be included in the metadata of the model card.
            model_name (`str`, *optional*):
                The name of the model.
            finetuned_from (`str`, *optional*):
                The name of the model used to fine-tune this one (if applicable). Will default to the name of the repo
                of the original model given to the `Trainer` (if it comes from the Hub).
            tasks (`str` or `list[str]`, *optional*):
                One or several task identifiers, to be included in the metadata of the model card.
            dataset_tags (`str` or `list[str]`, *optional*):
                One or several dataset tags, to be included in the metadata of the model card.
            dataset (`str` or `list[str]`, *optional*):
                One or several dataset identifiers, to be included in the metadata of the model card.
            dataset_args (`str` or `list[str]`, *optional*):
               One or several dataset arguments, to be included in the metadata of the model card.
        """
        if not self.is_world_process_zero():
            return

        model_card_filepath = os.path.join(self.args.output_dir, "README.md")
        is_peft_library = False
        if os.path.exists(model_card_filepath):
            library_name = ModelCard.load(model_card_filepath).data.get("library_name")
            is_peft_library = library_name == "peft"

            # Append existing tags in `tags`
            existing_tags = ModelCard.load(model_card_filepath).data.tags
            if tags is not None and existing_tags is not None:
                if isinstance(tags, str):
                    tags = [tags]
                for tag in existing_tags:
                    if tag not in tags:
                        tags.append(tag)

        training_summary = TrainingSummary.from_trainer(
            self,
            language=language,
            license=license,
            tags=tags,
            model_name=model_name,
            finetuned_from=finetuned_from,
            tasks=tasks,
            dataset_tags=dataset_tags,
            dataset=dataset,
            dataset_args=dataset_args,
        )
        model_card = training_summary.to_model_card()
        with open(model_card_filepath, "w") as f:
            f.write(model_card)

        if is_peft_library:
            self.accelerator.unwrap_model(self.model).create_or_update_model_card(self.args.output_dir)

    def push_to_hub(
        self,
        commit_message: str | None = "End of training",
        blocking: bool = True,
        token: str | None = None,
        revision: str | None = None,
        **kwargs,
    ) -> CommitInfo:
        """
        Upload `self.model` and `self.processing_class` to the 🤗 model hub on the repo `self.args.hub_model_id`.

        Parameters:
            commit_message (`str`, *optional*, defaults to `"End of training"`):
                Message to commit while pushing.
            blocking (`bool`, *optional*, defaults to `True`):
                Whether the function should return only when the `git push` has finished.
            token (`str`, *optional*, defaults to `None`):
                Token with write permission to overwrite Trainer's original args.
            revision (`str`, *optional*):
                The git revision to commit from. Defaults to the head of the "main" branch.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to [`~Trainer.create_model_card`].

        Returns:
            The URL of the repository where the model was pushed if `blocking=False`, or a `Future` object tracking the
            progress of the commit if `blocking=True`.
        """
        self.callback_handler.on_push_begin(self.args, self.state, self.control)

        model_name = kwargs.pop("model_name", None)
        if model_name is None and self.args.should_save:
            if self.args.hub_model_id is None:
                model_name = Path(self.args.output_dir).name
            else:
                model_name = self.args.hub_model_id.split("/")[-1]
        token = token if token is not None else self.args.hub_token

        # In case the user calls this method with args.push_to_hub = False
        if self.hub_model_id is None:
            self.init_hf_repo(token=token)

        # Needs to be executed on all processes for TPU training, but will only save on the processed determined by
        # self.args.should_save.
        self.save_model(_internal_call=True)

        # Only push from one node.
        if not self.is_world_process_zero():
            return

        # Add additional tags in the case the model has already some tags and users pass
        # "tags" argument to `push_to_hub` so that trainer automatically handles internal tags
        # from all models since Trainer does not call `model.push_to_hub`.
        if getattr(self.model, "model_tags", None) is not None:
            if "tags" not in kwargs:
                kwargs["tags"] = []

            # If it is a string, convert it to a list
            if isinstance(kwargs["tags"], str):
                kwargs["tags"] = [kwargs["tags"]]

            for model_tag in self.model.model_tags:
                if model_tag not in kwargs["tags"]:
                    kwargs["tags"].append(model_tag)

        self.create_model_card(model_name=model_name, **kwargs)

        if revision is None:
            revision = self.args.hub_revision

        # Wait for the current upload to be finished.
        self._finish_current_push()

        return upload_folder(
            repo_id=self.hub_model_id,
            folder_path=self.args.output_dir,
            commit_message=commit_message,
            token=token,
            run_as_future=not blocking,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
            revision=revision,
        )

    def _push_from_checkpoint(self, checkpoint_folder: str) -> None:
        """Push model and checkpoint files to the Hub from a checkpoint folder."""
        if not self.is_world_process_zero() or self.args.hub_strategy == HubStrategy.END:
            return
        # If we haven't finished the last push, we don't do this one unless args.hub_always_push=True.
        if not self.args.hub_always_push and self.push_in_progress is not None and not self.push_in_progress.is_done():
            return

        self.callback_handler.on_push_begin(self.args, self.state, self.control)
        output_dir = self.args.output_dir
        # To avoid a new synchronization of all model weights, we just copy the file from the checkpoint folder
        modeling_files = [CONFIG_NAME, GENERATION_CONFIG_NAME, WEIGHTS_NAME, SAFE_WEIGHTS_NAME]
        #  Add sharded checkpoints if we have an index
        for index_file in [WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME]:
            index_path = os.path.join(checkpoint_folder, index_file)
            if os.path.isfile(index_path):
                modeling_files.append(index_file)
                with open(index_path) as f:
                    index = json.loads(f.read())
                shard_files = list(set(index["weight_map"].values()))
                modeling_files.extend(shard_files)
        if is_peft_available():
            modeling_files.extend([ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, ADAPTER_SAFE_WEIGHTS_NAME])
        for modeling_file in modeling_files:
            if os.path.isfile(os.path.join(checkpoint_folder, modeling_file)):
                shutil.copy(os.path.join(checkpoint_folder, modeling_file), os.path.join(output_dir, modeling_file))
        # Saving the processing class is fast and we don't know how many files it may have spawned, so we resave it to be sure.
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        # Same for the training arguments
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.args.save_strategy == SaveStrategy.STEPS:
            commit_message = f"Training in progress, step {self.state.global_step}"
        else:
            commit_message = f"Training in progress, epoch {int(self.state.epoch)}"

        model_push_job = upload_folder(
            repo_id=self.hub_model_id,
            folder_path=output_dir,
            commit_message=commit_message,
            token=self.args.hub_token,
            run_as_future=True,
            ignore_patterns=["_*", f"{PREFIX_CHECKPOINT_DIR}-*"],
            revision=self.args.hub_revision,
        )

        push_jobs = [model_push_job]

        if self.args.hub_strategy in [HubStrategy.CHECKPOINT, HubStrategy.ALL_CHECKPOINTS]:
            path_in_repo = (
                "last-checkpoint" if self.args.hub_strategy == HubStrategy.CHECKPOINT else Path(checkpoint_folder).name
            )
            checkpoint_push = upload_folder(
                repo_id=self.hub_model_id,
                folder_path=checkpoint_folder,
                path_in_repo=path_in_repo,
                commit_message=commit_message + ", checkpoint",
                token=self.args.hub_token,
                run_as_future=True,
                revision=self.args.hub_revision,
            )
            push_jobs.append(checkpoint_push)

        if self.push_in_progress is None or self.push_in_progress.is_done():
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)

    def _finish_current_push(self) -> None:
        """Wait for any in-progress push to the Hub to complete."""
        if not hasattr(self, "push_in_progress"):
            return
        if self.push_in_progress is not None and not self.push_in_progress.is_done():
            logger.info("Waiting for the current checkpoint push to be finished, this might take a couple of minutes.")
            self.push_in_progress.wait_until_done()

    # ---- Hyperparameter Search ----

    def hyperparameter_search(
        self,
        hp_space: Callable[["optuna.Trial"], dict[str, float]] | None = None,
        compute_objective: Callable[[dict[str, float]], float] | None = None,
        n_trials: int = 20,
        direction: str | list[str] = "minimize",
        backend: str | HPSearchBackend | None = None,
        hp_name: Callable[["optuna.Trial"], str] | None = None,
        **kwargs,
    ) -> BestRun | list[BestRun]:
        """
        Launch a hyperparameter search using `optuna` or `Ray Tune`. The optimized quantity is determined
        by `compute_objective`, which defaults to a function returning the evaluation loss when no metric is provided,
        the sum of all metrics otherwise.

        <Tip warning={true}>

        To use this method, you need to have provided a `model_init` when initializing your [`Trainer`]: we need to
        reinitialize the model at each new run. This is incompatible with the `optimizers` argument, so you need to
        subclass [`Trainer`] and override the method [`~Trainer.create_optimizer_and_scheduler`] for custom
        optimizer/scheduler.

        </Tip>

        Args:
            hp_space (`Callable[["optuna.Trial"], dict[str, float]]`, *optional*):
                A function that defines the hyperparameter search space. Will default to
                [`~trainer_utils.default_hp_space_optuna`] or [`~trainer_utils.default_hp_space_ray`]
                depending on your backend.
            compute_objective (`Callable[[dict[str, float]], float]`, *optional*):
                A function computing the objective to minimize or maximize from the metrics returned by the `evaluate`
                method. Will default to [`~trainer_utils.default_compute_objective`].
            n_trials (`int`, *optional*, defaults to 100):
                The number of trial runs to test.
            direction (`str` or `list[str]`, *optional*, defaults to `"minimize"`):
                If it's single objective optimization, direction is `str`, can be `"minimize"` or `"maximize"`, you
                should pick `"minimize"` when optimizing the validation loss, `"maximize"` when optimizing one or
                several metrics. If it's multi objectives optimization, direction is `list[str]`, can be List of
                `"minimize"` and `"maximize"`, you should pick `"minimize"` when optimizing the validation loss,
                `"maximize"` when optimizing one or several metrics.
            backend (`str` or [`~training_utils.HPSearchBackend`], *optional*):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending
                on which one is installed. If all are installed, will default to optuna.
            hp_name (`Callable[["optuna.Trial"], str]]`, *optional*):
                A function that defines the trial/run name. Will default to None.
            kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments for each backend:

                - `optuna`: parameters from
                  [optuna.study.create_study](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html)
                  and also the parameters `timeout`, `n_jobs` and `gc_after_trial` from
                  [optuna.study.Study.optimize](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize)
                - `ray`: parameters from [tune.run](https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run).
                  If `resources_per_trial` is not set in the `kwargs`, it defaults to 1 CPU core and 1 GPU (if available).
                  If `progress_reporter` is not set in the `kwargs`,
                  [ray.tune.CLIReporter](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html) is used.
        Returns:
            [`trainer_utils.BestRun` or `list[trainer_utils.BestRun]`]: All the information about the best run or best
            runs for multi-objective optimization. Experiment summary can be found in `run_summary` attribute for Ray
            backend.
        """
        if backend is None:
            backend = default_hp_search_backend()
        backend = HPSearchBackend(backend)
        backend_obj = ALL_HYPERPARAMETER_SEARCH_BACKENDS[backend]()
        backend_obj.ensure_available()
        self.hp_search_backend = backend
        if self.model_init is None:
            raise RuntimeError(
                "To use hyperparameter search, you need to pass your model through a model_init function."
            )

        self.hp_space = backend_obj.default_hp_space if hp_space is None else hp_space
        self.hp_name = hp_name
        self.compute_objective = default_compute_objective if compute_objective is None else compute_objective

        best_run = backend_obj.run(self, n_trials, direction, **kwargs)

        self.hp_search_backend = None
        return best_run

    def call_model_init(self, trial: "optuna.Trial | dict[str, Any] | None" = None) -> nn.Module:
        """Invoke `model_init` to get a fresh model instance, optionally conditioned on a hyperparameter trial."""
        model_init_argcount = number_of_arguments(self.model_init)
        if model_init_argcount == 0:
            model = self.model_init()
        elif model_init_argcount == 1:
            model = self.model_init(trial)
        else:
            raise RuntimeError("model_init should have 0 or 1 argument.")

        if model is None:
            raise RuntimeError("model_init should not return None.")

        return model

    def _hp_search_setup(self, trial: "optuna.Trial | dict[str, Any] | None") -> None:
        """Set up training arguments and accelerator state for a hyperparameter search trial."""
        self._trial = trial

        if self.hp_search_backend is None or trial is None:
            return
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            params = self.hp_space(trial)
        elif self.hp_search_backend == HPSearchBackend.RAY:
            params = trial
            params.pop("wandb", None)
        elif self.hp_search_backend == HPSearchBackend.WANDB:
            params = trial

        for key, value in params.items():
            if not hasattr(self.args, key):
                logger.warning(
                    f"Trying to set {key} in the hyperparameter search but there is no corresponding field in"
                    " `TrainingArguments`."
                )
                continue
            old_attr = getattr(self.args, key, None)
            # Casting value to the proper type
            if old_attr is not None:
                value = type(old_attr)(value)

            setattr(self.args, key, value)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            logger.info(f"Trial: {trial.params}")
        if self.hp_search_backend == HPSearchBackend.WANDB:
            logger.info(f"W&B Sweep parameters: {trial}")
        if self.is_deepspeed_enabled:
            if self.args.deepspeed is None:
                raise ValueError("For sweeps with deepspeed, `args.deepspeed` must be set")

            self.accelerator.free_memory()

            # Rebuild the deepspeed config to reflect the updated training parameters
            from accelerate.utils import DeepSpeedPlugin

            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            self.args.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.args.deepspeed)
            self.args.hf_deepspeed_config.trainer_config_process(self.args)
            self.args.deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.args.hf_deepspeed_config)

            # From 1.0 on, we need to fully wipe the DS plugin when doing sweeps.
            # Simply calling `_reset_state` is enough and doesn't need a version pin.
            AcceleratorState()._reset_state()

        # `train_batch_size` might change when using HPO https://github.com/huggingface/transformers/pull/18918
        self._train_batch_size = self.args.train_batch_size
        self.create_accelerator_and_postprocess()

    def _report_to_hp_search(
        self, trial: "optuna.Trial | dict[str, Any] | None", step: int, metrics: dict[str, float]
    ) -> None:
        """Report intermediate metrics to the active hyperparameter search backend."""
        if self.hp_search_backend is None or trial is None:
            return
        metrics = metrics.copy()
        self.objective = self.compute_objective(metrics)
        if self.hp_search_backend == HPSearchBackend.OPTUNA:
            import optuna

            if hasattr(trial, "study") and not trial.study._is_multi_objective():
                trial.report(self.objective, step)
                if trial.should_prune():
                    self.callback_handler.on_train_end(self.args, self.state, self.control)
                    raise optuna.TrialPruned()
        elif self.hp_search_backend == HPSearchBackend.RAY:
            import ray.tune

            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                checkpoint = None
                if self.control.should_save:
                    self._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                    checkpoint = ray.tune.Checkpoint.from_directory(temp_checkpoint_dir)
                metrics["objective"] = self.objective
                ray.tune.report(metrics, checkpoint=checkpoint)

    def _tune_save_checkpoint(self, checkpoint_dir: str) -> None:
        """Save a checkpoint during a Ray Tune hyperparameter search trial."""
        output_dir = os.path.join(checkpoint_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        self.save_model(output_dir, _internal_call=True)
        if self.args.should_save:
            # Update the `TrainerControl` state to where we are currently
            self.state.stateful_callbacks["TrainerControl"] = self.control.state()
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

    # ---- Callbacks ----

    def add_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        """
        Add a callback to the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will instantiate a member of that class.
        """
        self.callback_handler.add_callback(callback)

    def pop_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> TrainerCallback | None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`] and returns it.

        If the callback is not found, returns `None` (and no error is raised).

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will pop the first member of that class found in the list of callbacks.

        Returns:
            [`~transformers.TrainerCallback`]: The callback removed, if found.
        """
        return self.callback_handler.pop_callback(callback)

    def remove_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        """
        Remove a callback from the current list of [`~transformers.TrainerCallback`].

        Args:
           callback (`type` or [`~transformers.TrainerCallback]`):
               A [`~transformers.TrainerCallback`] class or an instance of a [`~transformers.TrainerCallback`]. In the
               first case, will remove the first member of that class found in the list of callbacks.
        """
        self.callback_handler.remove_callback(callback)

    # ---- Utilities ----

    def is_local_process_zero(self) -> bool:
        """
        Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on several
        machines) main process.
        """
        return self.args.local_process_index == 0

    def is_world_process_zero(self) -> bool:
        """
        Whether or not this process is the global main process (when training in a distributed fashion on several
        machines, this is only going to be `True` for one process).
        """
        # Special case for SageMaker ModelParallel since there process_index is dp_process_index, not the global
        # process index.
        if is_sagemaker_mp_enabled():
            return smp.rank() == 0
        return self.args.process_index == 0

    def _move_model_to_device(self, model: nn.Module, device: torch.device) -> None:
        """Move the model to the specified device, re-tying weights on XLA if needed."""
        if getattr(model, "hf_device_map", None) is not None:
            logger.warning(
                "The model is already on multiple devices. Skipping the move to device specified in `args`."
            )
            return
        model = model.to(device)
        # Moving a model to an XLA device disconnects the tied weights, so we have to retie them.
        if self.args.parallel_mode == ParallelMode.TPU and hasattr(model, "tie_weights"):
            model.tie_weights()


class GRMTrainer(BaseTrainer):

    def __init__(
        self,
        model: "str | PreTrainedModel | PeftModel",
        reward_funcs: RewardFunc | list[RewardFunc] | None = None,
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        reward_processing_classes: PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        peft_config: "PeftConfig | None" = None,
        rollout_func: RolloutFunc | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        only_generate: bool | None = False,
    ):
        self.only_generate = only_generate
        self.analysis_records = []
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else get_config_model_id(model.config)
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model
        if isinstance(model, str):
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            model = create_model_from_path(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                logger.warning(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "The `model_init_kwargs` will be ignored."
                )

        # Some models (SmolVLM/Idefics3) don't support `logits_to_keep` argument and error out if we pass it
        # Inspect the forward method before we wrap the model with PEFT
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys()
            if not hasattr(model, "get_base_model")
            else inspect.signature(model.get_base_model().forward).parameters.keys()
        )

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(
                get_config_model_id(model.config), truncation_side="left", padding_side="left"
            )

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            self._tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            self._tokenizer = processing_class
        else:
            raise TypeError("The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`")

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        if is_peft_available() and is_peft_model(model) and peft_config is not None:
            raise ValueError(
                "You passed a `PeftModel` instance together with a `peft_config` to the trainer. Please first merge "
                "and unload the existing adapter, save the resulting base model, and then pass that base model along "
                "with the new `peft_config` to the trainer."
            )
        if is_peft_available() and is_peft_model(model) and args.beta != 0.0:
            # If the model is a PEFT model with a pretrained adapter, we need to create a "ref" adapter that is a copy
            # of the "default" adapter, so that we can use it as the reference model during GRPO training.
            model.add_adapter("ref", model.peft_config["default"])
            for name, param in model.named_parameters():
                if ".default." in name:
                    ref_name = name.replace(".default.", ".ref.")
                    ref_param = model.get_parameter(ref_name)
                    ref_param.data.copy_(param.data)

        # Create PEFT model
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # When using gradient checkpointing with PEFT, we need to enable input gradients. transformers.Trainer normally
        # handles this, but a bug currently prevents it; see https://github.com/huggingface/transformers/issues/42489
        if is_peft_available() and is_peft_model(model) and args.gradient_checkpointing:
            model.enable_input_require_grads()

        # When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
        # original paper (see https://huggingface.co/papers/2305.14314, paragraph 3). Normally, this can be done by
        # passing `autocast_adapter_dtype=False` to `get_peft_model`, but this option is not yet supported for
        # quantized models. See: https://github.com/huggingface/peft/issues/2889
        # Non-quantized models do not have the `is_loaded_in_{8,4}bit` attributes, whereas quantized models do
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            for param in model.parameters():
                if param.requires_grad:
                    param.data = param.data.to(torch.bfloat16)

        # Reward functions
        if reward_funcs is None:
            reward_funcs = []
        elif not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_func_names = []
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                model_init_kwargs = args.model_init_kwargs or {}
                # Distributed training requires device_map=None ("auto" fails)
                if args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                    model_init_kwargs["device_map"] = None
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
            if isinstance(reward_funcs[i], nn.Module):  # Use Module over PretrainedModel for compat w/ compiled models
                self.reward_func_names.append(get_config_model_id(reward_funcs[i].config).split("/")[-1])
            else:
                self.reward_func_names.append(reward_funcs[i].__name__)
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError(
                f"The number of reward processing classes ({len(reward_processing_classes)}) must match the number of "
                f"reward functions ({len(reward_funcs)})."
            )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs, strict=True)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(get_config_model_id(reward_func.config))
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Rollout function
        if rollout_func is not None and os.environ.get("TRL_EXPERIMENTAL_SILENCE", "0") != "1":
            warnings.warn(
                "You are using 'rollout_func', which is an experimental feature. This API may change or be removed at "
                "any time without prior notice. Silence this warning by setting environment variable "
                "TRL_EXPERIMENTAL_SILENCE=1.",
                UserWarning,
                stacklevel=2,
            )
        self.rollout_func = rollout_func

        # Check for async functions to start an event loop on a daemon thread
        self._has_async_funcs = any(inspect.iscoroutinefunction(func) for func in self.reward_funcs)

        if self._has_async_funcs:
            self.async_loop_thread, self.async_loop, self.async_loop_ready_event = start_event_loop_in_daemon(
                name="GRPOTrainer-AsyncLoop"
            )
            # wait until the event loop is running in the daemon thread
            self.async_loop_ready_event.wait()
            atexit.register(shutdown_event_loop_in_daemon, self.async_loop_thread, self.async_loop)

        self.chat_template = None

        # Training arguments
        # Parameters that control the data preprocessing
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.num_generations_eval = args.num_generations_eval or self.num_generations
        # Parameters that control generation
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.chat_template_kwargs = args.chat_template_kwargs or {}
        self.repetition_penalty = args.repetition_penalty
        self.pad_to_multiple_of = args.pad_to_multiple_of
        # Parameters that control generation acceleration powered by vLLM
        self.use_vllm = args.use_vllm
        self.vllm_mode = args.vllm_mode
        # Parameters that control colocated vLLM execution (only used when `vllm_mode` is `"colocate"`)
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization  # only applies to colocation mode
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size  # only applies to colocation mode
        # Parameters that control the training
        self.vllm_importance_sampling_correction = args.vllm_importance_sampling_correction
        self.vllm_importance_sampling_mode = args.vllm_importance_sampling_mode
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.loss_type = args.loss_type
        self.multi_objective_aggregation = args.multi_objective_aggregation
        self.scale_rewards = args.scale_rewards

        self.use_liger_kernel = args.use_liger_kernel
        self.importance_sampling_level = args.importance_sampling_level
        if self.use_liger_kernel and self.importance_sampling_level not in ("token", "sequence"):
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. "
                "Possible values are 'token' and 'sequence'."
            )
        self.off_policy_mask_threshold = args.off_policy_mask_threshold
        if self.use_liger_kernel and self.off_policy_mask_threshold is not None:
            raise ValueError("Liger kernel does not support off-policy sequence masking yet.")
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_kernel and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        self.mask_truncated_completions = args.mask_truncated_completions

        # Datasets
        self.shuffle_dataset = args.shuffle_dataset

        if train_dataset is None:
            raise ValueError("`train_dataset` is required")
        elif (
            isinstance(train_dataset, IterableDataset)
            or isinstance(eval_dataset, IterableDataset)
            or (
                isinstance(eval_dataset, dict) and any(isinstance(ds, IterableDataset) for ds in eval_dataset.values())
            )
        ):
            # See https://github.com/huggingface/trl/issues/3213
            raise NotImplementedError(
                "Iterable datasets are not yet supported in GRPOTrainer. Please use a standard dataset instead."
            )

        if args.loss_type == "luspo" and args.importance_sampling_level != "sequence":
            logger.warning(
                "When using `'luspo'` loss, `importance_sampling_level` should be set to `'sequence'` to mirror the "
                "paper's setup."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version("5.0.0"):
            args.gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
            compute_metrics=compute_metrics
        )

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            model_init_kwargs = args.model_init_kwargs or {}
            # Distributed training requires device_map=None ("auto" fails)
            if self.args.distributed_state.distributed_type in ["MULTI_GPU", "DEEPSPEED"]:
                model_init_kwargs["device_map"] = None
            self.ref_model = create_model_from_path(get_config_model_id(self.model.config), **model_init_kwargs)

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Cast LM Head To FP32
        if args.cast_lm_head_to_fp32:

            def _cast_lm_head_to_fp32(target_model: PreTrainedModel):
                """Cast lm_head to fp32 while preserving embedding output dtype if tied."""

                def cast_inputs_to_fp32(module, inputs):
                    # Preserve other positional args and kwargs untouched
                    if not inputs:
                        return inputs
                    return (inputs[0].to(torch.float32),) + inputs[1:]

                original_dtype_local = target_model.lm_head.weight.dtype
                target_model.lm_head = target_model.lm_head.float()
                target_model.lm_head.register_forward_pre_hook(cast_inputs_to_fp32)

                if target_model.config.tie_word_embeddings:

                    def cast_outputs_to_original_dtype(module, args, output):
                        return output.to(original_dtype_local)

                    # Only cast activations; weights are now fp32 (intentional for numerical stability of logits)
                    target_model.model.embed_tokens.register_forward_hook(cast_outputs_to_original_dtype)

            _cast_lm_head_to_fp32(model)
            if self.ref_model is not None:
                _cast_lm_head_to_fp32(self.ref_model)

        # Liger loss
        if self.use_liger_kernel:
            if not is_liger_kernel_available():
                raise ImportError(
                    "Liger is required to use `use_liger_kernel` as the GRPO loss. Run `pip install liger-kernel`."
                )
            # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
            self._forward_redirection = _ForwardRedirection()

            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.beta,
                epsilon_low=self.epsilon_low,
                epsilon_high=self.epsilon_high,
                temperature=self.temperature,
                use_ref_model=self.beta != 0.0,
                loss_type=self.loss_type,
                max_completion_length=self.max_completion_length,
                importance_sampling_level=self.importance_sampling_level,
            )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self._current_train_step_time = 0.0
        self.log_completions = args.log_completions
        self.log_unique_prompts = args.log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print
        # Keep logs sized to the generation batch to record only outputs from the latest model update.
        self._logs = {
            "prompt": deque(maxlen=args.generation_batch_size),
            "completion": deque(maxlen=args.generation_batch_size),
            "rewards": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
            "advantages": deque(maxlen=args.generation_batch_size),
            "extra": defaultdict(lambda: deque(maxlen=args.generation_batch_size)),
        }
        # Buffers for user-logged data from reward functions, flushed after gathering
        self._pending_extra_logs = defaultdict(list)
        self._pending_metrics = defaultdict(list)

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # Initialize vLLM generation backend
            self.vllm_generation = VLLMGenerationGRM(
                model=self.model,
                accelerator=self.accelerator,
                is_fsdp_enabled=self.is_fsdp_enabled,
                processing_class=self.processing_class,
                # vLLM configuration
                mode=args.vllm_mode,
                structured_outputs_regex=args.vllm_structured_outputs_regex,
                # Server mode configuration
                server_base_url=args.vllm_server_base_url,
                server_host=args.vllm_server_host,
                server_port=args.vllm_server_port,
                group_port=args.vllm_group_port,
                server_timeout=args.vllm_server_timeout,
                # Colocate mode configuration
                tensor_parallel_size=args.vllm_tensor_parallel_size,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                max_model_length=args.vllm_max_model_length,
                max_num_seqs=args.per_device_train_batch_size
                * args.vllm_tensor_parallel_size
                * args.steps_per_generation,
                enable_sleep_mode=args.vllm_enable_sleep_mode,
                model_impl=args.vllm_model_impl,
                # Generation configuration
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                max_completion_length=self.max_completion_length,
                logprobs=0,  # we only need the generated token logprobs for the importance sampling correction
                generation_kwargs=args.generation_kwargs,
            )

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation
        else:
            generation_kwargs = {
                "max_new_tokens": self.max_completion_length,
                "do_sample": True,
                "pad_token_id": self._tokenizer.pad_token_id,
                "bos_token_id": self._tokenizer.bos_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
                "repetition_penalty": self.repetition_penalty,
                "cache_implementation": args.cache_implementation,
            }

            if args.bad_word_list is not None:
                tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(self.model.config.name_or_path, add_prefix_space=True)
                bad_words_ids = []
                for word in args.bad_word_list:
                    tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
                    bad_words_ids.append(tokenized_word)
                generation_kwargs['bad_words_ids'] = bad_words_ids

            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            self.generation_config = GenerationConfig(**generation_kwargs, disable_compile=True)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            if self.beta == 0.0:
                raise ValueError(
                    "You passed `sync_ref_model=True` while `beta=0.0`, which means the reference model is not used "
                    "during training. Consequently, GRPOTrainer does not create a `ref_model` instance, and there is "
                    "nothing to synchronize. Please set `sync_ref_model=False`, or set `beta` to a non-zero value."
                )
            if is_peft_model(model):
                class SyncLoRARefModelCallback(TrainerCallback):
                    """
                    Callback to synchronize the model with a reference model.
                    """

                    def __init__(
                        self,
                        accelerator: Accelerator | None,
                        policy_adapter_name: str = "default",
                        ref_adapter_name: str = "ref"
                    ):
                        self.accelerator = accelerator
                        self.policy_adapter_name = policy_adapter_name
                        self.ref_adapter_name = ref_adapter_name

                    @staticmethod
                    def sync_lora_adapter(model, policy_adapter_name, ref_adapter_name, alpha):
                        policy_tag = f".{policy_adapter_name}."
                        ref_tag = f".{ref_adapter_name}."
                        for name, param in model.named_parameters():
                            if policy_tag in name:
                                ref_name = name.replace(policy_tag, ref_tag)
                                ref_param = model.get_parameter(ref_name)
                                ref_param.data.mul_(1.0 - alpha).add_(param.data, alpha=alpha)

                    def on_step_end(self, args, state, control, **kwargs):
                        model: PreTrainedModel = kwargs["model"]

                        if state.global_step % args.ref_model_sync_steps == 0:
                            if self.accelerator:
                                model = self.accelerator.unwrap_model(model)
                            self.sync_lora_adapter(model, self.policy_adapter_name, self.ref_adapter_name, args.ref_model_mixup_alpha)

                self.add_callback(SyncLoRARefModelCallback(policy_adapter_name="default", ref_adapter_name="ref", accelerator=self.accelerator))
            else:
                self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if self.is_deepspeed_enabled:
                    self.reward_funcs[i] = prepare_deepspeed(reward_func, self.accelerator)
                else:
                    # set device placement to True to make `prepare_model` move `reward_func` to device when using fsdp
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True, device_placement=True
                    )

        if self.accelerator.is_main_process and self.log_completions:
            os.makedirs(os.path.join(self.args.output_dir, "completions"), exist_ok=True)
            if self.args.log_completions_hub_repo is not None:
                repo_id = self.args.log_completions_hub_repo
                create_repo(repo_id, private=self.args.hub_private_repo, repo_type="dataset", exist_ok=True)
                template_path = pkg_resources.files("trl").joinpath("templates/completions_dataset_card.md")
                card_data = DatasetCardData(
                    pretty_name="TRL Completion logs",
                    tags=["trl", "trl-logs", "completions"],
                )
                card = DatasetCard.from_template(
                    card_data=card_data,
                    template_path=str(template_path),
                    repo_id=repo_id,
                    hub_model_id=self.args.hub_model_id,
                )
                card.push_to_hub(repo_id)
                self.commit_scheduler = CommitScheduler(
                    repo_id=repo_id,
                    repo_type="dataset",
                    folder_path=f"{self.args.output_dir}/completions",
                    every=2,  # minutes
                    allow_patterns=["*.parquet"],
                )

    # TRL
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't
        # work. Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    # TRL
    def _tokenize_prompts(self, prompts: list):
        """Tokenize prompts and extract images/multimodal fields for generation."""
        if is_conversational({"prompt": prompts[0]}):

            # Workaround for a bug in transformers 5.3.0 where some processors (e.g. Qwen2.5-VL) crash on
            # batched unpadded input (transformers#44514).
            # Fixed in transformers 5.4.0 (transformers#44563).
            needs_padding_workaround = Version("5.3.0") <= Version(transformers.__version__) < Version("5.4.0")
            tokenized = self.processing_class.apply_chat_template(
                conversation=prompts,
                tools=None,
                chat_template=self.chat_template,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **({"padding": True} if needs_padding_workaround else {}),
                **self.chat_template_kwargs,
            )
            if needs_padding_workaround:
                # Unpad input_ids: remove padding tokens using attention_mask to get per-sequence lists
                prompt_ids = [
                    [tok for tok, m in zip(ids, mask, strict=True) if m]
                    for ids, mask in zip(tokenized["input_ids"], tokenized["attention_mask"], strict=True)
                ]
            else:
                prompt_ids = tokenized["input_ids"]
        else:
            prompt_ids = self.processing_class(text=prompts, add_special_tokens=False)["input_ids"]
        return prompt_ids
    
    def _generate_single_turn(self, prompt_ids):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            step_changed = self.state.global_step != self._last_loaded_step
            weights_not_synced = not getattr(self.vllm_generation, "_weights_synced", False)
            if step_changed or weights_not_synced:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Generate using vLLM with raw token IDs
            num_generations = self.num_generations if mode == "train" else self.num_generations_eval
            _, completion_ids, logprobs, _ = self.vllm_generation.generate(
                prompts=prompt_ids,
                images=None,
                num_generations=num_generations,
                profiler=profiling_context(self, "vLLM.generate"),
            )
            # vLLM returns per-token top-k logprobs; keep only the top-1 (sampled token) logprob
            logprobs = [
                [
                    float("nan") if lp is None or len(lp) == 0 or lp[0] is None else lp[0]
                    for lp in seq
                ]
                for seq in logprobs
            ]

        else:
            # Regular generation path: left-pad token IDs into tensors
            prompt_tensors = [torch.tensor(ids) for ids in prompt_ids]
            padded_ids = pad(prompt_tensors, padding_value=self._tokenizer.pad_token_id, padding_side="left")
            attention_mask = pad([torch.ones_like(t) for t in prompt_tensors], padding_value=0, padding_side="left")
            generate_inputs = {"input_ids": padded_ids, "attention_mask": attention_mask}
            generate_inputs = super()._prepare_inputs(generate_inputs)

            with (
                profiling_context(self, "transformers.generate"),
                unwrap_model_for_generation(
                    self.model_wrapped,
                    self.accelerator,
                    gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                    generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
                ) as unwrapped_model,
                torch.no_grad(),
                FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
            ):
                prompt_completion_ids = unwrapped_model.generate(
                    **generate_inputs, generation_config=self.generation_config
                )

            # Compute prompt length and extract completion ids
            prompt_length = generate_inputs["input_ids"].size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            
            # Mask everything after the first EOS token
            is_eos = completion_ids == self._tokenizer.eos_token_id
            eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
            eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
            sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
            completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
            completion_ids = [
                c[m].tolist() for c, m in zip(completion_ids.cpu(), completion_mask.bool().cpu(), strict=True)
            ]
            logprobs = None  # not used in this case

        return completion_ids, logprobs

    # TRL
    def _generate(self, prompts: list):
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Copy the prompts to avoid modifying the original list
        prompts = copy.deepcopy(prompts)

        if self.rollout_func is not None:
            # Keep vLLM weights in sync for custom rollouts that rely on vLLM utilities.
            if self.use_vllm and self.state.global_step != self._last_loaded_step:
                with profiling_context(self, "sync_weights"):
                    self.vllm_generation.sync_weights()
                self._last_loaded_step = self.state.global_step

            # Pass prompts to rollout_func preserving structured messages.
            # Chat templating must happen inside rollout_func, at the backend boundary, so that
            # multimodal content (images, typed content blocks) is not lost before rollout logic runs.
            output = self.rollout_func(prompts, self)
            required_keys = {"prompt_ids", "completion_ids", "logprobs"}
            missing_keys = required_keys - output.keys()
            if missing_keys:
                missing_keys_list = sorted(missing_keys)
                raise ValueError(f"rollout_func must return keys {missing_keys_list} in its output dict.")
            extra_fields = {k: v for k, v in output.items() if k not in required_keys}
            prompt_ids, completion_ids, logprobs = output["prompt_ids"], output["completion_ids"], output["logprobs"]
        else:
            prompt_ids = self._tokenize_prompts(prompts)
            completion_ids, logprobs = self._generate_single_turn(prompt_ids)
            extra_fields = {}

        # Decode completions. It's important to use `parse_response` when possible, because it handles tool calls.
        if is_conversational({"prompt": prompts[0]}):
            if (
                Version(transformers.__version__) >= Version("5.0.0")  # parse_response added in v5
                and hasattr(self._tokenizer, "response_schema")  # attribute not set by default for now
                and self._tokenizer.response_schema is not None  # only works if the tokenizer has a schema
            ):
                completions = [[parse_response(self._tokenizer, ids)] for ids in completion_ids]
            else:
                contents = self.processing_class.batch_decode(
                    completion_ids, 
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False
                )
                completions = [[{"role": "assistant", "content": content}] for content in contents]
        else:
            completions = self.processing_class.batch_decode(
                completion_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False
            )

        # Get completion length per sequence, used for logging
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + agg_completion_lengths.sum()).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        # Identify sequences that terminated with EOS and log their lengths
        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
        is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids], device=device)
        agg_is_truncated = self.accelerator.gather(is_truncated)
        self._metrics[mode]["completions/clipped_ratio"].append(agg_is_truncated.float().mean().item())
        term_completion_lengths = agg_completion_lengths[~agg_is_truncated]
        if len(term_completion_lengths) == 0:  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_lengths.float().max().item())

        return (
            prompt_ids,
            completion_ids,
            completions,
            logprobs,
            extra_fields,
        )
   
    def gather_variable_length_1d(self, x: torch.Tensor, pad_value: float = 1e30):
        if x.dim() != 1:
            x = x.reshape(-1)

        local_len = torch.tensor([x.numel()], device=x.device, dtype=torch.long)
        gathered_lens = self.accelerator.gather(local_len).reshape(-1)
        max_len = int(gathered_lens.max().item()) if gathered_lens.numel() > 0 else 0

        if x.numel() < max_len:
            pad = torch.full(
                (max_len - x.numel(),),
                fill_value=pad_value,
                device=x.device,
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=0)

        gathered = self.accelerator.gather(x)
        if max_len == 0:
            return gathered[:0]
        
        gathered = gathered.reshape(gathered_lens.numel(), max_len)
        arange = torch.arange(max_len, device=x.device).unsqueeze(0)
        valid_mask = arange < gathered_lens.unsqueeze(1)
        return gathered[valid_mask]

    def _find_token_prefix_len_before_text_boundary(
        self,
        token_ids: list[int],
        full_text: str,
        boundary_text: str,
        include_boundary: bool = False,
    ) -> int:
        boundary_pos = full_text.rfind(boundary_text)
        if boundary_pos < 0:
            return len(token_ids)
        
        cut_pos = boundary_pos + len(boundary_text) if include_boundary else boundary_pos
        
        token_ids = list(token_ids)
        tokenizer = self._tokenizer

        # Fast path: exact round-trip + offsets
        try:
            if getattr(tokenizer, "is_fast", False):
                encoded = tokenizer(
                    full_text,
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_offsets_mapping=True,
                )

                encoded_ids = encoded["input_ids"]
                if hasattr(encoded_ids, "tolist"):
                    encoded_ids = encoded_ids.tolist()

                # Now full_text preserves special tokens,
                # so it should match the ORIGINAL token IDs.
                if encoded_ids == token_ids:
                    offsets = encoded["offset_mapping"]
                    prefix_len = 0
                    for start, end in offsets:
                        if end <= cut_pos:
                            prefix_len += 1
                        else:
                            break

                    return prefix_len
                
        except (TypeError, ValueError, NotImplementedError):
            pass

        # Robust fallback.        
        best = 0

        for n in range(1, len(token_ids) + 1):
            prefix_text = tokenizer.decode(
                token_ids[:n],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            if (
                len(prefix_text) <= cut_pos
                and full_text.startswith(prefix_text)
            ):
                best = n

        return best

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        prompts = [x["prompt"] for x in inputs]

        (
            prompt_ids_list,
            completion_ids_list,
            completions,
            sampling_per_token_logps_list, # None
            extra_fields, # {}
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(
                sampling_per_token_logps,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            ).to(device=device)
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions for attention and loss masking
        # 9/16
        eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]

        is_truncated = torch.tensor(
            [(len(ids) > 0 and ids[-1] not in eos_and_pad) for ids in completion_ids_list],
            device=device,
            dtype=torch.bool,
        )

        valid_response_mask = (
            ~is_truncated
            if self.mask_truncated_completions
            else torch.ones_like(is_truncated, dtype=torch.bool)
        )

        if self.mask_truncated_completions:
            completion_mask = (
                completion_mask
                * valid_response_mask.unsqueeze(1).to(completion_mask.dtype)
            )

        num_items_in_batch = self.accelerator.gather(completion_mask.sum()).sum()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        # When gradient checkpointing is enabled with use_reentrant=True (non default), calling the model inside a
        # torch.no_grad() block triggers a harmless PyTorch warning ("None of the inputs have requires_grad=True").
        # Temporarily disable checkpointing to avoid this warning during inference.
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                mask = completion_mask
                per_token_logps_diff = (old_per_token_logps - sampling_per_token_logps) * mask
                # Tokens whose sampling logprob was NaN (unavailable from vLLM) get a zero difference, so their
                # importance ratio is exactly 1 (no correction) rather than propagating NaN through the loss.
                per_token_logps_diff = torch.nan_to_num(per_token_logps_diff, nan=0.0)

                sequence_level_is = self.vllm_importance_sampling_mode in ["sequence_mask", "sequence_truncate"]
                if sequence_level_is:
                    per_sequence_logps_diff = per_token_logps_diff.sum(dim=-1, keepdim=True)
                    logps_diff = per_sequence_logps_diff
                else:
                    logps_diff = per_token_logps_diff

                vllm_importance_sampling_ratio = torch.exp(logps_diff)

                # vllm_importance_sampling_ratio.shape:
                #   token_* modes:     (B, T)  (per-token ratio)
                #   sequence_* modes:  (B, 1)  (per-sequence ratio)

                if self.vllm_importance_sampling_mode in ["sequence_truncate", "token_truncate"]:
                    vllm_importance_sampling_ratio = torch.clamp(
                        vllm_importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                    )
                elif self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]:
                    vllm_importance_sampling_ratio = vllm_importance_sampling_ratio.masked_fill(
                        vllm_importance_sampling_ratio > self.vllm_importance_sampling_cap, value=0.0
                    )
                else:
                    raise ValueError(
                        f"Unknown vLLM importance sampling level: {self.vllm_importance_sampling_mode}. Possible values are 'token_truncate', 'token_mask', 'sequence_truncate', and 'sequence_mask'."
                    )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size
                    )
                else:
                    # When training a PEFT adapter, how we obtain the reference depends on the setup:
                    # - New adapter: disabling adapters yields the base model.
                    # - Re-training an existing adapter: an initial copy is loaded under the name "ref".
                    unwrapped_model = self.accelerator.unwrap_model(self.model)
                    with use_adapter(unwrapped_model, adapter_name="ref" if "ref" in unwrapped_model.peft_config else None):
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.

        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        reward_extras = {}
        # 9/16
        # rewards_per_func has already been gathered across all processes,
        # so gather the validity mask in exactly the same order.
        all_valid_response_mask = gather(
            valid_response_mask.to(torch.long)
        ).bool()

        rewards_per_func_for_baseline = rewards_per_func.clone()

        # Truncated responses should not contribute to task-reward logging/statistics.
        rewards_per_func = rewards_per_func.masked_fill(
            ~all_valid_response_mask.unsqueeze(1),
            torch.nan,
        )

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval

        if self.multi_objective_aggregation == "sum_then_normalize":
            # Apply weights to each reward function's output and sum
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            '''mean_grouped_rewards = rewards.view(-1, num_generations).mean(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            if self.scale_rewards in ["group", "none"]:
                # If self.scale_rewards = "none", we'll only use std_rewards to check for zero std for logging
                if num_generations > 1:
                    std_rewards = rewards.view(-1, num_generations).std(dim=1)
                    std_rewards = std_rewards.repeat_interleave(num_generations, dim=0)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=1
                    std_rewards = torch.zeros_like(rewards)
            elif self.scale_rewards == "batch":
                # Compute global std
                if rewards.numel() > 1:
                    std_rewards = rewards.std().expand_as(rewards)
                else:  # doesn't occur during training, but could occur in eval when num_generations_eval=batch_size=1
                    std_rewards = torch.zeros_like(rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
                )

            advantages = rewards - mean_grouped_rewards
            if self.scale_rewards != "none":
                advantages = advantages / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging'''
            # 9/16
            grouped_rewards = rewards.view(-1, num_generations)
            grouped_valid = all_valid_response_mask.view(-1, num_generations)
            valid_f = grouped_valid.to(grouped_rewards.dtype)

            # -------------------------------------------------------------
            # Baseline-only truncation shaping
            #
            # Use PRE-MASK rewards to compute normalization statistics.
            # Truncated responses affect mean/std, but their own final
            # advantage is masked to zero below.
            # -------------------------------------------------------------
            baseline_rewards = (
                rewards_per_func_for_baseline
                * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)

            grouped_baseline_rewards = baseline_rewards.view(-1, num_generations)

            # Group baseline includes truncated responses exactly as in the
            # old normalization behavior.
            group_mean = grouped_baseline_rewards.mean(dim=1, keepdim=True)

            # Only valid responses retain an optimization signal.
            centered = (grouped_rewards - group_mean) * valid_f

            if self.scale_rewards == "none":
                grouped_advantages = centered
                # std is only needed for logging in this branch.
                if num_generations > 1:
                    group_std = grouped_baseline_rewards.std(dim=1, keepdim=True)
                else:
                    group_std = torch.zeros_like(group_mean)
                std_rewards_grouped = group_std.expand_as(grouped_rewards)
            elif self.scale_rewards == "group":
                if num_generations > 1:
                    group_std = grouped_baseline_rewards.std(dim=1, keepdim=True)
                else:
                    # Important for eval when num_generations_eval == 1.
                    group_std = torch.zeros_like(group_mean)
                grouped_advantages = centered / (group_std + 1e-4)
                std_rewards_grouped = group_std.expand_as(grouped_rewards)
            elif self.scale_rewards == "batch":
                if baseline_rewards.numel() > 1:
                    batch_std = baseline_rewards.std()
                else:
                    batch_std = rewards.new_zeros(())
                grouped_advantages = centered / (batch_std + 1e-4)
                std_rewards_grouped = batch_std.expand_as(grouped_rewards)
            else:
                raise ValueError(
                    f"Invalid value for scale_rewards: "
                    f"{self.scale_rewards}. "
                    "Must be one of 'batch', 'group', or 'none'."
                )

            # Critical:
            # truncated responses influence baseline statistics,
            # but NEVER receive their own policy advantage.
            grouped_advantages = grouped_advantages.masked_fill(~grouped_valid, 0.0)
            advantages = grouped_advantages.reshape(-1)
            mean_grouped_rewards = group_mean.expand_as(grouped_rewards).reshape(-1)
            std_rewards = std_rewards_grouped.reshape(-1)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))

        elif self.multi_objective_aggregation == "normalize_then_sum":
            grouped = rewards_per_func.view(-1, num_generations, len(self.reward_funcs))
            mean_k = torch.nanmean(grouped, dim=1, keepdim=True)
            std_k = nanstd(grouped, dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(mean_k)
            reward_k = (grouped - mean_k) / (std_k + 1e-4)
            reward_k = reward_k.view(-1, len(self.reward_funcs))
            rewards = (reward_k * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            std_rewards = rewards.std().expand_as(rewards) if rewards.numel() > 1 else torch.zeros_like(rewards)
            advantages = (rewards - rewards.mean()) / (std_rewards + 1e-4)
            is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))  # for logging

        else:
            raise ValueError(
                f"Invalid multi_objective_aggregation: {self.multi_objective_aggregation}. Must be "
                "'sum_then_normalize' or 'normalize_then_sum'."
            )
        
        len_only_reasoning_completions = []

        for completion_text, completion_id in zip(completions, completion_ids_list, strict=True):
            if "\\boxed{" in completion_text:
                len_only_reasoning = self._find_token_prefix_len_before_text_boundary(
                    token_ids=completion_id,
                    full_text=completion_text,
                    boundary_text="\\boxed{",
                    include_boundary=True
                )
            else:
                # If no boxed answer exists, treat the whole completion as reasoning.
                # Then a negative advantage can still penalize this invalid behavior.
                len_only_reasoning = len(completion_id)

            len_only_reasoning_completions.append(len_only_reasoning)
        
        len_only_reasoning_completions = torch.tensor(len_only_reasoning_completions, device=device, dtype=torch.long)
        
        reasoning_idx = torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
        reasoning_mask = (reasoning_idx < len_only_reasoning_completions.unsqueeze(1)).to(dtype=completion_mask.dtype)
        reasoning_mask = reasoning_mask * completion_mask

        if self.args.modulation is True:
            with torch.inference_mode():
                # Prepare self teacher prompt inputs
                self_teacher_prompts = [x["self_teacher_prompt"] for x in inputs]
                self_teacher_inputs = self.processing_class(
                    text=self_teacher_prompts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False
                )
                self_teacher_inputs = super()._prepare_inputs(self_teacher_inputs)
                self_teacher_ids, self_teacher_mask = self_teacher_inputs["input_ids"], self_teacher_inputs["attention_mask"]

                # This is for current - current difference
                # ===== Current-policy unconditioned token logps =====
                # We want:
                # log pi_theta(y_t | x+, y_<t) - log pi_theta(y_t | x,  y_<t)
                # Reuse old_per_token_logps when it is already available.
                # Otherwise compute the ordinary-prompt current-policy logps.
                if old_per_token_logps is not None:
                    current_uncond_per_token_logps = old_per_token_logps
                else:
                    current_uncond_per_token_logps, _ = (
                        self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                        )
                    )
                
                # Prepare modulation inputs
                modulation_logits_to_keep = int(len_only_reasoning_completions.max().item())

                only_reasoning_completions_mask = (
                    reasoning_idx[:, :modulation_logits_to_keep] < len_only_reasoning_completions.unsqueeze(1)
                ).to(dtype=completion_mask.dtype)

                only_reasoning_completions_mask = (
                    only_reasoning_completions_mask * completion_mask[:, :modulation_logits_to_keep]
                )

                only_reasoning_completions_ids = completion_ids[:, :modulation_logits_to_keep].clone()

                only_reasoning_completions_ids = only_reasoning_completions_ids.masked_fill(
                    ~only_reasoning_completions_mask.bool(),
                    self._tokenizer.pad_token_id,
                )

                self_teacher_completion_ids = torch.cat([self_teacher_ids, only_reasoning_completions_ids], dim=1) # [B, pad + self_teacher] cat [B, only_reason + pad]
                self_teacher_attention_mask = torch.cat([self_teacher_mask, only_reasoning_completions_mask], dim=1)

                if modulation_logits_to_keep == 0:
                    # No trainable reasoning-prefix tokens are available.
                    modulation_per_token_logps = torch.empty(
                        (self_teacher_completion_ids.size(0), 0),
                        device=device,
                        dtype=ref_per_token_logps.dtype, # dangerous
                    )
                else:
                    # This is for current - reference difference and current - current difference
                    modulation_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        self_teacher_completion_ids,
                        self_teacher_attention_mask,
                        modulation_logits_to_keep,
                        batch_size=batch_size,
                    )
                    '''
                    # This is for reference - reference difference
                    unwrapped_model = self.accelerator.unwrap_model(self.model)

                    with use_adapter(
                        unwrapped_model,
                        adapter_name=("ref" if "ref" in unwrapped_model.peft_config else None),
                    ):
                        modulation_per_token_logps, _ = (
                            self._get_per_token_logps_and_entropies(
                                self.model,
                                self_teacher_completion_ids,
                                self_teacher_attention_mask,
                                modulation_logits_to_keep,
                                batch_size=batch_size,
                            )
                        )
                    '''


                # modulation shaping value
                # This is for current - reference difference
                # modulation_log_ratio = modulation_per_token_logps - ref_per_token_logps[:, :modulation_logits_to_keep]

                # This is for current - current difference
                modulation_log_ratio = modulation_per_token_logps - current_uncond_per_token_logps[:, :modulation_logits_to_keep]

                # This is for reference - reference difference
                # modulation_log_ratio = modulation_per_token_logps - ref_per_token_logps[:, :modulation_logits_to_keep]
                
                modulation_mask = only_reasoning_completions_mask.to(modulation_per_token_logps.dtype)

                modulation_log_ratio_mean = (modulation_log_ratio * modulation_mask).sum(dim=1, keepdim=True) / modulation_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                modulation_centered = (modulation_log_ratio - modulation_log_ratio_mean) * modulation_mask

                local_sum_sq = modulation_centered.square().sum()
                local_token_count = modulation_mask.sum()

                local_stats = torch.stack([local_sum_sq.float(), local_token_count.float()]).detach()

                global_stats = self.accelerator.reduce(local_stats, reduction="sum")
                shared_var = global_stats[0] / global_stats[1].clamp_min(1.0)

                modulation_shared_std = (shared_var.sqrt().clamp_min(1e-6).to(dtype=modulation_centered.dtype))
                modulation_delta = (modulation_centered / modulation_shared_std).detach()

                # Log modulation shaping related info
                valid_modulation_mask = modulation_mask.bool()
                valid_modulation_log_ratio = modulation_log_ratio[valid_modulation_mask].float()
                valid_modulation_delta = modulation_delta[valid_modulation_mask].float()
                local_stats = torch.stack([
                    valid_modulation_log_ratio.sum(),
                    valid_modulation_log_ratio.square().sum(),
                    valid_modulation_delta.abs().sum(),
                    torch.isnan(valid_modulation_delta).sum().float(),
                    torch.isinf(valid_modulation_delta).sum().float(),
                    torch.tensor(valid_modulation_delta.numel(), device=device, dtype=torch.float32),
                ])

                global_stats = self.accelerator.reduce(local_stats, reduction="sum",)

                (
                    log_ratio_sum,
                    log_ratio_sq_sum,
                    delta_abs_sum,
                    nan_count,
                    inf_count,
                    valid_count,
                ) = global_stats

                valid_count = valid_count.clamp_min(1.0)

                log_ratio_mean = log_ratio_sum / valid_count
                log_ratio_var = (log_ratio_sq_sum / valid_count - log_ratio_mean.square()).clamp_min(0.0)

                self._metrics[mode]["modulation/log_ratio_mean"].append(log_ratio_mean.item())
                self._metrics[mode]["modulation/log_ratio_std"].append(log_ratio_var.sqrt().item())
                self._metrics[mode]["modulation/shared_std"].append(modulation_shared_std.float().item())
                self._metrics[mode]["modulation/delta_abs_mean"].append((delta_abs_sum / valid_count).item())
                self._metrics[mode]["modulation/nan_ratio"].append((nan_count / valid_count).item())
                self._metrics[mode]["modulation/inf_ratio"].append((inf_count / valid_count).item())

        
        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        #rewards = (rewards_per_func * self.reward_weights.to(rewards_per_func.device).unsqueeze(0)).nansum(dim=1)
        #self._metrics[mode]["reward"].append(rewards.mean().item())
        # 9/16
        valid_rewards = rewards[all_valid_response_mask]

        if valid_rewards.numel() > 0:
            self._metrics[mode]["reward"].append(
                valid_rewards.mean().item()
            )
            self._metrics[mode]["reward_std"].append(
                valid_rewards.std(unbiased=False).item()
            )
        else:
            self._metrics[mode]["reward"].append(0.0)
            self._metrics[mode]["reward_std"].append(0.0)

        self._metrics[mode]["responses/valid_ratio"].append(
            all_valid_response_mask.float().mean().item()
        )

        group_valid_count = (
            all_valid_response_mask
            .view(-1, num_generations)
            .sum(dim=1)
            .float()
        )

        self._metrics[mode]["responses/valid_per_group_mean"].append(
            group_valid_count.mean().item()
        )

        self._metrics[mode]["responses/groups_lt2_valid_ratio"].append(
            (group_valid_count < 2).float().mean().item()
        )

        #self._metrics[mode]["reward_std"].append(rewards.std().item())
        #self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())
        valid_std_zero = is_std_zero[all_valid_response_mask]
        self._metrics[mode]["frac_reward_zero_std"].append(
            valid_std_zero.float().mean().item() if valid_std_zero.numel() > 0 else 1.0
        )

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = self.accelerator.gather(torch.tensor(local_mean, device=device)).mean().item()
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()
        
        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            mask = completion_mask.bool()
            delta = delta[mask]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )
            if sequence_level_is:
                flat_is_ratio = vllm_importance_sampling_ratio.flatten()
            else:
                flat_is_ratio = vllm_importance_sampling_ratio[mask]

            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "reasoning_mask": reasoning_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = vllm_importance_sampling_ratio
        if sampling_per_token_logps is not None:
            output["sampling_per_token_logps"] = sampling_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if self.args.modulation is True:
            output["modulation_delta"] = modulation_delta
            output["modulation_mask"] = modulation_mask
        return output
    
    def _generate_and_no_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "eval"

        prompts = [x["prompt"] for x in inputs]

        (
            prompt_ids_list,
            completion_ids_list,
            completions,
            sampling_per_token_logps_list, # None
            extra_fields, # {}
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)

        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values


        # Flush user-logged extra columns (from log_extra), gathering across processes.
        # Keys must be sorted so that all ranks call gather_object in the same order, otherwise values
        # get mis-attributed across columns (dict insertion order may differ between processes).
        for column in sorted(self._pending_extra_logs):
            self._logs["extra"][column].extend(gather_object(self._pending_extra_logs[column]))
        self._pending_extra_logs.clear()

        # Flush user-logged metrics (from log_metric), averaging across processes.
        # Keys must be sorted so that all ranks call accelerator.gather in the same order, otherwise values
        # get mis-attributed across metrics (dict insertion order may differ between processes).
        for name in sorted(self._pending_metrics):
            values = self._pending_metrics[name]
            local_mean = sum(values) / len(values)
            global_mean = self.accelerator.gather(torch.tensor(local_mean, device=device)).mean().item()
            self._metrics[mode][name].append(global_mean)
        self._pending_metrics.clear()
        output = {
            "completion_ids": completion_ids,
        }
        return output

    @torch.inference_mode()
    def _generate_and_analyze_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        (
            prompt_ids_list,
            completion_ids_list,
            completions,
            sampling_per_token_logps_list, # None
            extra_fields, # {}
        ) = self._generate(prompts)

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(
            prompt_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="left",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        prompt_mask = pad(
            prompt_mask, padding_value=0, padding_side="left", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)

        completion_ids = [torch.tensor(ids) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(
            completion_ids,
            padding_value=self._tokenizer.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        ).to(device=device)
        completion_mask = pad(
            completion_mask, padding_value=0, padding_side="right", pad_to_multiple_of=self.pad_to_multiple_of
        ).to(device=device)

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_eval_batch_size

        # Use the SAME temperature as the training-time scoring distribution.
        # Current checkpoints were trained with temperature=1.0.
        scoring_temperature = self.temperature
        self.temperature = 1.0

        try:
            uncond_logps, entropies = self._get_per_token_logps_and_entropies(
                self.model,
                prompt_completion_ids,
                attention_mask,
                logits_to_keep,
                batch_size=batch_size,
                compute_entropy=True,
            )

            len_only_reasoning_completions = []
            
            for completion_text, completion_id in zip(completions, completion_ids_list, strict=True):
                if "\\boxed{" in completion_text:
                    len_only_reasoning = self._find_token_prefix_len_before_text_boundary(
                        token_ids=completion_id,
                        full_text=completion_text,
                        boundary_text="\\boxed{",
                        include_boundary=True
                    )
                else:
                    # If no boxed answer exists, treat the whole completion as reasoning.
                    # Then a negative advantage can still penalize this invalid behavior.
                    len_only_reasoning = len(completion_id)
    
                len_only_reasoning_completions.append(len_only_reasoning)
            
            len_only_reasoning_completions = torch.tensor(len_only_reasoning_completions, device=device, dtype=torch.long)
            
            reasoning_idx = torch.arange(completion_ids.size(1), device=device).unsqueeze(0)
            reasoning_mask = (reasoning_idx < len_only_reasoning_completions.unsqueeze(1)).to(dtype=completion_mask.dtype)
            reasoning_mask = reasoning_mask * completion_mask
            
            # Prepare self teacher prompt inputs
            self_teacher_prompts = [x["self_teacher_prompt"] for x in inputs]
            self_teacher_inputs = self.processing_class(
                text=self_teacher_prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False
            )
            self_teacher_inputs = super()._prepare_inputs(self_teacher_inputs)
            self_teacher_ids, self_teacher_mask = self_teacher_inputs["input_ids"], self_teacher_inputs["attention_mask"]

            reasoning_len = int(len_only_reasoning_completions.max().item())
            reasoning_token_mask = reasoning_mask[:, :reasoning_len].bool()

            only_reasoning_ids = completion_ids[:, :reasoning_len].clone()
            only_reasoning_ids = only_reasoning_ids.masked_fill(~reasoning_token_mask, self._tokenizer.pad_token_id)
            self_teacher_completion_ids = torch.cat([self_teacher_ids, only_reasoning_ids], dim=1)
            self_teacher_attention_mask = torch.cat(
                [self_teacher_mask, reasoning_token_mask.to(self_teacher_mask.dtype)], dim=1
            )

            if reasoning_len == 0:
                cond_logps = torch.empty(
                    (completion_ids.size(0), 0), device=device, dtype=uncond_logps.dtype
                )
            else:
                cond_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    self_teacher_completion_ids,
                    self_teacher_attention_mask,
                    reasoning_len,
                    batch_size=batch_size,
                    compute_entropy=False,
                )

            # Eq. (7): raw answer-conditioned compatibility signal r_tok.
            modulation_raw = cond_logps - uncond_logps[:, :reasoning_len]
            modulation_mask_f = reasoning_token_mask.to(modulation_raw.dtype)

            # Eqs. (8)-(10): response-wise centering + shared batch scale.
            modulation_mean = (
                (modulation_raw * modulation_mask_f).sum(dim=1, keepdim=True)
                / modulation_mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            )
            modulation_centered = (modulation_raw - modulation_mean) * modulation_mask_f
            local_stats = torch.stack(
                [modulation_centered.square().sum().float(), modulation_mask_f.sum().float()]
            ).detach()
            global_stats = self.accelerator.reduce(local_stats, reduction="sum")
            shared_var = global_stats[0] / global_stats[1].clamp_min(1.0)
            shared_std = shared_var.sqrt().clamp_min(1e-6).to(modulation_centered.dtype)
            modulation_delta = (modulation_centered / shared_std).detach()

            # Eqs. (17)-(18): entropy-selective positions.
            if self.top_entropy_quantile < 1.0:
                entropy_mask_full = self.get_high_entropy_mask(
                    entropies,
                    reasoning_mask,
                    1 - self.top_entropy_quantile,
                )
                entropy_selected = entropy_mask_full[:, :reasoning_len] & reasoning_token_mask
            else:
                entropy_selected = reasoning_token_mask.clone()

            # Eq. (14), or identity when controlled_modulation=False.
            modulation_weight = self._modulation_delta_to_redistribution_weight(
                modulation_delta=modulation_delta,
                modulation_mask=entropy_selected,
                controlled=self.args.controlled_modulation,
            ).detach()

            reasoning_entropies = entropies[:, :reasoning_len]

            # JSON-serializable side channel. Include both raw and normalized/projected
            # scores so later plots can distinguish "compatibility" from "actually applied credit".
            local_records = []
            for b in range(completion_ids.size(0)):
                n = int(len_only_reasoning_completions[b].item())
                token_ids_b = completion_ids[b, :n].detach().cpu().tolist()
                token_text_b = [
                    self._tokenizer.decode(
                        [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    for tid in token_ids_b
                ]
                local_records.append(
                    {
                        "is_correct": inputs[b].get("is_correct"),
                        "question": inputs[b].get("question"),
                        "answer": inputs[b].get("answer"),
                        "completion": completions[b],
                        "reasoning_token_ids": token_ids_b,
                        "reasoning_tokens": token_text_b,
                        "modulation_raw": modulation_raw[b, :n].float().cpu().tolist(),
                        "modulation_score": modulation_delta[b, :n].float().cpu().tolist(),
                        "modulation_weight": modulation_weight[b, :n].float().cpu().tolist(),
                        "entropy": reasoning_entropies[b, :n].float().cpu().tolist(),
                        "entropy_selected": entropy_selected[b, :n].cpu().tolist(),
                        "top_entropy_quantile": float(self.top_entropy_quantile),
                        "controlled_modulation": bool(self.args.controlled_modulation),
                        "analysis_scoring_temperature": float(self.temperature),
                        "modulation_shared_std": float(shared_std.float().item()),
                    }
                )

            gathered_records = gather_object(local_records)
            if self.accelerator.is_main_process:
                self.analysis_records.extend(gathered_records)

        finally:
            self.temperature = scoring_temperature

        return {"completion_ids": completion_ids}

    # TRL
    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        # Prepares inputs for model training/evaluation by managing completion generation and batch handling.
        # During training:
        #   - Receives the local generation batch (Per-GPU batch size × steps per generation)
        #     from the modified training dataloader instead of the standard local batch
        #   - Generates completions once for the entire generation batch and splits it into batches of size
        #     `per_device_train_batch_size`
        #   - Buffers these completions and returns the appropriate slice for the current accumulation step
        #   - Optimizes by regenerating completions only periodically (every steps_per_generation * num_iterations)
        # During evaluation:
        #   - The input is treated as a standard local batch (no accumulation, no multiple iterations)
        #   - Completions are generated for each batch without buffering or reuse
        # Returns a single local batch in both cases.

        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # self._buffered_inputs=None can occur when resuming from a checkpoint
                generation_batch = self._generate_and_score_completions(generation_batch)
                generation_batch = shuffle_sequence_dict(generation_batch)
                generation_batches = split_tensor_dict(generation_batch, self.args.steps_per_generation)
                self._buffered_inputs = [batch for batch in generation_batches]
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations, hence
            # local generation batch == local eval batch
            if self.only_generate and self.args.analysis:
                inputs = self._generate_and_analyze_completions(generation_batch)

            elif self.only_generate:
                inputs = self._generate_and_no_score_completions(generation_batch)

            else:
                inputs = self._generate_and_score_completions(generation_batch)
        return inputs

    # TRL
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        if self.use_liger_kernel:
            # Compute the loss using the liger grpo loss
            unwrapped_model = self.accelerator.unwrap_model(model)
            return self._forward_redirection(model, unwrapped_model, self.compute_liger_loss, unwrapped_model, inputs)
        else:
            return self._compute_loss(model, inputs)
    
    def _modulation_delta_to_redistribution_weight(
        self,
        modulation_delta: torch.Tensor,
        modulation_mask: torch.Tensor,
        controlled: bool,
    ) -> torch.Tensor:
        if modulation_delta.size(1) == 0:
            return torch.zeros_like(modulation_delta)

        modulation_mask = modulation_mask.to(device=modulation_delta.device).bool()
        mask_f = modulation_mask.to(dtype=modulation_delta.dtype)

        delta = modulation_delta.detach() * mask_f

        if controlled:
            x = delta.float()
            mask_fp32 = modulation_mask.to(dtype=x.dtype)

            low = x.amin(dim=-1, keepdim=True) - 1.0
            high = x.amax(dim=-1, keepdim=True) + 1.0

            for _ in range(32):
                tau = (low + high) * 0.5
                projected = (x - tau).clamp(-1.0, 1.0) * mask_fp32
                summed = projected.sum(dim=-1, keepdim=True)

                low = torch.where(summed > 0, tau, low)
                high = torch.where(summed > 0, high, tau)

            tau = (low + high) * 0.5
            delta_norm = (
                (x - tau).clamp(-1.0, 1.0) * mask_fp32
            ).to(dtype=delta.dtype)
        else:
            delta_norm = delta

        return delta_norm

    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        reasoning_mask = inputs.get("reasoning_mask", completion_mask)
        reasoning_mask = reasoning_mask.to(device=completion_mask.device, dtype=completion_mask.dtype)
        mask = completion_mask * reasoning_mask

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the loss
        advantages = inputs["advantages"]
        # In the base GRPO implementation, advantages are expected to have shape (B,). To support subclasses that
        # provide advantages with shape (B, T) (e.g., MiniLLM), we *conditionally* unsqueeze the tensor.
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        if self.off_policy_mask_threshold is not None:
            # OPSM should use inference-time logprobs to detect both sources of off-policyness:
            # 1. Drift from gradient updates (always present)
            # 2. Drift from training-inference mismatch (when using vLLM)
            # When using vLLM, prioritize sampling_per_token_logps, otherwise use old_per_token_logps
            sampling_per_token_logps = inputs.get("sampling_per_token_logps", old_per_token_logps)

            off_policy_mask = self.get_off_policy_mask(
                advantages=advantages,
                per_token_logps=per_token_logps,
                sampling_per_token_logps=sampling_per_token_logps,
                mask=mask,
                off_policy_threshold=self.off_policy_mask_threshold,
            )

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )

        coef_1 = torch.exp(log_importance_weights)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
            # Importance sampling correction for the KL divergence
            if self.args.use_bias_correction_kl:
                per_token_kl = per_token_kl * coef_1

        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)
        if self.loss_type == "cispo":
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            per_token_loss = -clamped_ratios * advantages * per_token_logps
        elif self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            # Two-sided clipping
            if self.args.delta is not None:
                coef_1 = torch.clamp(coef_1, max=self.args.delta)

            if self.args.modulation is False:
                per_token_loss1 = coef_1 * advantages
                per_token_loss2 = coef_2 * advantages
                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            else:
                modulation_delta = inputs['modulation_delta']
                modulation_mask = inputs["modulation_mask"].bool()

                reasoning_len = modulation_delta.size(1)

                if entropy_mask is not None:
                    reasoning_entropy_mask = entropy_mask[:, :reasoning_len].to(device=advantages.device).bool()
                    refinement_mask = modulation_mask & reasoning_entropy_mask
                else:
                    refinement_mask = modulation_mask
                

                used_modulation_delta = self._modulation_delta_to_redistribution_weight(
                    modulation_delta=modulation_delta,
                    modulation_mask=refinement_mask,
                    controlled=self.args.controlled_modulation
                )

                if self.args.perturbation == "shuffle":
                    # ===== Analysis: Shuffled-modulation control =====
                    if model.training:
                        with torch.no_grad():
                            shuffled_modulation_delta = used_modulation_delta.clone()

                            for b in range(used_modulation_delta.size(0)):
                                valid_idx = torch.nonzero(refinement_mask[b], as_tuple=False,).flatten()

                                if valid_idx.numel() > 1:
                                    perm = torch.randperm(valid_idx.numel(), device=valid_idx.device,)
                                    shuffled_modulation_delta[b, valid_idx] = (used_modulation_delta[b, valid_idx[perm],])

                            used_modulation_delta = shuffled_modulation_delta
                    # ===== Analysis: End Shuffled-modulation control =====
                elif self.args.perturbation == "reverse":
                    # ===== Ablation-2: Reverse-modulation control =====
                    if model.training:
                        used_modulation_delta = -used_modulation_delta
                    # ===== Ablation-2: End Reverse-modulation control =====

                if self.args.controlled_modulation is True:
                    bounded_modulation_bonus = self.args.modulation_lambda * advantages.abs() * used_modulation_delta
                else:
                    bounded_modulation_bonus = self.args.modulation_lambda * used_modulation_delta

                reasoning_advantages = advantages + bounded_modulation_bonus

                reasoning_per_token_loss1 = coef_1[:, :reasoning_len] * reasoning_advantages
                answer_token_loss1 = coef_1[:, reasoning_len:] * advantages
                per_token_loss1 = torch.cat([reasoning_per_token_loss1, answer_token_loss1], dim=-1)
                
                reasoning_per_token_loss2 = coef_2[:, :reasoning_len] * reasoning_advantages
                answer_per_token_loss2 = coef_2[:, reasoning_len:] * advantages
                per_token_loss2 = torch.cat([reasoning_per_token_loss2, answer_per_token_loss2], dim=-1)

                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

                # ===== Diagnostic: PPO-space effect of modulation =====
                with torch.no_grad():
                    modulation_base_ppo_loss1 = coef_1.detach() * advantages.detach()
                    modulation_base_ppo_loss2 = coef_2.detach() * advantages.detach()
                    modulation_base_ppo_loss = -torch.min(modulation_base_ppo_loss1, modulation_base_ppo_loss2,)
                    modulation_ppo_residual = per_token_loss.detach() - modulation_base_ppo_loss
                # ===== End diagnostic =====

                valid_adv_mask = refinement_mask & (advantages.abs() > 1e-8)
                modulation_advantage_flip_mask = (
                    (reasoning_advantages * advantages < 0) & valid_adv_mask
                )

        elif self.loss_type == "sapo":
            temperatures = torch.where(advantages > 0, self.args.sapo_temperature_pos, self.args.sapo_temperature_neg)
            soft_coef_1 = torch.sigmoid(temperatures * (coef_1 - 1)) * 4 / temperatures
            per_token_loss = -soft_coef_1 * advantages
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.off_policy_mask_threshold is not None:
            per_token_loss = per_token_loss * off_policy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        mode = "train" if self.model.training else "eval"
        if self.loss_type in ["grpo", "sapo"]:
            loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1.0)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
            loss = loss / normalizer
        elif self.loss_type in ["cispo", "dapo"]:
            normalizer = inputs["num_items_in_batch"].clamp(min=1.0) / self.accelerator.num_processes
            if mode == "train":  # in eval, the batch is neither split across steps nor accumulated
                normalizer = normalizer * self.current_gradient_accumulation_steps / self.args.steps_per_generation
            loss = (per_token_loss * mask).sum() / normalizer
        elif self.loss_type == "luspo":
            # Unless importance_sampling_level="token" (not recommended here), per_token_loss is expected to be (B, 1)
            loss = (per_token_loss * mask.sum(1, keepdim=True)).mean()
            normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0
            loss = loss / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if self.args.modulation is True:
            valid_final_delta = used_modulation_delta[refinement_mask]
            valid_entropy_gate = refinement_mask[modulation_mask].float()

            global_final_delta = self.gather_variable_length_1d(valid_final_delta, pad_value=1e30)
            global_entropy_gate = self.gather_variable_length_1d(valid_entropy_gate, pad_value=1e30)

            def _safe_abs_mean(x: torch.Tensor) -> float:
                return x.abs().mean().item() if x.numel() > 0 else 0.0

            def _safe_std(x: torch.Tensor) -> float:
                return x.std(unbiased=False).item() if x.numel() > 0 else 0.0

            def _log_scalar(name: str, value: float):
                self._metrics[mode][name].append(float(value))

            with torch.no_grad():
                valid_policy_mask = mask.bool()

                local_ppo_residual = (modulation_ppo_residual[valid_policy_mask].detach().float())
                local_base_ppo_loss = (modulation_base_ppo_loss[valid_policy_mask].detach().float())

                global_ppo_residual = self.gather_variable_length_1d(local_ppo_residual, pad_value=1e30,)
                global_base_ppo_loss = self.gather_variable_length_1d(local_base_ppo_loss, pad_value=1e30,)

                if (global_ppo_residual.numel() > 0 and global_base_ppo_loss.numel() > 0):
                    ppo_residual_relative_abs = (
                        global_ppo_residual.abs().mean()
                        / global_base_ppo_loss.abs().mean().clamp_min(1e-12)
                    ).item()
                else:
                    ppo_residual_relative_abs = 0.0

                policy_mask_f = mask.to(dtype=modulation_ppo_residual.dtype)
                seq_den = policy_mask_f.sum(dim=-1).clamp_min(1.0)

                local_seq_residual = (modulation_ppo_residual * policy_mask_f).sum(dim=-1) / seq_den

                valid_seq_mask = policy_mask_f.sum(dim=-1) > 0

                global_seq_residual = self.gather_variable_length_1d(
                    local_seq_residual[valid_seq_mask].detach().float(),
                    pad_value=1e30,
                )

                ppo_residual_seq_abs_mean = (
                    global_seq_residual.abs().mean().item()
                    if global_seq_residual.numel() > 0
                    else 0.0
                )

                _log_scalar("modulation/ppo_residual_relative_abs", ppo_residual_relative_abs)
                _log_scalar("modulation/ppo_residual_seq_abs_mean", ppo_residual_seq_abs_mean)

            _log_scalar("modulation/final_delta_abs_mean", _safe_abs_mean(global_final_delta))
            _log_scalar("modulation/final_delta_std", _safe_std(global_final_delta))
            _log_scalar(
                "modulation/entropy_gate_token_ratio",
                global_entropy_gate.mean().item() if global_entropy_gate.numel() > 0 else 0.0,
            )

            valid_flip = modulation_advantage_flip_mask[valid_adv_mask].float()

            global_flip = self.gather_variable_length_1d(valid_flip, pad_value=1e30)

            _log_scalar(
                "modulation/advantage_flip_ratio",
                global_flip.mean().item() if global_flip.numel() > 0 else 0.0,
            )

            global_abs_omega = global_final_delta.abs().float()
            global_abs_omega = global_abs_omega[torch.isfinite(global_abs_omega)]

            if global_abs_omega.numel() > 0:
                _log_scalar("modulation/omega_abs_p50", torch.quantile(global_abs_omega, 0.50).item())
                _log_scalar("modulation/omega_abs_p90", torch.quantile(global_abs_omega, 0.90).item())
                _log_scalar("modulation/omega_abs_p99", torch.quantile(global_abs_omega, 0.99).item())
            else:
                _log_scalar("modulation/omega_abs_p50", 0.0)
                _log_scalar("modulation/omega_abs_p90", 0.0)
                _log_scalar("modulation/omega_abs_p99", 0.0)

        # Log the metrics
        completion_token_count = mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        if self.loss_type in ["grpo", "bnpo", "dr_grpo", "dapo", "luspo"]:
            # Compute the clipped probability ratios
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages > 0)
            is_region_clipped = is_low_clipped | is_high_clipped

            low_clip = masked_batch_mean(is_low_clipped.float())
            high_clip = masked_batch_mean(is_high_clipped.float())
            clip_ratio = masked_batch_mean(is_region_clipped.float())

            gathered_low_clip = self.accelerator.gather(low_clip)
            self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
            gathered_high_clip = self.accelerator.gather(high_clip)
            self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
            self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
            gathered_clip_ratio = self.accelerator.gather(clip_ratio)
            self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        elif self.loss_type == "cispo":
            is_cispo_clipped = (coef_1 > self.epsilon_high) & (advantages > 0)
            cispo_clip_ratio = masked_batch_mean(is_cispo_clipped.float())
            gathered_cispo_clip_ratio = self.accelerator.gather(cispo_clip_ratio)
            self._metrics[mode]["cispo_clip_ratio"].append(gathered_cispo_clip_ratio.nanmean().item())

        return loss

    # TRL
    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Returns a binary mask identifying tokens whose entropy exceeds a given quantile threshold.

        Args:
            entropies (`torch.Tensor`):
                Tensor of shape (batch_size, seq_len) with per-token entropy values.
            mask (`torch.Tensor`):
                Binary mask of the same shape as `entropies`, where `1` indicates valid tokens and `0` padding.
            threshold (`float`):
                Quantile threshold between `0.0` and `1.0` to select high-entropy tokens.

        Returns:
            `torch.Tensor`:
                Boolean mask of shape (batch_size, seq_len), where `True` indicates tokens with entropy >= threshold
                and `False` otherwise.
        """
        local = entropies[mask.bool()].float()

        # Use a negative pad_value as a sentinel because entropy values are always >= 0.
        # This guarantees that the sentinel cannot collide with any real entropy value.
        pad_value = -1e9

        # Pad across processes so that every rank has the same tensor length
        padded = self.accelerator.pad_across_processes(local, dim=0, pad_index=pad_value)
        gathered = self.accelerator.gather(padded)

        # Drop sentinel values (safe because no entropy can be negative)
        gathered = gathered[gathered != pad_value]

        if gathered.numel() == 0:
            return torch.zeros_like(entropies, dtype=torch.bool)

        entropy_threshold = torch.quantile(gathered, threshold)
        masked_entropies = entropies * mask.float()
        entropy_mask = masked_entropies >= entropy_threshold
        return entropy_mask & mask.bool()  # ensure padding tokens are always masked out

    # TRL
    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            
            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    # TRL
    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep
        )

        loss_mask = completion_mask
        # Compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            # The attention_mask parameter in liger loss is actually used as a loss mask (not model attention)
            attention_mask=loss_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
            vllm_is_ratio=inputs.get("importance_sampling_ratio"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        normalizer = self.current_gradient_accumulation_steps if mode == "train" else 1.0  # no accum in eval
        return loss / normalizer

    # TRL
    @staticmethod
    def get_off_policy_mask(
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
        sampling_per_token_logps: torch.Tensor,
        mask: torch.Tensor,
        off_policy_threshold: float,
    ) -> torch.Tensor:
        """
        Computes the Off-Policy Sequence Mask from DeepSeek-V3.2 paper. Returns a (B, 1) tensor where 1.0 indicates
        "Keep" and 0.0 indicates "Drop".
        """
        # forward KL div: log(pi_old) - log(pi_theta)
        kl_div = sampling_per_token_logps - per_token_logps.detach()
        # Sequence-level Mean KL (ignoring prompt+padding)
        seq_kl_sum = (kl_div * mask).sum(dim=1, keepdim=True)
        avg_seq_kl = seq_kl_sum / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        # Keep if (Advantage >= 0) OR (KL <= delta)
        is_pos_adv = advantages >= 0
        is_low_kl = avg_seq_kl <= off_policy_threshold
        return (is_pos_adv | is_low_kl).to(dtype=mask.dtype)  # (B, 1)

    # TRL
    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state

    @profiling_decorator
    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Repeat all input columns (but "prompt", "completion", and "completion_ids") to match the num of generations
        keys = [key for key in inputs[0] if key not in ["prompt", "completion", "completion_ids"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}

        # This allows for dynamic reward shaping based on training progress.
        reward_kwargs["trainer_state"] = self.state

        # Allow reward functions to log extra columns to the completions table.
        reward_kwargs["log_extra"] = self._log_completion_extra

        # Allow reward functions to log additional scalar metrics.
        reward_kwargs["log_metric"] = self._log_metric

        async_funcs_info = []  # async custom functions for asyncio.gather

        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names, strict=True)
        ):
            if isinstance(reward_func, nn.Module):  # Module (no PretrainedModel) for compat with compiled models
                with profiling_context(self, reward_func_name):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions, strict=True)]
                        texts = [
                            apply_chat_template(x, reward_processing_class, **self.chat_template_kwargs)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions, strict=True)]
                    reward_inputs = reward_processing_class(
                        text=texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                    )
                    reward_inputs = super()._prepare_inputs(reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            elif inspect.iscoroutinefunction(reward_func):  # Separate async reward funcs to run them in parallel later
                async_funcs_info.append((i, reward_func, reward_func_name))
            else:
                # Run synchronous reward function
                eos_and_pad = [self._tokenizer.eos_token_id, self._tokenizer.pad_token_id]
                is_truncated = [completion_ids[-1] not in eos_and_pad for completion_ids in completion_ids_list]
                with profiling_context(self, reward_func_name):
                    output_reward_func = reward_func(
                        prompts=prompts,
                        completions=completions,
                        is_truncated=is_truncated,
                        processing_class=self.processing_class,
                        prepare_inputs_func=super()._prepare_inputs,
                        model=self.model,
                        **reward_kwargs
                    )
                    if isinstance(output_reward_func, dict):
                        rewards_list = output_reward_func["rewards"]
                    else:
                        rewards_list = output_reward_func
                    # Convert None values to NaN
                    rewards_list = [reward if reward is not None else torch.nan for reward in rewards_list]
                    rewards_per_func[:, i] = torch.tensor(rewards_list, dtype=torch.float32, device=device)
        # Execute async custom functions in parallel using asyncio.gather
        if async_funcs_info:

            async def _invoke_async(index, func, func_name):
                with profiling_context(self, func_name):
                    output = await func(
                        prompts=prompts, completions=completions, completion_ids=completion_ids_list, **reward_kwargs
                    )
                    output = [r if r is not None else torch.nan for r in output]
                    return index, output

            async def _run_async_funcs():
                coros = [_invoke_async(i, func, func_name) for (i, func, func_name) in async_funcs_info]
                return await asyncio.gather(*coros)

            async_results = asyncio.run_coroutine_threadsafe(_run_async_funcs(), self.async_loop).result()
            for idx, output_reward_func in async_results:
                rewards_per_func[:, idx] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {
                key: value[nan_row_idx]
                for key, value in reward_kwargs.items()
                if key not in ("trainer_state", "log_extra", "log_metric")
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            logger.warning(
                f"All reward functions returned None for the following kwargs:"#\n{row_reward_kwargs}\n"
                "Please ensure that at least one reward function returns a valid reward."
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    #####################
    ####### TRAIN #######
    #####################
    # TRL
    def training_step(self, model, inputs, num_items_in_batch):
        time_before = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step += 1
        time_after = time.perf_counter()
        self._current_train_step_time += time_after - time_before
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(self._current_train_step_time)
            self._current_train_step_time = 0.0
        return output

    # TRL
    # Ensure the model card is saved along with the checkpoint
    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        super()._save_checkpoint(model, trial)

    
    # TRL
    # This method overrides `Trainer.get_train_dataloader` to support our custom batching strategy.
    # Instead of returning a standard per-step batch (i.e., `per_device_batch_size), our dataloader loads an
    # *generation* batch (i.e., `per_device_batch_size × steps_per_generation`). This allows us to generate completions
    # once every steps_per_generation step—rather than once per accumulation step—which is significantly more
    # efficient. The only change from the original implementation is multiplying the batch size by
    # `steps_per_generation`. Thus, `_prepare_inputs` is called with this *generation* batch, and it handles the
    # splitting internally.
    # Maintenance note: This method is a copy-paste of the original `Trainer.get_train_dataloader` with only one line
    # modification.
    def get_train_dataloader(self):
        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size * self.args.steps_per_generation,  # < this is the change
            sampler_fn=self._get_train_sampler,
            is_training=True,
        )

    # TRL
    class RepeatSampler(Sampler):
        """
        Sampler that repeats the indices of a dataset in a structured manner.

        Args:
            data_source (`Sized`):
                Dataset to sample from.
            mini_repeat_count (`int`):
                Number of times to repeat each index per batch.
            batch_size (`int`, *optional*, defaults to `1`):
                Number of unique indices per batch.
            repeat_count (`int`, *optional*, defaults to `1`):
                Number of times to repeat the full sampling process.
            shuffle (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the dataset.
            seed (`int`, *optional*):
                Random seed for reproducibility (only affects this sampler).

        Example:
        ```python
        >>> sampler = RepeatSampler(["a", "b", "c", "d", "e", "f", "g"], mini_repeat_count=2, batch_size=3, repeat_count=4)
        >>> list(sampler)
        [4, 4, 3, 3, 0, 0,
        4, 4, 3, 3, 0, 0,
        4, 4, 3, 3, 0, 0,
        4, 4, 3, 3, 0, 0,
        1, 1, 2, 2, 6, 6,
        1, 1, 2, 2, 6, 6,
        1, 1, 2, 2, 6, 6,
        1, 1, 2, 2, 6, 6]
        ```

        ```txt
        mini_repeat_count = 3
            -   -   -
            [0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,     |
            4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
            8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,      |
                                                                    repeat_count = 2
            0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,      |
            4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,      |
            8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11, ...] |
            ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
            ---------   ---------   ---------   ---------
                            batch_size = 12
        ```
        """

        def __init__(
            self,
            data_source: Sized,
            mini_repeat_count: int,
            batch_size: int = 1,
            repeat_count: int = 1,
            shuffle: bool = True,
            seed: int | None = None,
        ):
            self.data_source = data_source
            self.mini_repeat_count = mini_repeat_count
            self.batch_size = batch_size
            self.repeat_count = repeat_count
            self.num_samples = len(data_source)
            self.shuffle = shuffle
            self.seed = seed
            if shuffle:
                self.generator = torch.Generator()  # Create a local random generator
                if seed is not None:
                    self.generator.manual_seed(seed)

        def __iter__(self):
            if self.shuffle:
                # E.g., [2, 4, 3, 1, 0, 6, 5] (num_samples = 7)
                indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
            else:
                indexes = list(range(self.num_samples))

            #    [2, 4, 3, 1, 0, 6, 5]
            # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
            indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]

            #    [[2, 4, 3], [1, 0, 6], [5]]
            # -> [[2, 4, 3], [1, 0, 6]]
            indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

            for chunk in indexes:
                for _ in range(self.repeat_count):
                    for index in chunk:
                        for _ in range(self.mini_repeat_count):
                            yield index

        def __len__(self) -> int:
            return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count
    
    # TRL
    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        # Returns a sampler that
        # 1. ensures each prompt is repeated across multiple processes. This guarantees that identical prompts are
        #    distributed to different GPUs, allowing rewards to be computed and normalized correctly within each prompt
        #    group. Using the same seed across processes ensures consistent prompt assignment, preventing discrepancies
        #    in group formation.
        # 2. repeats the batch multiple times to allow reusing generations across multiple updates. Refer to
        #    _prepare_inputs to see how the generations are stored and reused.

        # In the following figure, the values are the prompt indices. The first row shows the first sampled batch, the
        # second row shows the second sampled batch, and so on.
        #
        #                                      |   GPU 0  |   GPU 1  |
        #
        #                 global_step   step    <-───>  num_generations=2
        #                                       <-───────> per_device_train_batch_size=3
        #  grad_accum    ▲  ▲  0          0     0   0   1   1   2   2   <- Generate for the first `steps_per_generation` (prompts 0 to 11); store the completions; use the first slice to compute the loss
        #     =2         ▼  |  0          1     3   3   4   4   5   5   <- Take the stored generations and use the second slice to compute the loss
        #                   |
        #                   |  1          2     6   6   7   7   8   8   <- Take the stored generations and use the third slice to compute the loss
        #  steps_per_gen=4  ▼  1          3     9   9  10  10  11  11   <- Take the stored generations and use the fourth slice to compute the loss
        #
        #                      2          4    12  12  13  13  14  14   <- Generate for the second `steps_per_generation` (prompts 12 to 23); store the completions; use the first slice to compute the loss
        #                      2          5    15  15  16  16  17  17   <- Take the stored generations and use the second slice to compute the loss
        #                                          ...
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,                             # how many times for each unique prompt to generate
            batch_size=self.args.generation_batch_size // self.num_generations, # how many unique prompt
            repeat_count=self.num_iterations * self.args.steps_per_generation,  # how many times the total generated results will be used
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    ####################
    ####### EVAL #######
    ####################
    # TRL
    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # See _get_train_sampler for an explanation of the sampler.
        return RepeatSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations_eval,
            shuffle=False,
            seed=self.args.seed,
        )

    # TRL
    def _log_completion_extra(self, column: str, values: list):
        """
        Log extra columns to the completions table. Called from reward functions via the `log_extra` kwarg.

        Args:
            column (`str`):
                Name of the column to add.
            values (`list`):
                Values for the column, one per sample in the batch.
        """
        self._pending_extra_logs[column].extend(values)
    
    # TRL
    def _log_metric(self, name: str, value: float):
        """
        Log a scalar metric from a reward function. Called via the `log_metric` kwarg. Values are averaged over each
        logging step and reported alongside built-in metrics like `kl` and `entropy`.

        Args:
            name (`str`):
                Name of the metric.
            value (`float`):
                Scalar value for this batch.
        """
        self._pending_metrics[name].append(value)
    
    # TRL
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {}
        for key, val in self._metrics[mode].items():
            # Filter out NaN values before averaging. A reward function that returns None for all samples
            # in a batch produces NaN for that batch's metric. With logging_steps > 1, a naive sum()/len()
            # would let a single NaN contaminate valid data from other batches. Only return None when no
            # valid values remain (e.g. JSON loggers crash on float NaN).
            valid = [v for v in val if not math.isnan(v)]
            metrics[key] = sum(valid) / len(valid) if valid else None

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if is_rich_available():
                print_prompt_completions_sample(
                    self._logs["prompt"],
                    self._logs["completion"],
                    self._logs["rewards"],
                    self._logs["advantages"],
                    self.state.global_step,
                    self.num_completions_to_print,
                    extra=dict(self._logs["extra"]),
                )

            logging_backends = []
            if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:
                logging_backends.append(wandb)
            if self.args.report_to and "trackio" in self.args.report_to:
                logging_backends.append(trackio)

            table = {
                "step": [self.state.global_step] * len(self._logs["prompt"]),
                "prompt": self._logs["prompt"],
                "completion": self._logs["completion"],
                **self._logs["rewards"],
                **self._logs["extra"],
                "advantage": self._logs["advantages"],
            }

            df_base = pd.DataFrame(table)
            df_base.to_parquet(
                os.path.join(
                    self.args.output_dir,
                    "completions",
                    f"completions_{self.state.global_step:05d}.parquet",
                )
            )

            for logging_backend in logging_backends:
                df = df_base

                if self.log_unique_prompts:
                    df = df.drop_duplicates(subset=["prompt"])

                logging_backend.log({"completions": logging_backend.Table(dataframe=df)})

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled or (self.is_fsdp_enabled and not self.args.torch_compile)
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)
        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        metrics = None
        eval_set_kwargs = {}

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = (
                self._prepare_input(inputs[main_input_name]) if "inputs" in args.include_for_metrics else None
            )

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if losses is not None:
                losses = self.gather_function(losses.repeat(batch_size))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function(inputs_decode)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                # Pad labels here, preparing for preprocess_logits_for_metrics in next logits block.
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function(logits)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_preds.add(logits)
            if labels is not None:
                labels = self.gather_function(labels)
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            if self.args.batch_eval_metrics:
                if self.compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    batch_kwargs = {}
                    batch_kwargs["losses"] = losses if "loss" in args.include_for_metrics else None
                    batch_kwargs["inputs"] = inputs if "inputs" in args.include_for_metrics else None
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=logits, label_ids=labels, **batch_kwargs),
                        compute_result=is_last_step,
                    )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
            and not self.args.batch_eval_metrics
        ):
            eval_set_kwargs["losses"] = all_losses if "loss" in args.include_for_metrics else None
            eval_set_kwargs["inputs"] = all_inputs if "inputs" in args.include_for_metrics else None
            metrics = self.compute_metrics(
                EvalPrediction(predictions=all_preds, label_ids=all_labels, **eval_set_kwargs),
                self.eval_dataset,
                self.processing_class
            )
        elif metrics is None:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "model_preparation_time"):
            metrics[f"{metric_key_prefix}_model_preparation_time"] = self.model_preparation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    # During eval, Trainer calls prediction_step. If no labels are present in the inputs, it only runs forward and
    # returns logits. We override prediction_step to force compute_loss, because this trainer doesn't involve labels.
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        labels = self.processing_class(
            text=[data['answer'] for data in inputs], padding=True, padding_side="left", return_tensors="pt"
        )

        inputs = self._prepare_inputs(inputs)
        generated_tokens = inputs['completion_ids'].clone().detach()

        if self.only_generate:
            return torch.tensor(0.0).to(model.device), generated_tokens, labels['input_ids'].to(model.device)
        else:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
            return loss, generated_tokens, labels['input_ids'].to(model.device)