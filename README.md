# :bulb: GeneralThinker

Official code for "**GeneralThinker: Domain-General Reasoning through Answer-Conditioned Likelihood Optimization**"

## :bookmark_tabs: Table of Contents

- :hammer_and_wrench: [Getting Started](#hammer_and_wrench-getting-started)
- :rocket: [Running GeneralThinker](#rocket-running-generalthinker)
  - :memo: [Data](#memo-data)
  - :dart: [Train GeneralThinker](#dart-train-generalthinker)
  - :hourglass_flowing_sand: [Inference](#hourglass_flowing_sand-inference)
- :page_facing_up: [License](#page_facing_up-license)

## :hammer_and_wrench: Getting Started

Clone the repository with:

```bash
git clone https://github.com/shengminp/GeneralThinker.git
cd GeneralThinker

conda env create -f environment.yml
conda activate grm

python -m pip install uv==0.11.7
uv pip install -r requirements.txt
```

Environment setup instructions will be added later.

The project directory is organized as follows:

```text
.
├── datasets
│   ├── general
│   │   ├── train.json
│   │   └── valid.json
│   └── ...
├── models                                      # Stores model checkpoints
├── results                                     # Saves generated results during inference
├── scripts
│   ├── finetune.py                             # Handles GeneralThinker training
│   └── generate.py                             # Handles generation and evaluation
├── src
│   ├── __init__.py
│   ├── config.py                               # Training and generation configurations
│   ├── data.py                                 # Data loading and preprocessing
│   ├── trainer.py                              # GeneralThinker trainer
│   └── utils.py                                # Utility functions and reward computation
└── README.md
```

## :rocket: Running GeneralThinker

### :memo: Data

The training data is included in this repository under:

```text
./datasets/general/
├── train.json
└── valid.json
```

Evaluation datasets should be placed under `./datasets/DATASET_NAME/` following the file names expected by `src/config.py`.

GeneralThinker currently supports the following backbone models:

- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen3-4B`

### :dart: Train GeneralThinker

GeneralThinker uses answer-conditioned likelihood as a response-level reward and supports token-level advantage modulation with direction-preserving control.

Run the following command:

```bash
python scripts/finetune.py \
    --base_model $model_name \
    --data_name general \
    --reward_type likelihood \
    --modulation \
    --controlled_modulation \
    --modulation_lambda $modulation_lambda \
    --per_gpu_batch_size $batch_size \
    --num_train_epochs $num_epoch \
    --lora_r $lora_r \
    --learning_rate $lr \
    --grad_accum $grad_accum
```

- **`$model_name`**: Backbone model name from Hugging Face:
  - `Qwen/Qwen2.5-3B-Instruct`
  - `meta-llama/Llama-3.2-3B-Instruct`
  - `Qwen/Qwen3-4B`
- **`$modulation_lambda`**: Strength of token-level advantage modulation.
- **`$batch_size`**: Batch size per GPU.
- **`$num_epoch`**: Number of training epochs.
- **`$lora_r`**: LoRA rank.
- **`$lr`**: Learning rate.
- **`$grad_accum`**: Number of gradient accumulation steps.

To disable token-level modulation and train with only the response-level likelihood reward, omit:

```text
--modulation
--controlled_modulation
--modulation_lambda $modulation_lambda
```

For token-signal perturbation analysis, `--perturbation` supports:

```text
shuffle
reverse
```

The `--top_entropy_quantile` argument can be used to restrict modulation to high-entropy tokens.

### :hourglass_flowing_sand: Inference

Use the following command to generate predictions from a trained GeneralThinker checkpoint:

```bash
python scripts/generate.py \
    --base_model $model_name \
    --checkpoint_name $checkpoint_name \
    --data_name $data_name \
    --reward_type likelihood \
    --modulation \
    --controlled_modulation \
    --modulation_lambda $modulation_lambda \
    --per_gpu_batch_size $batch_size \
    --with_prompt false \
    --generation_type greedy \
    --generation_times 1
```

- **`$model_name`**: Backbone model name.
- **`$checkpoint_name`**: Name of the trained checkpoint to evaluate.
- **`$data_name`**: Evaluation dataset name.
- **`$modulation_lambda`**: Modulation strength used during training.
- **`$batch_size`**: Batch size per GPU.
- **`--with_prompt`**: Whether to prepend demonstrations from `prompt.json` (`true` or `false`).
- **`--generation_type`**: Generation strategy (`greedy` or `sample`).
- **`--generation_times`**: Number of generation rounds.

You can also evaluate an explicitly specified checkpoint path:

```bash
python scripts/generate.py \
    --base_model $model_name \
    --checkpoint_path $checkpoint_path \
    --data_name $data_name \
    --per_gpu_batch_size $batch_size \
    --with_prompt false \
    --generation_type greedy \
    --generation_times 1
```

Generation outputs are saved under `./results/`.

## :page_facing_up: License

This project is licensed under the [MIT](LICENSE) © Shengmin Piao
