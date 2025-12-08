# Seq2Seq English-Vietnamese Translation

This project implements a Sequence-to-Sequence (Seq2Seq) model with Attention mechanism for translating English to Vietnamese using the IWSLT'15 dataset.

## Project Structure

- `src/`: Source code
    - `data_utils.py`: Data loading, cleaning, and processing.
    - `model.py`: Seq2Seq model architecture (Encoder, Decoder, Attention).
    - `trainer.py`: Trainer class with logging and checkpointing.
    - `train.py`: Training entry point.
    - `translator.py`: Translation logic.
    - `inference.py`: Inference entry point.
    - `test.py`: Test model entry point.
    - `visualize_attn_map .py`: Result viewer tool.
- `datasets/`: Dataset directory (IWSLT'15 en-vi).

## Setup

1. **Prerequisite:** Ensure [uv](https://github.com/astral-sh/uv) is installed.

2. Clone the repository:
   ```bash
   git clone https://github.com/peterhub2003/Seq2Seq-English-Vietnamese-Translation.git
   cd Seq2Seq-English-Vietnamese-Translation/
   ```

3. Setup Python environment with uv:
   ```bash
   uv python pin 3.11
   uv sync
   ```

## Training

Run the training script:

```bash
uv run python -m src.train --epochs 20 --batch_size 64 --hidden_dim 512 --emb_dim 256 --bidirectional --checkpoint best_bi_model.pt
```

Arguments:
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension (default: 512)
- `--emb_dim`: Embedding dimension (default: 256)
- `--bidirectional`: Use Bidirectional Encoder (Recommended)
- `--use_wandb`: Enable Weights & Biases logging. (Default using Console Logger)
- `--checkpoint`: File to save the best model (default: `best_model.pt` if not using bidirectional).

## Inference

Run the inference script to translate a sentence. You can choose between **Greedy Decoding** (default) or **Beam Search**.

```bash
# Greedy Decoding
uv run python -m src.inference --text "Hello world" --checkpoint best_bi_model.pt --bidirectional

# Beam Search
uv run python -m src.inference --text "Hello world" --checkpoint best_bi_model.pt --bidirectional --beam_size 3
```

Arguments:
- `--beam_size`: Size of beam. Set to 1 for Greedy. (Default: 1)
- `--length_penalty_alpha`: Alpha for length penalty in Beam Search (Default: 0.7)
- `--no_repeat_ngram_size`: Size of N-grams to prevent repetition. Set to 2 or 3 to fix word repetitions. (Default: 0 - Disabled)
- `--bidirectional`: Must match the training configuration.

It will output the translation and save the attention map to `attention.png`.

## Testing

Evaluate the trained model on the test set (`tst2013`):

```bash
uv run python -m src.test --checkpoint best_bi_model.pt --output_file test_results.json --bidirectional --beam_size 3 --no_repeat_ngram_size 2
```

This will calculate the BLEU score and save the results (Source, Reference, Prediction) to `test_results.json` in folder `results`.

## Results Visualization

Inspect the test results and visualize Attention Maps using the viewer tool:

```bash
uv run python -m src.visualize_attn_map --results_file test_results.json --bidirectional --checkpoint best_bi_model.pt --beam_size 3 --no_repeat_ngram_size 2
```

-  Select a sample index to re-run translation and generate/save the Attention Map.


## Data Processing Details


- **Tokenization:** Space-based (pre-tokenized data).
- **Vocab:** Provided vocab files are used. Smart casing lookup is implemented.
- **Special Tokens:** `<unk>`, `<pad>`, `<s>`, `</s>` are handled.

## Model Architecture

- **Encoder:** GRU with Embedding and Dropout. Optionally **Bidirectional**.
- **Decoder:** GRU with Attention (Bahdanau-style), Embedding, and Dropout.
- **Attention:** Additive Attention.
