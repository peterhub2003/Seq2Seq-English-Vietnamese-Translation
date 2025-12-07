# Seq2Seq English-Vietnamese Translation

This project implements a Sequence-to-Sequence (Seq2Seq) model with Attention mechanism for translating English to Vietnamese using the IWSLT'15 dataset.

## Project Structure

- `src/`: Source code
    - `data_utils.py`: Data loading, cleaning, and processing.
    - `model.py`: Seq2Seq model architecture (Encoder, Decoder, Attention).
    - `trainer.py`: Trainer class with logging and checkpointing.
    - `train.py`: Training entry point.
    - `translator.py`: Inference logic.
    - `inference.py`: Inference entry point.
    <!-- - `result_viewer.py`: Result viewer tool. -->
    - `test_model.py`: Test model entry point.
- `datasets/`: Dataset directory (IWSLT'15 en-vi).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Install NLTK data if needed for BLEU score (though script handles basic import).

## Training

Run the training script:

```bash
python src/train.py --epochs 20 --batch_size 64 --hidden_dim 512 --emb_dim 256 --bidirectional --use_wandb
```

Arguments:
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension (default: 256)
- `--emb_dim`: Embedding dimension (default: 128)
- `--bidirectional`: Use Bidirectional Encoder (Recommended)
- `--use_wandb`: Enable Weights & Biases logging.
- `--checkpoint_path`: Path to save the best model (default: `checkpoints/best_model.pt`).

## Inference

Run the inference script to translate a sentence. You can choose between **Greedy Decoding** (default) or **Beam Search**.

```bash
# Greedy Decoding
python src/inference.py --text "Hello world" --checkpoint checkpoints/best_model.pt --bidirectional

# Beam Search (Better quality)
python src/inference.py --text "Hello world" --checkpoint checkpoints/best_model.pt --bidirectional --beam_size 3
```

Arguments:
- `--beam_size`: Size of beam. Set to 1 for Greedy. (Default: 1)
- `--length_penalty_alpha`: Alpha for length penalty in Beam Search (Default: 0.7)
- `--bidirectional`: Must match the training configuration.

It will output the translation and save the attention map to `attention.png`.

## Testing

Evaluate the trained model on the test set (`tst2013`):

```bash
python src/test_model.py --checkpoint checkpoints/best_model.pt --output_file test_results.json --bidirectional --beam_size 3
```

This will calculate the BLEU score and save the results (Source, Reference, Prediction) to `test_results.json`.

## Results Visualization

Inspect the test results and visualize Attention Maps using the viewer tool:

```bash
python src/result_viewer.py --results_file test_results.json --bidirectional
```

- **Option 1:** List samples (Source, Reference, Prediction).
- **Option 2:** Select a sample index to re-run translation and generate/save the Attention Map.

<!-- ## Model Sharing (Hugging Face Hub)

This project includes a built-in workflow to upload your trained model to Hugging Face Hub.

**Upload:**
```bash
python src/upload_to_hub.py --checkpoint checkpoints/best_model.pt --repo_id your-username/seq2seq-en-vi
```
(Token is loaded from `HF_TOKEN` in `.env` file or passed via `--token`)

**Workflow Documentation:**
For detailed steps on uploading and loading the model from the Hub, see: `.agent/workflows/huggingface_upload.md` -->

## Data Processing Details

- **HTML Entities:** Decoded (e.g., `&apos;` -> `'`).
- **Tokenization:** Space-based (pre-tokenized data).
- **Vocab:** Provided vocab files are used. Smart casing lookup is implemented.
- **Special Tokens:** `<unk>`, `<pad>`, `<s>`, `</s>` are handled.

## Model Architecture

- **Encoder:** GRU with Embedding and Dropout. Optionally **Bidirectional**.
- **Decoder:** GRU with Attention (Bahdanau-style), Embedding, and Dropout.
- **Attention:** Additive Attention.
