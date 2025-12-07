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
python src/train.py --epochs 10 --batch_size 32 --use_wandb
```

Arguments:
- `--epochs`: Number of epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--use_wandb`: Enable Weights & Biases logging.
- `--checkpoint_path`: Path to save the best model (default: `best_model.pt`).

## Inference

Run the inference script to translate a sentence:

```bash
python src/inference.py --text "Hello world" --checkpoint best_model.pt
```

It will output the translation and save the attention map to `attention.png`.

## Testing

Evaluate the trained model on the test set (`tst2013`):

```bash
python src/test_model.py --checkpoint best_model.pt --output_file test_results.json
```

This will calculate the BLEU score and save the results (Source, Reference, Prediction) to `test_results.json`.

## Results Visualization

Inspect the test results and visualize Attention Maps using the viewer tool:

```bash
python src/result_viewer.py --results_file test_results.json
```

- **Option 1:** List samples (Source, Reference, Prediction).
- **Option 2:** Select a sample index to re-run translation and generate/save the Attention Map.

## Data Processing Details

- **HTML Entities:** Decoded (e.g., `&apos;` -> `'`).
- **Tokenization:** Space-based (pre-tokenized data).
- **Vocab:** Provided vocab files are used. Smart casing lookup is implemented.
- **Special Tokens:** `<unk>`, `<pad>`, `<s>`, `</s>` are handled.

## Model Architecture

- **Encoder:** GRU with Embedding and Dropout.
- **Decoder:** GRU with Attention (Bahdanau-style), Embedding, and Dropout.
- **Attention:** Additive Attention.
