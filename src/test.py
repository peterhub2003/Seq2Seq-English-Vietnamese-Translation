import torch
import argparse
import os
from pathlib import Path
from .data_utils import build_dataloaders
from .model import Encoder, Decoder, Attention, Seq2Seq
from .evaluator import Evaluator, ResultSaver
from .config import DATASETS_DIR, CHECKPOINTS_DIR, RESULTS_DIR



def main():
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq Model on Test Set')
    parser.add_argument('--dataset_dir', type=str, default=DATASETS_DIR, help='Path to dataset (for vocab)')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--output_file', type=str, default='test_results.json', help='File to save results')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (keep 1 for simple loop)')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size (1 for greedy)')
    parser.add_argument('--length_penalty_alpha', type=float, default=0.7, help='Length penalty alpha')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Data
    config = {
        'dataset_dir': args.dataset_dir,
        'batch_size': args.batch_size,
        'max_length': 100,
        'num_workers': 0
    }
    # We only need test_loader and vocabs
    _, _, test_loader, en_vocab, vi_vocab = build_dataloaders(config)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(vi_vocab)
    
    # 2. Model
    attn = Attention(args.hidden_dim)
    enc = Encoder(INPUT_DIM, args.emb_dim, args.hidden_dim, args.n_layers, args.dropout)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, args.hidden_dim, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    CKP_PATH = os.path.join(CHECKPOINTS_DIR, args.checkpoint)
    if os.path.exists(CKP_PATH):
        print(f"Loading checkpoint from {CKP_PATH}...")
        checkpoint = torch.load(CKP_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"Checkpoint {CKP_PATH} not found! Please train the model first.")
        return

    # 3. Evaluate
    os.makedirs(RESULTS_DIR, exist_ok=True)
    RESULTS_PATH = os.path.join(RESULTS_DIR, args.output_file)
    saver = ResultSaver(RESULTS_PATH)
    evaluator = Evaluator(model, en_vocab, vi_vocab, device)
    
    bleu_score = evaluator.evaluate_batch(test_loader, result_saver=saver, beam_size=args.beam_size, length_penalty_alpha=args.length_penalty_alpha)
    
    print(f"\nTest Set BLEU Score: {bleu_score*100:.2f}")
    
    saver.save()

if __name__ == "__main__":
    main()
