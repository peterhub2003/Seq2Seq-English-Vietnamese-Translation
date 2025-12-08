import json
import argparse
import os
import torch
from pathlib import Path
from .data_utils import load_vocab
from .model import Encoder, Decoder, Attention, Seq2Seq
from .translator import Translator
from .config import CHECKPOINTS_DIR, DATASETS_DIR, RESULTS_DIR



class ResultViewer:
    def __init__(self, results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        print(f"Loaded {len(self.results)} results from {results_file}")

    def list_samples(self, limit=10, start=0):
        print(f"\n--- Samples {start} to {start+limit} ---")
        for i in range(start, min(start+limit, len(self.results))):
            item = self.results[i]
            
            print(f"[{i}] Src: {item['src']}")
            print(f"    Ref: {item['ref']}")
            print(f"    Pred: {item['pred']}")
            print("-" * 20)

    def get_sample(self, index):
        if 0 <= index < len(self.results):
            return self.results[index]
        return None

def view_attention(index, args, viewer):
    sample = viewer.get_sample(index)
    if not sample:
        print("Invalid index.")
        return

    print(f"\nGenerating Attention Map for Sample {index}...")
    print(f"Src: {sample['src']}")
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    en_vocab_path = os.path.join(args.dataset_dir, "vocab.en.txt")
    vi_vocab_path = os.path.join(args.dataset_dir, "vocab.vi.txt")
    en_vocab = load_vocab(en_vocab_path)
    vi_vocab = load_vocab(vi_vocab_path)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(vi_vocab)
    
    # Load Model
    ENC_HID_DIM = args.hidden_dim
    DEC_HID_DIM = args.hidden_dim
    ENC_OUTPUT_DIM = args.hidden_dim * 2 if args.bidirectional else args.hidden_dim
    
    attn = Attention(ENC_OUTPUT_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, args.emb_dim, ENC_HID_DIM, args.n_layers, args.dropout, args.bidirectional)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, ENC_OUTPUT_DIM, DEC_HID_DIM, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    CKP_PATH = os.path.join(CHECKPOINTS_DIR, args.checkpoint)
    if os.path.exists(CKP_PATH):
        checkpoint = torch.load(CKP_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Checkpoint not found.")
        return


    translator = Translator(model, en_vocab, vi_vocab, device)
    translation, attention, src_tokens = translator.translate_sentence(
        sample['src'], 
        beam_size=args.beam_size, 
        length_penalty_alpha=args.length_penalty_alpha, 
        no_repeat_ngram_size=args.no_repeat_ngram_size
    )
    
    save_path = f"attention_sample_{index}.png"
    translator.display_attention(src_tokens, translation, attention, save_path=save_path)
    print(f"Attention map saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='View Test Results')
    parser.add_argument('--results_file', type=str, default='test_results.json', help='Path to results file')
    parser.add_argument('--dataset_dir', type=str, default=DATASETS_DIR, help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Path to save/load checkpoint')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional encoder')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size (1 for greedy)')
    parser.add_argument('--length_penalty_alpha', type=float, default=0.7, help='Length penalty alpha')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='Size of N-grams to prevent repetition (0 to disable)')

    args = parser.parse_args()
    
    RESULTS_PATH = os.path.join(RESULTS_DIR, args.results_file)
    if not os.path.exists(RESULTS_PATH):
        print(f"Results file {args.results_file} not found. Run src/test.py first.")
        return

    viewer = ResultViewer(RESULTS_PATH)
    
    while True:
        print("\nOptions:")
        print("1. View specific sample & attention")
        print("2. Exit")
        choice = input("Enter choice: ")
        
        if choice == '1':
            idx = int(input("Enter sample index: "))
            view_attention(idx, args, viewer)
        elif choice == '2':
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main()
