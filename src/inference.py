import torch
import argparse
import os
from data_utils import load_vocab
from model import Encoder, Decoder, Attention, Seq2Seq
from translator import Translator
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = f"{ROOT_DIR}/datasets/IWSLT'15 en-vi"
CHECKPOINTS_DIR = f"{ROOT_DIR}/checkpoints" 


def main():
    parser = argparse.ArgumentParser(description='Translate English to Vietnamese')
    parser.add_argument('--text', type=str, help='Input English sentence')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--dataset_dir', type=str, default=DATASETS_DIR, help='Path to dataset (for vocab)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--emb_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--save_attention', type=str, default='attention.png', help='Path to save attention map')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    en_vocab_path = os.path.join(args.dataset_dir, "vocab.en.txt")
    vi_vocab_path = os.path.join(args.dataset_dir, "vocab.vi.txt")
    
    en_vocab = load_vocab(en_vocab_path)
    vi_vocab = load_vocab(vi_vocab_path)
    
    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(vi_vocab)

    # -------
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
        print(f"Checkpoint {CKP_PATH} not found! Using initialized weights (random output).")

    # ----
    translator = Translator(model, en_vocab, vi_vocab, device)
    
    text = args.text
    if not text:
        text = input("Enter English sentence: ")
        
    translation, attention, src_tokens = translator.translate_sentence(text)
    
    print(f"\nInput: {text}")
    print(f"Translation: {translation}")
    
    # 4. Plot Attention
    translator.display_attention(src_tokens, translation, attention, save_path=args.save_attention)



if __name__ == "__main__":
    main()
