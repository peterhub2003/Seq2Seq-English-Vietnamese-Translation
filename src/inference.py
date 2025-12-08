import torch
import argparse
import os
from .data_utils import load_vocab
from .model import Encoder, Decoder, Attention, Seq2Seq
from .translator import Translator
from .config import DATASETS_DIR, CHECKPOINTS_DIR


def main():
    parser = argparse.ArgumentParser(description='Translate English to Vietnamese')
    parser.add_argument('--text', type=str, help='Input English sentence')
    parser.add_argument('--checkpoint', type=str, default='best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--dataset_dir', type=str, default=DATASETS_DIR, help='Path to dataset (for vocab)')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--save_attention', type=str, default='attention.png', help='Path to save attention map')
    parser.add_argument('--beam_size', type=int, default=1, help='Beam size (1 for greedy)')
    parser.add_argument('--length_penalty_alpha', type=float, default=0.7, help='Length penalty alpha')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional encoder')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0, help='Size of N-grams to prevent repetition (0 to disable)')

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
    ENC_HID_DIM = args.hidden_dim
    DEC_HID_DIM = args.hidden_dim
    ENC_OUTPUT_DIM = args.hidden_dim * 2 if args.bidirectional else args.hidden_dim
    
    attn = Attention(ENC_OUTPUT_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, args.emb_dim, ENC_HID_DIM, args.n_layers, args.dropout, args.bidirectional)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, ENC_OUTPUT_DIM, DEC_HID_DIM, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    

    # ---------
    CKP_PATH = os.path.join(CHECKPOINTS_DIR, args.checkpoint)
    if os.path.exists(CKP_PATH):
        print(f"Loading checkpoint from {CKP_PATH}...")
        checkpoint = torch.load(CKP_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.init_weights(mean=0, std=0.01)
        print(f"Checkpoint {CKP_PATH} not found! Using initialized weights (random output).")

    # ----
    translator = Translator(model, en_vocab, vi_vocab, device)
    
    text = args.text
    if not text:
        text = input("Enter English sentence: ")
        
    translation, attention, src_tokens = translator.translate_sentence(text, beam_size=args.beam_size, length_penalty_alpha=args.length_penalty_alpha, no_repeat_ngram_size=args.no_repeat_ngram_size)
    
    print(f"\nInput: {text}")
    print(f"Translation: {translation}")
    

    translator.display_attention(src_tokens, translation, attention, save_path=args.save_attention)



if __name__ == "__main__":
    main()
