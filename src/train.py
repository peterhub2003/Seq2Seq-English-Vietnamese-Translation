import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from .data_utils import build_dataloaders
from .model import Encoder, Decoder, Attention, Seq2Seq
from .trainer import Trainer, WandbLogger, ConsoleLogger, LocalCheckpointManager
from .config import DATASETS_DIR, CHECKPOINTS_DIR



def main():

    parser = argparse.ArgumentParser(description='Train Seq2Seq Model')
    parser.add_argument('--dataset_dir', type=str, default=DATASETS_DIR, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--emb_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--teacher_forcing', type=float, default=0.5, help='Teacher forcing ratio')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--project_name', type=str, default='seq2seq-en-vi', help='WandB project name')
    parser.add_argument('--checkpoint_path', type=str, default='best_model.pt', help='Path to save/load checkpoint')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load from checkpoint if exists')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional encoder')


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

    train_loader, val_loader, test_loader, en_vocab, vi_vocab = build_dataloaders(config)

    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(vi_vocab)
    
    print(f"Input Vocab Size: {INPUT_DIM}")
    print(f"Output Vocab Size: {OUTPUT_DIM}")


    # 2. Model
    ENC_HID_DIM = args.hidden_dim
    DEC_HID_DIM = args.hidden_dim
    ENC_OUTPUT_DIM = args.hidden_dim * 2 if args.bidirectional else args.hidden_dim
    
    attn = Attention(ENC_OUTPUT_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, args.emb_dim, ENC_HID_DIM, args.n_layers, args.dropout, args.bidirectional)
    dec = Decoder(OUTPUT_DIM, args.emb_dim, ENC_OUTPUT_DIM, DEC_HID_DIM, args.n_layers, args.dropout, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    model.init_weights(mean=0, std=0.01)    
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')


    # 3. Optimizer & Criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Ignore padding index in loss calculation
    pad_idx = vi_vocab.lookup_token('<pad>')
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 4. Logger & Checkpoint Manager
    if args.use_wandb:
        logger = WandbLogger(args.project_name, vars(args))
    else:
        logger = ConsoleLogger()

    checkpoint_manager = LocalCheckpointManager()
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    CKP_PATH = os.path.join(CHECKPOINTS_DIR, args.checkpoint_path)
    if args.load_checkpoint and os.path.exists(CKP_PATH):
        checkpoint_manager.load(CKP_PATH, model, optimizer)
        print("Resuming training from checkpoint...")


    # 5. Trainer
    trainer_config = {
        'clip': args.clip,
        'teacher_forcing_ratio': args.teacher_forcing,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'hidden_dim': args.hidden_dim,
        'emb_dim': args.emb_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'max_length': 100,
        'bidirectional': args.bidirectional
    }
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
        config=trainer_config
    )

    # 6. Start Training
    try:
        trainer.show_config()
        trainer.train(args.epochs, save_path=CKP_PATH, en_vocab=en_vocab, vi_vocab=vi_vocab)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if isinstance(logger, ConsoleLogger):
            logger.plot_history()

if __name__ == "__main__":
    main()

    