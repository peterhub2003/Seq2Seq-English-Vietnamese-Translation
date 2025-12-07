import torch
import torch.nn as nn
import time
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from tqdm import tqdm
import os


# --- Interfaces (Abstractions) ---
class ILogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics, step): pass
    
    @abstractmethod
    def finish(self): pass

class ICheckpointManager(ABC):
    @abstractmethod
    def save(self, model, optimizer, epoch, metrics, path): pass

    @abstractmethod
    def load(self, path, model, optimizer): pass


# ---------     Implementations   ---------
class WandbLogger(ILogger):
    def __init__(self, project_name, config, name=None):
        try:
            import wandb
            self.wandb = wandb
            self.wandb.init(project=project_name, config=config, name=name)
            self.available = True
        except ImportError:
            print("WandB not installed. Falling back to console logging.")
            self.available = False
    
    def log_metrics(self, metrics, step):
        if self.available:
            self.wandb.log(metrics, step=step)
        
    def finish(self):
        if self.available:
            self.wandb.finish()

class ConsoleLogger(ILogger):
    def __init__(self):
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_bleu': []}

    def log_metrics(self, metrics, step):
        # Store for plotting later
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)
        
    def finish(self):
        pass
    
    def plot_history(self, save_path='training_history.png'):
        plt.figure(figsize=(18, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Loss History')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_acc'], label='Train Acc')
        plt.plot(self.history['val_acc'], label='Val Acc')
        plt.title('Accuracy History')
        plt.legend()
        
        if 'val_bleu' in self.history and len(self.history['val_bleu']) > 0:
            plt.subplot(1, 3, 3)
            plt.plot(self.history['val_bleu'], label='Val BLEU')
            plt.title('BLEU Score History')
            plt.legend()
        
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")


class LocalCheckpointManager(ICheckpointManager):
    def save(self, model, optimizer, epoch, metrics, path):
        print(f"Saving checkpoint to {path}...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
    def load(self, path, model, optimizer=None):
        print(f"Loading checkpoint from {path}...")
        # weights_only=False needed for older pytorch/custom classes safety check
        checkpoint = torch.load(path, weights_only=False) 
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 criterion: nn.Module, 
                 train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, 
                 logger: ILogger, 
                 checkpoint_manager: ICheckpointManager, 
                 device: torch.device, 
                 config: dict):
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        self.config = config
        
        self.clip = config.get('clip', 1.0)
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.best_val_loss = float('inf')

    def show_config(self):
        print("\n" + "="*30)
        print("TRAINING CONFIGURATION")
        print("="*30)
        print(f"Device: {self.device}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        print(f"Criterion: {self.criterion.__class__.__name__}")
        print("-" * 30)
        for k, v in self.config.items():
            print(f"{k}: {v}")
        print("="*30 + "\n")

    def calculate_accuracy(self, output, target, ignore_index):
        # output = [batch size * trg len - 1, output dim]
        # target = [batch size * trg len - 1]
        
        predictions = output.argmax(dim=1, keepdim=True).squeeze()
        # Filter out ignore_index (padding)
        mask = target != ignore_index
        correct = (predictions[mask] == target[mask]).sum().item()
        total = mask.sum().item()
        
        if total == 0:
            return 0
        return correct / total


    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for i, (src, trg) in enumerate(pbar):
            src = src.to(self.device)
            trg = trg.to(self.device)
            
            self.optimizer.zero_grad()
            
            # trg = [batch size, trg len]
            # output = [batch size, trg len, output dim]
            output = self.model(src, trg, self.teacher_forcing_ratio)
            
            # trg = [batch size, trg len]
            # output = [batch size, trg len, output dim]
            
            output_dim = output.shape[-1]
            
            # Discard the first token of output (which corresponds to <sos> input) 
            # and first token of trg (which is <sos>)
            # Actually, model output at t=0 corresponds to prediction for trg[1]
            # Let's align carefully.
            # Model forward loop:
            #   input = trg[:, 0] (<sos>) -> output[:, 1] (prediction for trg[:, 1])
            #   So output[:, 1:] aligns with trg[:, 1:]
            #   output[:, 0] is 0s (initialized)
            
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            
            loss = self.criterion(output, trg)
            acc = self.calculate_accuracy(output, trg, self.criterion.ignore_index)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc
            
            pbar.set_postfix({'loss': loss.item(), 'acc': acc})
            
        return epoch_loss / len(self.train_loader), epoch_acc / len(self.train_loader)

    def calculate_bleu(self, data_loader, en_vocab, vi_vocab):
        self.model.eval()
        targets = []
        outputs = []
        
        with torch.no_grad():
            for src, trg in data_loader:
                src = src.to(self.device)
                trg = trg.to(self.device)
                
                # Generate output
                # teacher_forcing_ratio = 0 to let model generate on its own
                output = self.model(src, trg, 0) 
                
                # output = [batch size, trg len, output dim]
                # Get predicted tokens
                preds = output.argmax(2) # [batch size, trg len]
                
                for i in range(preds.shape[0]):
                    # Convert indices to words
                    # Remove <sos> (if present in output logic, but output usually starts from trg[1])
                    # Our model output aligns with trg[1:]
                    
                    pred_tokens = []
                    for idx in preds[i]:
                        idx = idx.item()
                        if idx == vi_vocab.lookup_token('</s>'):
                            break
                        pred_tokens.append(vi_vocab.lookup_index(idx))
                    
                    outputs.append(pred_tokens)
                    
                    # Target
                    trg_tokens = []
                    # trg[i] includes <s> at start, we should skip it
                    # And stop at </s>
                    for idx in trg[i]:
                        idx = idx.item()
                        if idx == vi_vocab.lookup_token('<s>'):
                            continue
                        if idx == vi_vocab.lookup_token('</s>'):
                            break
                        trg_tokens.append(vi_vocab.lookup_index(idx))
                    
                    targets.append([trg_tokens]) # BLEU expects list of references
        
        # Calculate BLEU
        try:
            from nltk.translate.bleu_score import corpus_bleu
            # NLTK expects references as list of list of strings
            # references: [[ref1a, ref1b], [ref2a]]
            # hypotheses: [hyp1, hyp2]
            return corpus_bleu(targets, outputs)
        except ImportError:
            print("NLTK not found. Please install it (pip install nltk) for BLEU score. Returning 0.")
            return 0.0

    def evaluate(self, loader, en_vocab=None, vi_vocab=None):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        
        with torch.no_grad():
            for i, (src, trg) in enumerate(loader):
                src = src.to(self.device)
                trg = trg.to(self.device)
                
                output = self.model(src, trg, 0) # 0 = no teacher forcing
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = self.criterion(output, trg)
                acc = self.calculate_accuracy(output, trg, self.criterion.ignore_index)
                
                epoch_loss += loss.item()
                epoch_acc += acc
        
        bleu = 0
        if en_vocab and vi_vocab:
            bleu = self.calculate_bleu(loader, en_vocab, vi_vocab)
            
        return epoch_loss / len(loader), epoch_acc / len(loader), bleu


    def train(self, num_epochs, save_path='best_model.pt', en_vocab=None, vi_vocab=None):
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_bleu = self.evaluate(self.val_loader, en_vocab, vi_vocab)
            
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            
            # Log metrics
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_bleu': val_bleu,
                'epoch': epoch
            }
            self.logger.log_metrics(metrics, step=epoch)
            
            print(f'Epoch: {epoch:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}% | Val. BLEU: {val_bleu*100:.2f}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.checkpoint_manager.save(self.model, self.optimizer, epoch, metrics, save_path)
                print(f"\tNew best model saved with Val Loss: {val_loss:.3f}")
        
        self.logger.finish()
        print("Training complete.")