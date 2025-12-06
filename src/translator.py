import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from .data_utils import process_line

class Translator:
    def __init__(self, model, en_vocab, vi_vocab, device):
        self.model = model
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.device = device
        self.model.eval()

    def translate_sentence(self, sentence, max_len=50):
        """
        Translates a single sentence.
        Returns:
            translation (str): Translated sentence.
            attention (tensor): Attention weights [trg_len, src_len].
        """
        # 1. Process Input
        # Clean and tokenize (using same logic as training)
        tokens = process_line(sentence)
        
        # Add <sos> (optional for encoder?) No, encoder usually just </s>
        # Add </s>
        tokens = tokens + ['</s>']
        
        # Convert to indices
        src_indexes = [self.en_vocab.lookup_token(token) for token in tokens]
        
        # Convert to tensor
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)
        # src_tensor = [1, src_len]
        
        # 2. Forward Encoder
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
        
        # 3. Loop Decoder
        # Initial input is <sos>
        trg_indexes = [self.vi_vocab.lookup_token('<s>')]
        
        # Store attention weights
        attentions = torch.zeros(max_len, 1, len(src_indexes)).to(self.device)
        
        for i in range(max_len):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
            
            with torch.no_grad():
                output, hidden, attention = self.model.decoder(trg_tensor, hidden, encoder_outputs)
            
            attentions[i] = attention
            
            pred_token = output.argmax(1).item()
            
            trg_indexes.append(pred_token)
            
            if pred_token == self.vi_vocab.lookup_token('</s>'):
                break
        
        # 4. Convert Output to Words
        trg_tokens = [self.vi_vocab.lookup_index(i) for i in trg_indexes]
        
        # Remove <s>
        trg_tokens = trg_tokens[1:]
        
        # Remove </s> if present
        if trg_tokens[-1] == '</s>':
            trg_tokens = trg_tokens[:-1]
            
        return " ".join(trg_tokens), attentions[:len(trg_tokens)], tokens

    def display_attention(self, sentence, translation, attention, save_path=None):
        """
        Plots the attention map.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        attention = attention.squeeze(1).cpu().detach().numpy()
        
        cax = ax.matshow(attention, cmap='bone')
        fig.colorbar(cax)
        
        # Set axes labels
        ax.set_xticklabels([''] + sentence, rotation=90)
        ax.set_yticklabels([''] + translation.split())
        
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
        if save_path:
            plt.savefig(save_path)
            print(f"Attention map saved to {save_path}")
        else:
            plt.show()
