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

    def translate_sentence(self, sentence, max_len=50, beam_size=1, length_penalty_alpha=0.7):
        """
        Translates a single sentence using Greedy or Beam Search.
        Args:
            sentence (str): Input sentence.
            max_len (int): Maximum length.
            beam_size (int): 1 for Greedy, > 1 for Beam Search.
            length_penalty_alpha (float): Alpha for length normalization.
        """

        tokens = process_line(sentence)
        tokens = tokens + ['</s>']
        
        src_indexes = [self.en_vocab.lookup_token(token) for token in tokens]
        
        # Convert to tensor
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(self.device)
        # src_tensor = [1, src_len]
        
        # 2. Forward Encoder
        with torch.no_grad():
            encoder_outputs, hidden = self.model.encoder(src_tensor)
        
        if beam_size == 1:
            return self._greedy_decode(hidden, encoder_outputs, src_indexes, max_len, tokens)
        else:
            return self._beam_search_decode(hidden, encoder_outputs, src_indexes, max_len, beam_size, length_penalty_alpha, tokens)


    def _greedy_decode(self, hidden, encoder_outputs, src_indexes, max_len, src_tokens):
        trg_indexes = [self.vi_vocab.lookup_token('<s>')]
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
        
        return self._post_process(trg_indexes, attentions[:len(trg_indexes)-1], src_tokens)

    def _beam_search_decode(self, hidden, encoder_outputs, src_indexes, max_len, beam_size, alpha, src_tokens):
        beams = [(0, [self.vi_vocab.lookup_token('<s>')], hidden, [])]
        
        completed_beams = []
        
        for _ in range(max_len):
            candidates = []
            
            for score, trg_seq, hid, attns in beams:
                # If sequence already ended, push to completed and skip
                if trg_seq[-1] == self.vi_vocab.lookup_token('</s>'):
                    completed_beams.append((score, trg_seq, hid, attns))
                    continue
                
                # Decoder Step
                dec_input = torch.LongTensor([trg_seq[-1]]).to(self.device)
                with torch.no_grad():
                    output, new_hid, attention = self.model.decoder(dec_input, hid, encoder_outputs)
                
                # output: [1, vocab_size] -> log_softmax for probabilities
                log_probs = torch.nn.functional.log_softmax(output, dim=1)
                
                # Get top K candidates for this beam
                topk_probs, topk_ids = log_probs.topk(beam_size)
                
                for k in range(beam_size):
                    sym_prob = topk_probs[0][k].item()
                    sym_id = topk_ids[0][k].item()
                    
                    new_score = score + sym_prob
                    new_seq = trg_seq + [sym_id]
                    new_attns = attns + [attention]
                    
                    candidates.append((new_score, new_seq, new_hid, new_attns))
            
            # If no candidates (all beams finished), break
            if not candidates:
                break
                
            # Select Top K candidates globally
            # Sort by score (descending)
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]
            
            # If all current beams are finished, stop early
            if all(b[1][-1] == self.vi_vocab.lookup_token('</s>') for b in beams):
                completed_beams.extend(beams)
                break
        
        # Add remaining active beams to completed list
        completed_beams.extend(beams)
        
        # Apply Length Penalty: score = log_prob / length^alpha
        # We want to maximize this score
        best_score = float('-inf')
        best_beam = None
        
        for score, seq, _, attns in completed_beams:
            length = len(seq)
            # Avoid division by zero, though seq usually has at least <s>
            lp = (length ** alpha) if length > 0 else 1.0
            avg_score = score / lp
            
            if avg_score > best_score:
                best_score = avg_score
                best_beam = (seq, attns)
        
        final_seq, final_attns = best_beam
        
        # Convert attention list to tensor
        # final_attns is list of [1, src_len] tensors
        if final_attns:
            final_attns_tensor = torch.cat(final_attns, dim=0).unsqueeze(1) # [trg_len, 1, src_len]
        else:
            final_attns_tensor = torch.zeros(len(final_seq), 1, len(src_indexes)).to(self.device)

        return self._post_process(final_seq, final_attns_tensor, src_tokens)


    def _post_process(self, trg_indexes, attentions, src_tokens):
        trg_tokens = [self.vi_vocab.lookup_index(i) for i in trg_indexes]
        
        trg_tokens = trg_tokens[1:]
        
        if trg_tokens and trg_tokens[-1] == '</s>':
            trg_tokens = trg_tokens[:-1]
            
        return " ".join(trg_tokens), attentions[:len(trg_tokens)], src_tokens

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
