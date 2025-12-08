import json
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from .translator import Translator

class ResultSaver:
    def __init__(self, save_path):
        self.save_path = save_path
        self.results = []

    def add_result(self, src, ref, pred):
        self.results.append({
            "src": src,
            "ref": ref,
            "pred": pred
        })

    def save(self):
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        print(f"Results saved to {self.save_path}")

class Evaluator:
    def __init__(self, model, en_vocab, vi_vocab, device):
        self.translator = Translator(model, en_vocab, vi_vocab, device)
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.device = device

    def evaluate_batch(self, test_loader, result_saver=None, beam_size=1, length_penalty_alpha=0.7, no_repeat_ngram_size=0):
        """
        Returns:
            bleu_score (float): BLEU score of the entire test set.
        """
        refs = []
        preds = []
        
        print(f"Evaluating on test set (Beam Size={beam_size})...")
        for src, trg in tqdm(test_loader, desc="Testing"):
            # src = [batch size, src len]
            # trg = [batch size, trg len]
            
            batch_size = src.shape[0]
            for i in range(batch_size):
                src_indices = src[i]
                trg_indices = trg[i]
                

                src_tokens = [self.en_vocab.lookup_index(idx.item()) for idx in src_indices if idx.item() not in [self.en_vocab.lookup_token('<pad>'), self.en_vocab.lookup_token('<s>'), self.en_vocab.lookup_token('</s>')]]
                src_text = " ".join(src_tokens)
                
                trg_tokens = [self.vi_vocab.lookup_index(idx.item()) for idx in trg_indices if idx.item() not in [self.vi_vocab.lookup_token('<pad>'), self.vi_vocab.lookup_token('<s>'), self.vi_vocab.lookup_token('</s>')]]
                ref_text = " ".join(trg_tokens)
                

                pred_text, _, _ = self.translator.translate_sentence(src_text, beam_size=beam_size, length_penalty_alpha=length_penalty_alpha, no_repeat_ngram_size=no_repeat_ngram_size)
                

                refs.append([trg_tokens])
                preds.append(pred_text.split())
                
                if result_saver:
                    result_saver.add_result(src_text, ref_text, pred_text)
                    

        score = corpus_bleu(refs, preds)
        return score

