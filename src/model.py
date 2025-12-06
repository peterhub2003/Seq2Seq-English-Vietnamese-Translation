import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src = [batch size, src len]
        
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch size, src len, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        # outputs are always from the top hidden layer
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        
        # Bahdanau Attention
        # We concatenate hidden state (from decoder) and encoder output
        # So input dim is hid_dim (decoder) + hid_dim (encoder)
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, hid dim] (last layer hidden state from decoder)
        # encoder_outputs = [batch size, src len, hid dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        # hidden = [batch size, src len, hid dim]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        
        # Calculate energy
        # energy = [batch size, src len, hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        # Calculate attention
        # attention = [batch size, src len, 1]
        attention = self.v(energy)
        
        # Squeeze and softmax
        # return [batch size, src len]
        return F.softmax(attention.squeeze(2), dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(hid_dim + emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size] (one token at a time)
        # hidden = [n layers, batch size, hid dim]
        # encoder_outputs = [batch size, src len, hid dim]
        
        input = input.unsqueeze(1)
        # input = [batch size, 1]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [batch size, 1, emb dim]
        
        # Calculate attention weights
        # We use the last layer of hidden state for attention
        # a = [batch size, src len]
        a = self.attention(hidden[-1].unsqueeze(0), encoder_outputs)
        
        # a = [batch size, 1, src len]
        a = a.unsqueeze(1)
        
        # Calculate weighted source (context vector)
        # weighted = [batch size, 1, hid dim]
        weighted = torch.bmm(a, encoder_outputs)
        
        # Concatenate embedded input and weighted context
        # rnn_input = [batch size, 1, (emb dim + hid dim)]
        rnn_input = torch.cat((embedded, weighted), dim=2)
        
        # Pass to GRU
        output, hidden = self.rnn(rnn_input, hidden)
        
        # output = [batch size, 1, hid dim]
        # hidden = [n layers, batch size, hid dim]
        
        # Prediction
        # We concatenate embedded, weighted, and output to feed to FC
        # This is a common variation to improve performance
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        # prediction = [batch size, output dim]
        # a = [batch size, 1, src len] -> squeeze to [batch size, src len]
        return prediction, hidden, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Encoder forward
        encoder_outputs, hidden = self.encoder(src)
        
        # First input to decoder is the <sos> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Decoder forward
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # Store output
            outputs[:, t] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Next input is either ground truth or predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs