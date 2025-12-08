import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
        if bidirectional:
            self.fc = nn.Linear(hid_dim * 2, hid_dim)
        
    def forward(self, src):
        # src = [batch size, src len]
        
        embedded = self.dropout(self.embedding(src)) # [batch size, src len, emb dim]
        
        outputs, hidden = self.rnn(embedded)
        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        if self.bidirectional:
            
            # Reshape to [n_layers, 2, batch, hid]
            hidden = hidden.view(self.n_layers, 2, -1, self.hid_dim)
            
            # Concatenate the two directions: [n_layers, batch, hid * 2]
            hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
            
            hidden = torch.tanh(self.fc(hidden)) # [n_layers, batch, hid]
            
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, dec_hid_dim]
        # encoder_outputs = [batch size, src len, enc_hid_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat decoder hidden state src_len times
        # hidden = [batch size, src len, hid dim]
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        

        # [batch size, src len, hid dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        attention = self.v(energy) # [batch size, src len, 1]
        
        return F.softmax(attention.squeeze(2), dim=1)  # [batch size, src len]

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Input to GRU is [embedded input; weighted source context]
        # Context comes from encoder_outputs, so it has size enc_hid_dim
        self.rnn = nn.GRU(enc_hid_dim + emb_dim, dec_hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input = [batch size]
        # hidden = [n layers, batch size, dec_hid_dim]
        # encoder_outputs = [batch size, src len, enc_hid_dim]
        
        input = input.unsqueeze(1) # [batch size, 1]
        
        embedded = self.dropout(self.embedding(input)) # [batch size, 1, emb dim]
        
        # a = [batch size, 1, src len]
        a = self.attention(hidden[-1].unsqueeze(0), encoder_outputs)
        a = a.unsqueeze(1)
        
        
        weighted = torch.bmm(a, encoder_outputs) # [batch size, 1, enc_hid_dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2) # [batch size, 1, (emb dim + enc_hid_dim)]
        
        output, hidden = self.rnn(rnn_input, hidden)
        
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)
        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        
        return prediction, hidden, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def init_weights(self, mean=0, std=0.01):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=mean, std=std)
            else:
                nn.init.constant_(param.data, 0)
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = [batch size, src len]
        # trg = [batch size, trg len]
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src)
        
        input = trg[:, 0] # <sos>
        
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            outputs[:, t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            # Next input is either ground truth or predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs
