import torch.nn as nn
import torch
import torch.nn.functional as F
import bert_model
# from torch.autograd import Variable

class uniRNN(nn.Module):
    def __init__(self, options):
        super(uniRNN, self).__init__()
        self.options = options
        self.char_embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])
        self.lstm = nn.LSTM(options['embedding_size'], options['hidden_size'], batch_first = True)
        self.output_layer = nn.Linear(options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, hidden = None):
        # sentence_batch = Variable(sentence_batch)
        if len(sentence_batch.size()) == 2:
            char_embedding = self.char_embedding(sentence_batch)
        else:
            shape = sentence_batch.size()
            sentence_batch_flat = sentence_batch.view(shape[0] * shape[1], shape[2])
            char_embedding = torch.mm(sentence_batch_flat, self.char_embedding.weight)
            char_embedding = char_embedding.view(shape[0], shape[1], char_embedding.size()[-1])

        char_embedding = F.tanh(char_embedding)

        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        
        # print lstm_out.shape, lstm_out.shape[0] * lstm_out.shape[1]
        lstm_out = lstm_out.contiguous()
        # print type(lstm_out)
        # print lstm_out.shape
        lstm_out = lstm_out[:,-1,:]
        # lstm_out = lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        logits = self.output_layer(lstm_out)

        return logits


class biRNN(nn.Module):
    def __init__(self, options):
        super(biRNN, self).__init__()
        self.options = options
        self.drop = nn.Dropout(0.7)
        self.char_embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])
        self.lstm = nn.LSTM(options['embedding_size'], options['hidden_size'], batch_first = True, bidirectional=True)
        self.output_layer = nn.Linear(2*options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, hidden = None):
        # sentence_batch = Variable(sentence_batch)

        if len(sentence_batch.size()) == 2:
            char_embedding = self.char_embedding(sentence_batch)
        else:
            shape = sentence_batch.size()
            sentence_batch_flat = sentence_batch.view(shape[0] * shape[1], shape[2])
            char_embedding = torch.mm(sentence_batch_flat, self.char_embedding.weight)
            char_embedding = char_embedding.view(shape[0], shape[1], char_embedding.size()[-1])
            
        char_embedding = F.tanh(char_embedding)
        
        if not hidden:
            lstm_out, new_hidden = self.lstm(char_embedding)
        else:
            lstm_out, new_hidden = self.lstm(char_embedding, hidden)
        
        # print lstm_out.shape, lstm_out.shape[0] * lstm_out.shape[1]
        lstm_out = lstm_out.contiguous()
        # print lstm_out.size()
        # print type(lstm_out)
        # print lstm_out.shape
        lstm_out = lstm_out[:,-1,:] + lstm_out[:,0,:]
        # lstm_out = lstm_out.view(lstm_out.size(0) * lstm_out.size(1), lstm_out.size(2))
        logits = self.output_layer(lstm_out)

        return logits


class CnnTextClassifier(nn.Module):
    def __init__(self, options, window_sizes=(3, 4, 5)):
        super(CnnTextClassifier, self).__init__()
        self.options = options

        self.embedding = nn.Embedding(options['vocab_size'], options['embedding_size'])

        self.convs = nn.ModuleList([
            nn.Conv2d(1, options['hidden_size'], [window_size, options['embedding_size']], padding=(window_size - 1, 0))
            for window_size in window_sizes
        ])

        self.fc = nn.Linear(options['hidden_size'] * len(window_sizes), options['target_size'])

    def forward(self, sentence_batch):
        if len(sentence_batch.size()) == 2:
            char_embedding = self.embedding(sentence_batch)
        else:
            shape = sentence_batch.size()
            sentence_batch_flat = sentence_batch.view(shape[0] * shape[1], shape[2])
            char_embedding = torch.mm(sentence_batch_flat, self.embedding.weight)
            char_embedding = char_embedding.view(shape[0], shape[1], char_embedding.size()[-1])
            
        # x = self.embedding(char_embedding)           # [B, T, E]

        # Apply a convolution + max pool layer for each window size
        x = torch.unsqueeze(char_embedding, 1)       # [B, C, T, E] Add a channel dim.
        xs = []
        for conv in self.convs:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]

        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(x)             # [B, class]

        return logits

class BERT(nn.Module):
   def __init__(self):
       super(BERT, self).__init__()
       self.embedding = Embedding()
       self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
       self.fc = nn.Linear(d_model, d_model)
       self.activ1 = nn.Tanh()
       self.linear = nn.Linear(d_model, d_model)
       self.activ2 = gelu
       self.norm = nn.LayerNorm(d_model)
       self.classifier = nn.Linear(d_model, 2)
       # decoder is shared with embedding layer
       embed_weight = self.embedding.tok_embed.weight
       n_vocab, n_dim = embed_weight.size()
       self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
       self.decoder.weight = embed_weight
       self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

   def forward(self, input_ids, segment_ids, masked_pos):
       output = self.embedding(input_ids, segment_ids)
       enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
       for layer in self.layers:
           output, enc_self_attn = layer(output, enc_self_attn_mask)
       # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
       # it will be decided by first token(CLS)
       h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
       logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

       masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]

       # get masked position from final output of transformer.
       h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
       h_masked = self.norm(self.activ2(self.linear(h_masked)))
       logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

       return logits_lm




def main():
    rnn_options = {
        'vocab_size' : 100,
        'hidden_size' : 200,
        'target_size' : 2,
        'embedding_size' : 200,
    }

    chrrnn = CnnTextClassifier(rnn_options)
    sent_batch = torch.FloatTensor(32, 10, 100).random_(0, 10)


if __name__ == '__main__':
    main()