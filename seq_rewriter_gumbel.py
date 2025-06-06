# old imported code
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class source_model(nn.Module):
    def __init__(self, options):
        super(source_model, self).__init__()
        self.options = options
        self.eps = 1e-6
        self.char_embedding = nn.Embedding(options['vocab_size'], options['vocab_size'])
        self.char_embedding.weight.data = torch.eye(options['vocab_size'])
        self.char_embedding.weight.requires_grad = False

        self.conv1 = nn.Conv1d(options['vocab_size'], options['target_size'], 
            kernel_size = options['filter_width'], padding = int(options['filter_width']/2))
        self.saved_log_probs = []
        self.probs = None
        self.entropy = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_seq_length = options['target_sequence_length']
        # self.lstm = nn.LSTM(options['vocab_size'], options['hidden_size'], batch_first = True)
        # self.output_layer = nn.Linear(options['hidden_size'], options['target_size'])
        

    def forward(self, sentence_batch, temp = 1.0):
        if self.target_seq_length > sentence_batch.size()[1]:
            sentence_batch = F.pad(sentence_batch, (0, self.target_seq_length - sentence_batch.size()[1]))
        else:
            sentence_batch = sentence_batch[:,0:self.target_seq_length]
            
        one_hot = self.char_embedding(sentence_batch)
        logits = self.conv1(one_hot.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        
        logits_shape = logits.size()
        logits = logits.view(logits.size(0) * logits.size(1), logits.size(2))
        if self.training:
            probs = F.gumbel_softmax(logits, tau=temp)
        else:
            probs = F.softmax(logits)
        self.probs = probs.view(logits_shape[0], logits_shape[1], logits_shape[2])
        
        _, new_seq = torch.max(probs, 1)
        new_seq = new_seq.view(sentence_batch.size(0), sentence_batch.size(1))

        return new_seq

def main():
    rnn_options = {
        'vocab_size' : 100,
        'hidden_size' : 200,
        'target_size' : 3
    }

    chrrnn = source_model(rnn_options)
    sent_batch = torch.LongTensor(32, 10).random_(0, 10)
    chrrnn(sent_batch)


if __name__ == '__main__':
    main()