import transformers
import torch.nn as nn
import torch
import torch.nn.functional as F


class BERTClassification(nn.Module):
    def __init__ (self):
        super(BERTClassification, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-cased')
        self.bert_drop = nn.Dropout(0.4)
        self.out = nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, pooledOut = self.bert(ids, attention_mask = mask,
                                token_type_ids=token_type_ids)
        bertOut = self.bert_drop(pooledOut)
        output = self.out(bertOut)

        return output

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


    def get_attn_pad_mask(seq_q, seq_k):
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        # eq(zero) is PAD token
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
        return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class EncoderLayer(nn.Module):
   def __init__(self):
       super(EncoderLayer, self).__init__()
       self.enc_self_attn = MultiHeadAttention()
       self.pos_ffn = PoswiseFeedForwardNet()

   def forward(self, enc_inputs, enc_self_attn_mask):
       enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
       enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
       return enc_outputs, attn


class MultiHeadAttention(nn.Module):
   def __init__(self):
       super(MultiHeadAttention, self).__init__()
       self.W_Q = nn.Linear(d_model, d_k * n_heads)
       self.W_K = nn.Linear(d_model, d_k * n_heads)
       self.W_V = nn.Linear(d_model, d_v * n_heads)

   def forward(self, Q, K, V, attn_mask):
       # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
       residual, batch_size = Q, Q.size(0)
       # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
       q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
       k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
       v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

       attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

       # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
       output = nn.Linear(n_heads * d_v, d_model)(context)


       return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]

class ScaledDotProductAttention(nn.Module):
   def __init__(self):
       super(ScaledDotProductAttention, self).__init__()

   def forward(self, Q, K, V, attn_mask):
       scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
       scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
       attn = nn.Softmax(dim=-1)(scores)
       context = torch.matmul(attn, V)
       return score, context, attn

emb = Embedding()
embeds = emb(input_ids, segment_ids)

attenM = get_attn_pad_mask(input_ids, input_ids)

SDPA= ScaledDotProductAttention()(embeds, embeds, embeds, attenM)

S, C, A = SDPA


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

       return logits_lm, logits_clsf


model = BERT()
batch = make_batch()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(*batch))

for epoch in range(100):
    optimizer.zero_grad()
    logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
    loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens) # for masked LM
    loss_lm = (loss_lm.float()).mean()
    loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
    loss = loss_lm + loss_clsf
    if (epoch + 1) % 10 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    loss.backward()
    optimizer.step()

# Predict mask tokens
input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[0]))
print(text)
print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])

logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext : ', True if isNext else False)
print('predict isNext : ',True if logits_clsf else False)