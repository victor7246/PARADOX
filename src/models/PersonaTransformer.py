import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Embedder, EncoderwithPersona, DecoderwithPersona, Encoder, Decoder

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, use_fame=True, bert_model=None, use_alignment=False):
        super().__init__()
        self.d_model = d_model
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout, use_fame=use_fame)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, use_fame=use_fame)
        self.out = nn.Linear(d_model, trg_vocab)

        self.use_alignment = use_alignment

        if self.use_alignment == True:
          self.alignment_proj1 = nn.Linear(d_model, int(d_model**.5))
          self.alignment_proj2 = nn.Linear(d_model, int(d_model**.5))

        if bert_model:
          self.bert_model = bert_model
          for j, p in enumerate(self.bert_model.parameters()):
            p.requires_grad_(False)

        else:
          self.bert_model = None

    def forward(self, src, trg, speaker_id, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        #output = self.out(d_output)

        if self.use_alignment == True:
          projected_embeddings1 = self.alignment_proj1(self.decoder.embed.embed.weight.clone())
          projected_embeddings2 = self.alignment_proj2(self.decoder.embed.embed.weight.clone())

          proj_matrix = nn.Softmax()(torch.matmul(projected_embeddings1, projected_embeddings2.T)/math.sqrt(self.d_model**.5))

          output = torch.matmul(self.out(d_output), proj_matrix) + self.out(d_output)
        else:
          output = self.out(d_output)
          proj_matrix = 0

        if self.bert_model:
          e_output = e_outputs[:,0,:]
          
          bert_output = self.bert_model(src)[1]
          
          #kl_loss = nn.KLDivLoss(reduction='mean')(e_output.view(-1,e_output.size()[-1]), bert_output.view(-1,bert_output.size()[-1]))
          kl_loss = nn.MSELoss()(e_output, bert_output)
        else:
          kl_loss = 0

        self.kl = 0
        
        return output, kl_loss, e_outputs, 0


class TransformerwithPersona(nn.Module):
    def __init__(self, src_vocab, trg_vocab, speaker_count, d_model, N, heads, dropout, \
                 bert_model=None, generate_speaker_emb=False, use_fame=True, use_alignment=True,\
                  use_linear_speaker_emb=False, no_speaker_id=False):
        super().__init__()
        self.speaker_emb = Embedder(speaker_count, d_model)
        self.encoder = EncoderwithPersona(src_vocab, d_model, N, heads, dropout, use_fame)
        self.decoder = DecoderwithPersona(trg_vocab, d_model, N, heads, dropout, use_fame)
        
        self.d_model = d_model
        self.no_speaker_id = no_speaker_id
        self.use_alignment = use_alignment
        self.use_linear_speaker_emb = use_linear_speaker_emb

        if self.use_alignment == True:
          self.alignment_proj1 = nn.Linear(d_model, int(d_model**.5))
          self.alignment_proj2 = nn.Linear(d_model, int(d_model**.5))
        self.out = nn.Linear(d_model, trg_vocab)

        self.generate_speaker_emb = generate_speaker_emb

        self.normal = torch.distributions.Normal(0, 1)
        if torch.cuda.is_available():
            self.normal.loc = self.normal.loc.cuda() # hack to get sampling on the GPU
            self.normal.scale = self.normal.scale.cuda()

        if bert_model:
          self.bert_model = bert_model
          for j, p in enumerate(self.bert_model.parameters()):
            p.requires_grad_(False)

        else:
          self.bert_model = None

    def forward(self, src, trg, speaker_id, src_mask, trg_mask):
        if self.no_speaker_id == False:
          speaker_emb_ = self.speaker_emb(speaker_id)
        else:
          speaker_emb_ = torch.zeros((src.shape[0], src.shape[1], self.d_model)).to(src.device)
          
        if len(speaker_emb_.size()) == 2:
            speaker_emb_ = speaker_emb_.unsqueeze(1)

        e_outputs, mu, sigma = self.encoder(src, speaker_emb_, src_mask)

        if self.generate_speaker_emb:
          speaker_emb = mu + sigma*self.normal.sample(mu.shape)
        else:
          speaker_emb = torch.zeros_like(e_outputs)

        if self.use_linear_speaker_emb == False:
          e_output2 = e_outputs + speaker_emb
        else:
          e_output2 = e_outputs + mu
        
        if self.generate_speaker_emb:
          self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp()) #
        else:
          self.kl = 0

        #print("DECODER")
        d_output = self.decoder(trg, e_output2, src_mask, trg_mask)

        if self.use_alignment == True:
          projected_embeddings1 = self.alignment_proj1(self.decoder.embed.embed.weight.clone())
          projected_embeddings2 = self.alignment_proj2(self.decoder.embed.embed.weight.clone())

          proj_matrix = nn.Softmax()(torch.matmul(projected_embeddings1, projected_embeddings2.T)/math.sqrt(self.d_model**.5))

          output = torch.matmul(self.out(d_output), proj_matrix) + self.out(d_output)
        else:
          output = self.out(d_output)
          proj_matrix = 0
        
        if self.bert_model:
          e_output = e_outputs[:,0,:]
          
          bert_output = self.bert_model(src)[1]
          
          #kl_loss = nn.KLDivLoss(reduction='mean')(e_output.view(-1,e_output.size()[-1]), bert_output.view(-1,bert_output.size()[-1]))
          kl_loss = nn.MSELoss()(e_output, bert_output)
        else:
          kl_loss = 0

        return output, kl_loss, e_outputs, proj_matrix