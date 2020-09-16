from __future__ import print_function
#import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
from absl import flags,app,logging
from attention_rasha import Attention

class Encoder_Rasha(nn.Module):
    def __init__(self, net='resnet50'):
        super().__init__()
        if net == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.image_enc_dim = 2048
        if net== 'resnet152' :
            self.model = torchvision.models.resnet50(pretrained = True)
            self.model = nn.Sequential(*list(self.model.children())[:-2])
            self.image_enc_dim = 2048
            
            
    def forward(self,feat):
        with torch.no_grad():
            feat = self.model(feat)
           # print(feat.shape)
            feat = feat.permute(0,2,3,1)
           # print(feat.shape)
            feat = feat.reshape(feat.shape[0],-1,feat.shape[-1])
           # print(feat.shape)
        return feat

class Decoder_Rasha(nn.Module):
    def __init__(self, vocab_len, embed_size, hidden_dim, enc_dim):
        super().__init__()
        self.vocab= vocab_len
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.enc_dim = enc_dim
        self.gate = nn.Linear(hidden_dim, enc_dim)
        self.h = nn.Linear(enc_dim, hidden_dim)
        self.c = nn.Linear(enc_dim, hidden_dim)
        self.lstm = nn.LSTMCell(embed_size + enc_dim, hidden_dim) # We need to concatenate the embedding and the image at each LSTM input now for attention
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.attention = Attention(enc_dim, hidden_dim)
        self.act= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.embedding = nn.Embedding(vocab_len, embed_size)
        self.out = nn.Linear(hidden_dim, vocab_len)
        
    def forward(self, img_feat, captions, length):
        h = self.act(self.h(img_feat.mean(dim=1)))
        c = self.act(self.c(img_feat.mean(dim=1)))
        max_len = max(length)
        embedding = self.embedding(captions)
        out_matrix = torch.zeros(img_feat.shape[0], max_len ,self.vocab).cuda()
        alpha_matrix = torch.zeros(img_feat.shape[0], max_len ,img_feat.shape[1]).cuda()
        
        for i in range(max(length)):
               context, alpha = self.attention(img_feat, h)
               #print(context.shape, alpha.shape)
               gate_out = self.sigmoid(self.gate(h))
               #print(gate_out.shape)
               context_gate = context * gate_out 
               in_ = torch.cat([embedding[:,i], context_gate],dim=1)
               #print(in_.shape)
               h,c = self.lstm(in_, (h,c))
               h = self.dropout(h)
               #h,c = self.lstm(in_, (h,c))
               out = self.out(h)
               out_matrix[:,i]=out
               alpha_matrix[:,i]=alpha
               
        return out_matrix, alpha_matrix
     
    def get_bs_pred(self,img_feat, cap, hidden):
        h,c = hidden
        context, alpha = self.attention(img_feat, h)
        gate_out = self.sigmoid(self.gate(h))
        context_gate = context * gate_out
        embedding = self.embedding(cap)
        in_ = torch.cat([embedding, context_gate],dim=1)
        h,c = self.lstm(in_, (h,c))
        out = self.out(h)
        return out, (h,c)
        
        
            
            
        
        
        
        
