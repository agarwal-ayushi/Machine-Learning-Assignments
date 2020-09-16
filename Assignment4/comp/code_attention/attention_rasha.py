import torch
import torch.nn as nn

#Inspired by Show, Attend and Tell Module

class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_lin = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.img_lin = nn.Linear(encoder_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
        self.concat_lin = nn.Linear(hidden_dim, 1)

    def forward(self, img_features, hidden_state):
        #print("hidden state dim=", hidden_state.shape)
        hidden_h = self.hidden_lin(hidden_state).unsqueeze(1)
        #print(hidden_h.shape)
        img_s = self.img_lin(img_features)
        #print(img_s.shape)
        att_ = self.tanh(img_s + hidden_h)
        e_ = self.concat_lin(att_).squeeze(2)
        alpha = self.softmax(e_)
        context_vec = (img_features * alpha.unsqueeze(2)).sum(1)
        return context_vec, alpha
