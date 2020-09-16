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

#Ramya
from vgg19_model import VGG19
class Encoder_Rasha(nn.Module):
    '''
    This class represents the encoder CNN module. Used network is VGG-19 architecture. 
    Parameters:
        embed_size : The embedding dimension to which images are encoded into
    Inputs :
        Batch of images
    Outputs :
        Encoded feature vectors
    '''
    def __init__(self, embed_size=256, Resnet50 = False, Resnet152 = False, vgg19=True):
        super().__init__()
        self.embed_size = embed_size
        	
        if (vgg19):
            print("----Training VGG-19 Model from scratch for the Encoder----")
            self.model = VGG19(self.embed_size)
    
    def forward(self, feat, train_enc=True):
       feat = self.model(feat)
       return feat
    
class Decoder_Rasha(nn.Module):
    '''
    This class represents the Decoder Module which consists of LSTM layers
    Parameters:
        embed_size : Embedding dimension of words and images
        hidden_size : hidden_state dimension of LSTM
        vocab_size : Length of vocabulary
        num_layers : Number of LSTM layers
        
    Input :
        features : Encoded image features
        captions : Tokenized training captions
        lengths: Length of each sequence 
    
    Output :
        Outputs probability distribution over vocabulary ( dimension : 1 * vocab_size)
    '''
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """Set the hyper-parameters and build the layers."""
        super(Decoder_Rasha, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        #self.relu = nn.ReLU(inplace = True)   # Performs worse since LSTM already has sigmoid activation
        self.dropout = nn.Dropout(p=0.5, inplace = True)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)   #Embedd tokenized captions into latent space
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # Concatenate image enocded features with embedded captions
        embeddings = self.dropout(embeddings)
        # Dropout after concatenation leads to better Bleu Score
        packed_seq = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted= False)
        hiddens_rasha, _ = self.lstm(packed_seq)
        outputs = self.linear(hiddens_rasha[0])  #Pass output of lstm through a linear layer to get prob. dist. over vocab
        #outputs = self.relu(self.linear_1(hiddens_rasha[0]))  #Pass output of lstm through a linear layer to get prob. dist. over vocab
        return outputs


#This function is used for max_prediction. Due to lack of time, we could not clean the code
    def get_pred(self, features, hidden=None):
        output, hidden = self.lstm(features, hidden)
        #print("Output shape= ", output.shape)
        output = self.linear(output.squeeze(1))
        return output, hidden

#This function is used in Beam Search
    def get_bs_pred(self, features, hidden=None):
        ''' Helper Function for Beam Search'''
        if(hidden != None):
            features = self.embed(features).unsqueeze(1)
        output, hidden = self.lstm(features, hidden)
        output = self.linear(output.squeeze(1))
        return output, hidden
