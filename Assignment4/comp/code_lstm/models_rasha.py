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
    def __init__(self, embed_size=256, train_enc=False):
        super().__init__()
        self.embed_size = embed_size
        if not train_enc:
            # This is for Resnet 50
            print("----Picking up a pretrained ResNet50 Model for the Encoder----")
            pre_model = torchvision.models.resnet50(pretrained = True)
            self.model = nn.Sequential(*list(pre_model.children())[:-1]) # This rmeoves the last layer of the ResNet
            self.linear = nn.Linear(pre_model.fc.in_features, embed_size) 
            self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
            self.relu = nn.ReLU(inplace=True)
            
            # This is for VGG 
            #print("----Picking up a pretrained VGG-19 BN Model for the Encoder----")
            #pre_model = torchvision.models.vgg19_bn(pretrained = True)
            #module = list(pre_model.children())[:-1]
            #self.model = nn.Sequential(*list(pre_model.children())[:-1]) # This rmeoves the last layer of the ResNet
            #self.linear = nn.Linear(pre_model.classifier[0].in_features, embed_size) 
            #self.relu = nn.ReLU(inplace=True)
            # End VGG

            #print("----Picking up a pretrained GoogleNet Model for the Encoder----") Performs worse
            #pre_model = torchvision.models.googlenet(pretrained = True)
            
        else:
            print("----Training VGG-19 Model from scratch for the Encoder----")
            self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(128, 128, kernel_size=2, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(kernel_size=2, stride=2),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                       nn.ReLU(inplace = True),
                                       nn.MaxPool2d(kernel_size=2, stride=2))
            self.classifier = nn.Sequential(                  
               nn.Linear(in_features=512*7*7, out_features=4096),
               nn.ReLU(inplace=True),
               nn.Dropout(p=0.5, inplace=False),
               nn.Linear(in_features=4096, out_features=4096),
               nn.ReLU(inplace=True),
               nn.Linear(in_features=4096, out_features=self.embed_size),
               nn.ReLU(inplace=True),
                 )
    def forward(self, feat, train_enc=False):
       # Choose these lines for pretrained
       if not train_enc:
           with torch.no_grad():
               features = self.model(feat)
           features = features.view(features.shape[0], -1)
           features = self.bn(self.relu(self.linear(features)))
           # For VGG
           #features = self.relu(self.linear(features))
       # Choose these lines for training from sctratch
       else:
           feat = self.model(feat)
           # print("Feature Shape after Conv + Pool =", feat.shape)
           # Flatten the output of the convolution layer to the FC layer. 
           feat = feat.view(feat.shape[0],-1)
           features = self.classifier(feat) 
       return features
    
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
        self.dropout = nn.Dropout(p=0.5, inplace = False)
#        self.init_weights
        
    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)   #Embedd tokenized captions into latent space
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # Concatenate image enocded features with embedded captions
        #embeddings = self.dropout(embeddings)
        # Dropout after concatenation leads to better Bleu Score
        packed_seq = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted= False)
        hiddens_rasha, _ = self.lstm(packed_seq)
        #outputs = self.linear(hiddens_rasha[0])  #Pass output of lstm through a linear layer to get prob. dist. over vocab
        outputs = self.linear(self.dropout(hiddens_rasha[0]))  #Pass output of lstm through a linear layer to get prob. dist. over vocab
        return outputs


    def get_pred(self, features, hidden=None):
        '''Helper function for max_predictions'''
        output, hidden = self.lstm(features, hidden)
        output = self.linear(output.squeeze(1))
        return output, hidden

    def get_bs_pred(self, features, hidden=None):
        ''' Helper Function for Beam Search'''
        if(hidden != None):
            features = self.embed(features).unsqueeze(1)
        output, hidden = self.lstm(features, hidden)
        output = self.linear(output.squeeze(1))
        return output, hidden
