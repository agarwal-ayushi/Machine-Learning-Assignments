from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
from utils_rasha import to_device
import numpy as np
from absl import flags,app,logging
from utils_rasha import get_data
from new_models_rasha import Encoder_Rasha, Decoder_Rasha
import time
from test_rasha import test_rasha
import matplotlib.pyplot as plt

alpha_c = 1
x=[]
criterion = nn.CrossEntropyLoss()   # This Loss does Softmax and NLL (Neg Log Likelihood) Loss
path = '.' 
#num_layers=1 
def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def train_image_caption_rasha(embed_dim, hidden_dim, batch, epochs, learning_rate, num_layers, train, train_enc=False):
    '''
    Function for training Encoder-Decoder network
    
    Parameters
    ----------
    embed_dim : int
        Embedding Dimension.
    hidden_dim : int
        LSTM Hidden state dimension
    batch : int
        Batch Size
    epochs : int
        Num of epochs
    learning_rate : float
        Learning rate for training
    train : bool
        Whether to train from scratch or load pretrained modules

    Returns
    -------
    vocab_len : Length of vocabulary
    

    '''
    device = get_device()
    train_dl, vocab_len, vocab_dict = get_data(batch)
    logging.info("---------Loaded Training Data-------------")
    encoder_model = Encoder_Rasha("resnet50")
    decoder_model = Decoder_Rasha( vocab_len, embed_dim, hidden_dim, encoder_model.image_enc_dim)
    encoder_model = encoder_model.cuda()
    decoder_model = decoder_model.cuda()
    params_rasha = list(decoder_model.parameters())
    optimizer = torch.optim.Adam(params_rasha, lr = learning_rate)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    if train :
        train_total_loss =[]
        logging.info("-------Starting Training------------")
        for epoch in range(epochs):
            i=0
#            print("The current learning rate is =", scheduler.get_lr())
            
            training_loss=[]
            start_epoch = time.time()
            for batchid,(images, captions, lengths) in enumerate(train_dl):
                start = time.time()
                # Experiments with length to LSTM
                # Remove <end> token from the train token because it does not need to predict the next token after <end>
                train = torch.zeros(len(captions), max(lengths)-1)
                for i, cap in enumerate(captions):
                    train[i]=np.delete(cap.data, lengths[i]-1)
                
                train = train.cuda().to(dtype = torch.long)
                lengths = [l-1 for l in lengths]
                
                img_features = encoder_model(images.float().cuda())
                #print(encoder_model.model[7][0].conv1.weight.grad)
                preds, alphas = decoder_model(img_features, train, lengths)
                preds = pack_padded_sequence(preds, lengths, batch_first=True, enforce_sorted = False)[0]
                
                targets = captions[:, 1:].cuda()
                targets = pack_padded_sequence(targets,lengths, batch_first=True,enforce_sorted = False)[0]
                
                loss = criterion(preds, targets)
                att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
                loss += to_device(att_regularization, device)
                optimizer.zero_grad()
                loss.backward()
                #print(loss.device)
                optimizer.step()
                #training_loss.append(loss)
                if (batchid % 300 == 0) :
                    logging.info(f"Batch Number: {batchid} \t loss:{loss.item()}")
                if (batchid % 500 == 0) :
                    logging.info(f"Batch Number: {batchid} \t Time taken:{time.time()-start}")
               # if batchid ==1:break
           
            #scheduler.step()
            print(f'Time Taken to train Epoch : {epoch} is : {time.time()-start_epoch}')
            torch.save(encoder_model.state_dict(), os.path.join(path,f"encoder_model_rasha.pth"))
            torch.save(decoder_model.state_dict(), os.path.join(path,f"decoder_model_rasha.pth"))
            test_rasha(embed_dim, hidden_dim, 1, 5, vocab_len, vocab_dict, train_enc)  # To calculate BS after every epoch on public test data
    
    else :
        ## Load pretrained models
        encoder_model.load_state_dict(torch.load(os.path.join(path,"encoder_model_rasha.pth")))
        decoder_model.load_state_dict(torch.load(os.path.join(path,"decoder_model_rasha.pth")))
        
    return vocab_len, vocab_dict
    
    
#train_image_caption_rasha(256, hidden_dim=256, batch=8, epochs=1 ,learning_rate=0.01, train=True) 
