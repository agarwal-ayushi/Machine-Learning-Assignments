from __future__ import print_function
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
from utils_rasha import get_data
from models_rasha import Encoder_Rasha, Decoder_Rasha
import time
from test_rasha import test_rasha
import matplotlib.pyplot as plt

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
    logging.info("Loaded Training Data")
    encoder_model = Encoder_Rasha(embed_dim, train_enc)
    decoder_model = Decoder_Rasha(embed_dim, hidden_dim, vocab_len, num_layers)
    #encoder_model.load_state_dict(torch.load(os.path.join(path,"encoder_model_rasha.pth")))  ##Load the models
    #decoder_model.load_state_dict(torch.load(os.path.join(path,"decoder_model_rasha.pth")))
    encoder_model = encoder_model.cuda()
    decoder_model = decoder_model.cuda()
    if (train_enc):
        params_rasha = list(encoder_model.parameters()) + list(decoder_model.parameters()) #+ list(encoder_model.bn.parameters())
    # Use this when using pretrained
    else:
        # For VGG
        #params_rasha = list(encoder_model.linear.parameters()) + list(decoder_model.parameters()) #+ list(encoder_model.bn.parameters())
        params_rasha = list(encoder_model.linear.parameters()) + list(decoder_model.parameters()) + list(encoder_model.bn.parameters())
        #params_rasha = list(encoder_model.parameters()) + list(decoder_model.parameters()) # For fine-tuning : Does not give significant gain
    optimizer = torch.optim.Adam(params_rasha, lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    if train :
        train_total_loss =[]
        logging.info("-------Starting Training------------")
        for epoch in range(epochs):
            i=0
            print(scheduler.get_lr())
            training_loss=[]
            start_epoch = time.time()
            for batchid,(images, captions, lengths) in enumerate(train_dl):
                start = time.time()
                optimizer.zero_grad()
                
                # Experiments with length to LSTM
                train = torch.zeros(len(captions), max(lengths)-1)
                for i, cap in enumerate(captions):
                    train[i]=np.delete(cap.data, lengths[i]-1)
                train = train.cuda().to(dtype = torch.long)

                images= images.cuda();captions= captions.cuda() ## Load into GPU
                target_rasha = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted =False)[0]  ## Pack by removing zero pads using lengths
                #print(target_rasha.shape)
                encode_feat_rasha = encoder_model(images.float(), train_enc)  # Encoded image features
                decode_out = decoder_model(encode_feat_rasha, train, lengths) # Output of Decoder module
                #print(decode_out.shape)
                #decode_out = decoder_model(encode_feat_rasha, captions, lengths) # Output of Decoder module
                loss = criterion(decode_out, target_rasha);
                loss.backward()
                optimizer.step()
                training_loss.append(loss)
                if (batchid % 300 == 0) :
                    logging.info(f"Batch Number: {batchid} \t loss:{training_loss[-1]}")
                if (batchid % 500 == 0) :
                    logging.info(f"Batch Number: {batchid} \t Time taken:{time.time()-start}")
                
            scheduler.step()
            train_loss_=torch.stack(training_loss)
            train_total_loss.append(train_loss_)
            train_loss=torch.stack(training_loss).mean().item()
            print(f'Epoch : {epoch}  Training Loss:{train_loss}')
            print(f'Time Taken to train Epoch : {epoch} is : {time.time()-start_epoch}')
            torch.save(encoder_model.state_dict(), os.path.join(path,f"encoder_model_rasha.pth"))
            torch.save(decoder_model.state_dict(), os.path.join(path,f"decoder_model_rasha.pth"))
            test_rasha(embed_dim, hidden_dim, 1, 5, vocab_len, vocab_dict, train_enc)  # To calculate BS after every epoch
        total_loss = [l.cpu().detach().numpy() for l in train_total_loss]
        total_loss = np.ravel(total_loss)
        fig , ax = plt.subplots(1,1)
        x = range(len(total_loss))
        ax.plot(x,total_loss)
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("Loss")
        ax.set_xticks(x)
        fig.savefig("plot.png")
    
    else :
        ## Load pretrained models
        encoder_model.load_state_dict(torch.load(os.path.join(path,"encoder_model_rasha.pth")))
        decoder_model.load_state_dict(torch.load(os.path.join(path,"decoder_model_rasha.pth")))
        
    return vocab_len, vocab_dict
    
    
#train_image_caption_rasha(256, hidden_dim=256, batch=8, epochs=1 ,learning_rate=0.01, train=True) 
