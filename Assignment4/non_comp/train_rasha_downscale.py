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
from utils_rasha_downscale import get_data
from models_rasha_downscale import Encoder_Rasha, Decoder_Rasha
import time
from test_rashadownscale import test_rasha
import matplotlib.pyplot as plt

criterion = nn.CrossEntropyLoss()   # This Loss does Softmax and NLL (Neg Log Likelihood) Loss
path = '.' 
#num_layers=1 
def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

#Ramya - added new flags : FLAGS.Resnet50, FLAGS.Resnet152
# Ayushi - added flag for VGG19
def train_image_caption_rasha(embed_dim, hidden_dim, batch, epochs, learning_rate, num_layers, train, train_enc, Resnet50, Resnet152, vgg19):
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
    encoder_lr = 0.1
    train_dl, vocab_len, vocab_dict = get_data(batch)
    logging.info("Loaded Training Data")
    # Ramya: Pass Flags to invoke ResNet
    # removed train_enc for resnet_training
    encoder_model = Encoder_Rasha(embed_dim, Resnet50, Resnet152, vgg19)
    decoder_model = Decoder_Rasha(embed_dim, hidden_dim, vocab_len, num_layers)
    encoder_model = encoder_model.cuda()
    decoder_model = decoder_model.cuda()
    # params_rasha = list(encoder_model.parameters()) + list(decoder_model.parameters()) #+ list(encoder_model.bn.parameters())
    # Ramya: If pretrained take encoder_model.linear.parameters otherwise encoder_model,parameters()  
    params_rasha = list(encoder_model.parameters()) + list(decoder_model.parameters()) 
    decoder_optimizer = torch.optim.Adam(list(decoder_model.parameters()), lr = learning_rate)
    encoder_optimizer = torch.optim.SGD(list(encoder_model.parameters()), lr = encoder_lr, momentum=0.9, weight_decay=0.0005)
    encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=2, gamma=0.5) 
    if train :
        logging.info("-------Starting Training------------"); train_total_loss =[];
        for epoch in range(epochs): 
            training_loss=[]
            print(encoder_scheduler.get_lr())
            start_epoch = time.time()
            for batchid,(images, captions, lengths) in enumerate(train_dl):
                start = time.time()
                
                # Experiments with length to LSTM
                train = torch.zeros(len(captions), max(lengths)-1)
                for i, cap in enumerate(captions):
                    train[i]=np.delete(cap.data, lengths[i]-1)
                train = train.cuda().to(dtype = torch.long)

                images= images.cuda();captions= captions.cuda() ## Load into GPU
                target_rasha = pack_padded_sequence(captions, lengths, batch_first=True, enforce_sorted =False)[0]  ## Pack by removing zero pads using lengths
                encode_feat_rasha = encoder_model(images.float())  # Encoded image features
                decode_out = decoder_model(encode_feat_rasha, train, lengths) # Output of Decoder module
                loss = criterion(decode_out, target_rasha);
                decoder_optimizer.zero_grad();
                encoder_optimizer.zero_grad();
                loss.backward()
#                print(encoder_model.model.model[0].weight)
                decoder_optimizer.step()
                encoder_optimizer.step()
                training_loss.append(loss.cpu().detach().numpy()) 
                if (batchid % 300 == 0) :
                    logging.info(f"Batch Number: {batchid} \t loss:{training_loss[-1]}")
                if (batchid % 500 == 0) :
                    logging.info(f"Batch Number: {batchid} \t Time taken:{time.time()-start}")
               # if(batchid == 1): 
               #     break
            encoder_scheduler.step() 
            train_total_loss.append(training_loss)
            
            print(f'Epoch : {epoch}  Training Loss:{np.mean(training_loss)}')
            print(f'Time Taken to train Epoch : {epoch} is : {time.time()-start_epoch}')
            # Saving model
            torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder_model.state_dict(),
            'decoder_state_dict': decoder_model.state_dict(),
            'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
            'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
            'loss': loss,
            }, "curr_model.pk")
            #torch.save(encoder_model.state_dict(), os.path.join(path,f"encoder_model_rasha_resnet_scratch.pth"))
            #torch.save(decoder_model.state_dict(), os.path.join(path,f"decoder_model_rasha_resnet_scratch.pth"))
            test_rasha(embed_dim, hidden_dim, 1, vocab_len, vocab_dict, Resnet50, Resnet152, vgg19) 
      #  total_loss = [l.cpu().detach().numpy() for l in train_total_loss];
        total_loss = np.ravel(train_total_loss)
        fig , ax = plt.subplots(1,1);x = range(len(total_loss));ax.plot(x,total_loss);ax.set_xticks(x);
        fig.savefig("plot.png")       
    
    else :
        ## Load pretrained models
        encoder_model.load_state_dict(torch.load(os.path.join(path,"resent_from_scratch_encoder_model_rasha.pth")))
        decoder_model.load_state_dict(torch.load(os.path.join(path,"resent_from_scratch_decoder_model_rasha.pth")))
        
    return vocab_len, vocab_dict
    
    
#train_image_caption_rasha(256, hidden_dim=256, batch=8, epochs=1 ,learning_rate=0.01, train=True) 
