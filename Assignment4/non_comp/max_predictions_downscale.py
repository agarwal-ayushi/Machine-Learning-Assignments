import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [to_device(x,device) for x in data]
        return data.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def max_prediction(encoded_features, decoder_model, vocab_dict, max_length=80):
    '''
       Function to perform max prediction to generate captions
    '''
    
    word_2_ix, ix_2_word = vocab_dict
    start_token = word_2_ix['<start>']
    #print(start_token)
    end_token = word_2_ix['<end>']
    #print(end_token)
    hidden = None # In the beginning the hidden state is None
    caption_word_id = []
    for i in range(max_length):
        encoded_features = encoded_features.unsqueeze(1)
        if(hidden == None):
            output, hidden = decoder_model.get_pred(encoded_features.cuda())
        else:
            output, hidden = decoder_model.get_pred(encoded_features.cuda(), to_device(hidden, device))
     
        #print(output.shape)
        _ , predicted_id = output.max(1)
        caption_word_id.append(predicted_id)
        if (predicted_id == end_token):
            break
        encoded_features = decoder_model.embed(predicted_id)
    caption_word_id = torch.stack(caption_word_id, 1)
    return caption_word_id.cpu().numpy()[0]
