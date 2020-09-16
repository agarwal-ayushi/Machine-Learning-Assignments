import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

''' Function to transfer data that is a tupe or a list to GPU and back'''
def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [to_device(x,device) for x in data]
        return data.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beam_search_pred(encoded_features, decoder_model, vocab_dict, beam_width = 3, max_length=80):
    '''
    Function to perform beam search to generate captions of the images
    Beam search is an algorithm that finds the best n predictions at 
    each step and uses those to find the next n predictions

    Input: 
    encoded_features: contains the test image encoded features
    '''
    word_2_ix , ix_2_word = vocab_dict
    start_token = word_2_ix['<start>']
    end_token = word_2_ix['<end>']
    # In the beginning the hidden state has to be None, since we are entering into the LSTM
    hidden = None
    caption_word_id = []
    temp_seq = []
    next_seq = []
    encoded_features = encoded_features.unsqueeze(1)
    # Generating the first set of three best tokens for the input features
    output, hidden = decoder_model.get_bs_pred(encoded_features.cuda())
    # Instead of maximizing product of probablities (can underflow for long captions), we maximize sum of log probs
    output = F.log_softmax(output, dim=1)
    prob , predicted_ids = torch.topk(output, beam_width)
    prob = prob.cpu().detach().numpy().squeeze(0)
    predicted_ids = predicted_ids.cpu().detach().numpy().squeeze(0)
    for ids in range(len(predicted_ids)):
        temp_seq.append([[predicted_ids[ids]], prob[ids], hidden])
    #i = 0
    while(len(caption_word_id) < max_length):
    #    i = i + 1
        for j in range(len(temp_seq)):
           #print("Findind next three words for the caption =", temp_seq[j][0])
           output, hidden = decoder_model.get_bs_pred(torch.tensor([temp_seq[j][0][-1]]).cuda(), to_device(temp_seq[j][-1], device))
           output = F.log_softmax(output, dim=1)
           prob, predicted_ids = torch.topk(output, beam_width)
           prob = prob.cpu().detach().numpy().squeeze(0)
           predicted_ids = predicted_ids.cpu().detach().numpy().squeeze(0)
           for ids in range(len(predicted_ids)):
               next_seq.append([[temp_seq[j][0], [predicted_ids[ids]]], (temp_seq[j][1] + prob[ids]), hidden])
               # Tried Length Normalization but it performs worse
               #next_seq.append([[temp_seq[j][0], [predicted_ids[ids]]], ((temp_seq[j][1] + prob[ids])/np.power((len(caption_word_id)),0.2)), hidden])
        
        for seq_ in next_seq:
            seq_[0] = [item for sublist in seq_[0] for item in sublist]
            temp_seq.append(seq_)
        
        next_seq = []
        temp_seq = temp_seq[beam_width:]
        temp_seq = sorted(temp_seq, reverse=True, key=lambda l: l[1])
        temp_seq = temp_seq[:beam_width]
        caption_word_id = temp_seq[0][0]
        #if (i == 9): break
        if (caption_word_id[-1] == end_token):
            break
    return caption_word_id      
        
