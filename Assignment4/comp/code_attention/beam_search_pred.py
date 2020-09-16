import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [to_device(x,device) for x in data]
        return data.to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beam_search_pred(encoded_features, decoder_model, vocab_dict, beam_width = 10, max_length=20):
    '''
    Function to perform beam search to generate captions of the images
    Beam search is an algorithm that finds the best n predictions at 
    each step and uses those to find the next n predictions
    '''
    word_2_ix , ix_2_word = vocab_dict
    start_token = word_2_ix['<start>']
    #print(start_token)
    end_token = word_2_ix['<end>']
    h = decoder_model.act(decoder_model.h(encoded_features.mean(dim=1)))
    c = decoder_model.act(decoder_model.c(encoded_features.mean(dim=1)))
    # In the beginning the hidden state has to be None, since we are entering into the LSTM
    hidden = None
    caption_word_id = []
    temp_seq = []
    next_seq = []
    cap = torch.tensor([start_token], device = 'cuda')
    encoded_features = encoded_features.cuda()
    # Generating the first set of three best tokens for the input features
    output, hidden = decoder_model.get_bs_pred(encoded_features, cap, (h,c))
    # Instead of maximizing product of probablities (can underflow for long captions), we maximize sum of log probs
    output = F.log_softmax(output, dim=1)
    prob , predicted_ids = torch.topk(output, beam_width)
    prob = prob.cpu().detach().numpy().squeeze(0)
    predicted_ids = predicted_ids.cpu().detach().numpy().squeeze(0)
    for ids in range(len(predicted_ids)):
        temp_seq.append([[predicted_ids[ids]], prob[ids], hidden])
    i = 0
    while(len(caption_word_id) < max_length):
        i = i + 1
        temp = [] 
        for j in range(len(temp_seq)):
           output, hidden = decoder_model.get_bs_pred(encoded_features, torch.tensor([temp_seq[j][0][-1]]).cuda(), to_device(temp_seq[j][-1], device))
           #print(output)
           output = F.log_softmax(output, dim=1)
           prob, predicted_ids = torch.topk(output, beam_width)
           prob = prob.cpu().detach().numpy().squeeze(0)
           predicted_ids = predicted_ids.cpu().detach().numpy().squeeze(0)
           for ids in range(len(predicted_ids)):
               next_seq.append([[temp_seq[j][0], [predicted_ids[ids]]], (temp_seq[j][1] + prob[ids]), hidden])
              
        for seq_ in next_seq:
            seq_[0] = [item for sublist in seq_[0] for item in sublist]
            temp_seq.append(seq_)
        next_seq = []
        temp_seq = temp_seq[beam_width:]
        temp_seq = sorted(temp_seq, reverse=True, key=lambda l: l[1])
        temp_seq = temp_seq[:beam_width]
        caption_word_id = temp_seq[0][0]
       # for i,j in enumerate(temp_seq):
       #     print(temp_seq[i][0])
        
      #  if (i == 5): break
        if (caption_word_id[-1] == end_token):
            break
    return caption_word_id 
    
           
        
