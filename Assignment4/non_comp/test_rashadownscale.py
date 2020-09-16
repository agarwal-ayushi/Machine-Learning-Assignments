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
from utils_rasha_downscale import get_test_data, get_private_test_data
from beam_search_pred_downscale import beam_search_pred
import os
from max_predictions_downscale import max_prediction

#Ramya
from vgg19_model import VGG19
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
path ='.'
import sacrebleu

def process_captions(data):
    '''
    Function to clean the testing captions ( Remove <start> and <end> )
    '''
    process_seq = []
    whole_seq = []
    for i in data:
        seq= i[0]
        seq = seq.strip().split(' ')[1:-1]
        process_seq.append(seq)
        x = " ".join(seq)
        whole_seq.append(x)
    return process_seq, whole_seq
        
def calculate_bleu(ref, hypo):
    ''' Calculates bleu score
    Input :
        ref : Reference captions (given in test file)
        hypo : Captions generated by model
    '''
    #for i in hypo:
        #w= (0.25,0.25,0.25,0.25)
    w=(1,0,0,0)
         
    score = sentence_bleu(ref,hypo,w)
    return score

def convert_rasha(out, ix_2_word):
    '''
    Converts tokenized outputs into words
    '''
    s_=[]
    string_out_=[]
    for i in out:
        seq = np.ravel(np.array(i))
        seq = [j.item() for j in seq]
        string_out = [ix_2_word[j] for j  in seq[1:]]
        s =" ".join(string_out)  
        s_.append(s); string_out_.append(string_out)
    return s_, string_out_
        
def convert_rasha_max_pred(out, ix_2_word):
    '''
    Converts max prediction to words
    '''
    caption=[]
    out = out[1:-1]
    for word_id in out:
        word = ix_2_word[word_id]
        caption.append(word)
    return caption

f = open("public_captions.tsv", 'w')    
private = open("private_captions.tsv", 'w')
# Ramya : added ResNet50, ResNet152
def test_rasha(embed_dim, hidden_dim,num_layer, vocab_len, vocab_dict, ResNet50, ResNet152, vgg19):

    word_2_ix, ix_2_word = vocab_dict
    #Added flags ResNet50, ResNet152
    encoder_model = Encoder_Rasha(embed_dim, ResNet50, ResNet152, vgg19)
    decoder_model = Decoder_Rasha(embed_dim, hidden_dim, vocab_len, num_layer)
    try :
        checkpoint = torch.load("curr_model.pk")
        encoder_model.load_state_dict(checkpoint["encoder_state_dict"])  ##Load the models
        decoder_model.load_state_dict(checkpoint["decoder_state_dict"])
    except Exception as e:
        print(e)
        
    encoder_model = encoder_model.cuda()
    encoder_model.eval() # Makes the model ready to be used in evaluation by taking care of batch norm and dropout
    decoder_model = decoder_model.cuda()
    decoder_model.eval()
    
    test_dl = get_test_data()
    private_test_dl = get_private_test_data()
    bleu_score=[]
    hypo_complete = []
    ref = [[],[],[],[],[]]

    for batchid, (images, img_id) in enumerate(private_test_dl):
        images = images.cuda()
        encode_feat_rasha = encoder_model(images.float())
        output_bs_pred =  beam_search_pred(encode_feat_rasha, decoder_model, vocab_dict, 5)
        hypo = convert_rasha_max_pred(output_bs_pred, ix_2_word)
        private.write(f'{(img_id[0])}\t {" ".join(hypo)}\n') 
       
    for batchid, (images, captions, img_id) in enumerate(test_dl):
        bl_captions_, captions_ = process_captions(captions)
        images = images.cuda()
        encode_feat_rasha = encoder_model(images.float())
        output_bs = beam_search_pred(encode_feat_rasha, decoder_model, vocab_dict, 5)
        #print("Encoded features shape during testing", encode_feat_rasha.shape)
        #output_max_pred = max_prediction(encode_feat_rasha, decoder_model, vocab_dict)
        #print("the output shape is", output_max_pred.shape)
        hypo =  convert_rasha_max_pred(output_bs, ix_2_word)
        score = calculate_bleu(bl_captions_, hypo)
        
        f.write(f'{(img_id[0])} \t {" ".join(hypo)}\n') 
        bleu_score.append(score)
        
        hypo_complete.append(" ".join(hypo))
        for j in range(len(captions_)):
            ref[j].append(captions_[j])
        #ref.append(captions_)

    sacre_bleu = sacrebleu.corpus_bleu(hypo_complete, ref).score
    print(f"BLEU SCORE on public test data by beam search : {np.mean(np.array(bleu_score))}")
    print(f"SACREBLEU SCORE by beam search : {sacre_bleu}")
    return np.mean(np.array(bleu_score))
        
#  
#import pickle
#with open('vocab_dict_rasha.pickle', 'rb') as handle:
#    b = pickle.load(handle)
#        
#test_dl = test_rasha(256, 256, 1 , 34113, b, ResNet50=False, ResNet152=False, vgg19=True)
