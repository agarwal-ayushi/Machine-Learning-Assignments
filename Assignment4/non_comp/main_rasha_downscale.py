from __future__ import print_function
#import argparse
import os
import torch
import numpy as np
import torchvision

import numpy as np
# Does this same thing as argparse
from absl import flags,app,logging
import time
from train_rasha_downscale import train_image_caption_rasha
from test_rashadownscale import test_rasha


FLAGS=flags.FLAGS

flags.DEFINE_integer(name='gpu', default=0, help='GPU Index')
flags.DEFINE_integer(name='embed_size', default=256, help='Embedding Size')
flags.DEFINE_integer(name='hidden_dim', default=256, help='Hidden Dimensions')
flags.DEFINE_integer(name='num_layers', default=1, help='Number of Layers in LSTM')
flags.DEFINE_integer(name='batch_size', default=8, help='Batch Size')
flags.DEFINE_integer(name='epochs', default=10, help='Number of Epochs')
flags.DEFINE_float(name='lr', default=0.01, help='Learning Rate')
flags.DEFINE_bool(name='train', default= True, help='Whether to train the network from scratch or not')
flags.DEFINE_bool(name='train_enc', default= False, help='Whether to train the encoder from scratch or not')

#Ayushi adding flag for training VGG from scratch
flags.DEFINE_bool(name='vgg19', default= True, help='Use VGG-19 or not for scratch training')

#Ramya
flags.DEFINE_bool(name='Resnet50', default= False, help='Use Resnet50 or not')

#Ramya
flags.DEFINE_bool(name='Resnet152', default= False, help='Use Resnet152 or not')




def main(argvs):
    vocab_len, vocab_dict = train_image_caption_rasha(FLAGS.embed_size, FLAGS.hidden_dim, FLAGS.batch_size, FLAGS.epochs, FLAGS.lr, FLAGS.num_layers, FLAGS.train, FLAGS.train_enc, FLAGS.Resnet50, FLAGS.Resnet152, FLAGS.vgg19)
    test_rasha(FLAGS.embed_size, FLAGS.hidden_dim, FLAGS.num_layers,  vocab_len, vocab_dict, FLAGS.Resnet50, FLAGS.Resnet152, FLAGS.vgg19)

if __name__=='__main__':
    app.run(main)
    
