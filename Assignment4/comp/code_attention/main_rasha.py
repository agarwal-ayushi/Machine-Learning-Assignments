
from __future__ import print_function
#import argparse
import os
import torch
import numpy as np
import torchvision

import numpy as np
from absl import flags,app,logging
import time
from train_attention_rasha import train_image_caption_rasha
from test_rasha import test_rasha

FLAGS=flags.FLAGS

flags.DEFINE_integer(name='gpu', default=0, help='GPU Index')
flags.DEFINE_integer(name='embed_size', default=512, help='Embedding Size')
flags.DEFINE_integer(name='hidden_dim', default=512, help='Hidden Dimensions')
flags.DEFINE_integer(name='num_layers', default=1, help='Number of Layers in LSTM')
flags.DEFINE_integer(name='batch_size', default=8, help='Batch Size')
flags.DEFINE_integer(name='epochs', default=10, help='Number of Epochs')
flags.DEFINE_float(name='lr', default=4e-4, help='Learning Rate')
flags.DEFINE_bool(name='train', default= True, help='Whether to train the network from scratch or not')
flags.DEFINE_bool(name='train_enc', default= False, help='Whether to train the encoder from scratch or not')
flags.DEFINE_integer(name='bw', default=5, help='Beam width for beam search algorithm')


def main(argvs):
    vocab_len, vocab_dict = train_image_caption_rasha(FLAGS.embed_size, FLAGS.hidden_dim, FLAGS.batch_size, FLAGS.epochs, FLAGS.lr, FLAGS.num_layers, FLAGS.train, FLAGS.train_enc)
    test_rasha(FLAGS.embed_size, FLAGS.hidden_dim, FLAGS.num_layers, FLAGS.bw, vocab_len, vocab_dict, FLAGS.train_enc)

if __name__=='__main__':
    app.run(main)
    
