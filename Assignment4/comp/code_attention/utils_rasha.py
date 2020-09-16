import torch
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from skimage import io, transform
import string

import matplotlib.pyplot as plt # for plotting
import numpy as np
from PIL import Image
import pickle
import os

'''Paths'''
IMAGE_DIR = './train_images/'
CAPTIONS_FILE_PATH = './train_captions.tsv'
IMAGE_DIR_TEST = './public_test_images/'
IMAGE_DIR_PRIVATE_TEST = './private_test_images/'
CAPTIONS_FILE_PATH_TEST = './public_test_captions.tsv'
# We changed the size of the image resizing from 256x256 as provided to 224x224 because we wanted to compare
# our model with a pretrained model and the pretrained model was giving error with 256x256 since it would 
# have been trained on 224x224
IMAGE_RESIZE = (224, 224)

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        return img


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
#         if (idx == 0):
#             print("Final Image shape= ", image.shape)
        return torch.tensor(image)


class CaptionsPreprocessing():
    """Preprocess the captions, generate vocabulary and convert words to tensor tokens

    Args:
        captions_file_path (string): captions tsv file path
    """
    def __init__(self, captions_file_path):

        self.captions_file_path = captions_file_path

        # Read raw captions
        self.raw_captions_dict = self.read_raw_captions()

        # Preprocess captions
        self.captions_dict = self.process_captions()

        # Create vocabulary
        self.vocab = self.generate_vocabulary()
        
        # Max length of the captions in the training set
        self.max_length = self.gen_max_length()

    def read_raw_captions(self):
        """
        Returns:
            Dictionary with raw captions list keyed by image ids (integers)
        """

        captions_dict = {}
        with open(self.captions_file_path, 'r', encoding='utf-8') as f:
            for img_caption_line in f.readlines():
                img_captions = img_caption_line.strip().split('\t')
                captions_dict[int(img_captions[0])] = img_captions[1:]
        
        print('Number of Loaded captions: %d ' % len(captions_dict))

        return captions_dict

    def process_captions(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """
        raw_captions_dict = self.raw_captions_dict

        # Do the preprocessing here
        captions_dict = raw_captions_dict
        
        print("---Beginning the pre-processing of the training dataset captions---")
        
        # Pre-processing of the captions by removing the punctuations, digits, all words with numbers since they won't contribute to captions
        for idx, all_caption in captions_dict.items():
            for i in range(len(all_caption)):
                #caption = '' + all_caption[i].translate(str.maketrans('','',string.punctuation)).lower() + ''
                #caption = ' '.join([w for w in caption.split() if w.isalpha()]) #Removes words which are numbers
                # Add <start> and <end> to the caption as and when they are processed for each image
                all_caption[i] = '<start> '+''.join(all_caption[i])+' <end>'
                
        return captions_dict
    
    def gen_max_length(self):
        """
        Use this function to return the maximum possible length of the caption
        present in the training dataset
        This is needed to generate tensors of equal sizes
        """
        max_length = 0
        captions_dict = self.captions_dict
        
        for idx, all_caption in captions_dict.items():
            for caption in all_caption:
                if (max_length < len(caption.split())):
                    max_length = len(caption.split())
        #print("Maximum length of the captions in the dataset is = ", max_length) 
        return max_length
    
    def generate_vocabulary(self):
        """
        Use this function to generate dictionary and other preprocessing on captions
        """

        captions_dict = self.captions_dict
        """
        Generate the vocabulary
        We use python set() and use the update() method which does not add repeated words in the set.
        Since we only need words and not their frequency we dont use dictionary. 
        """
        temp = set() 
        for idx in captions_dict.keys():
            [temp.update(caption.split()) for caption in captions_dict[idx]]
        #temp.update({'<pad>'})
        print("The number of words in the generated vocabulary are=", len(temp))
        return temp

    def captions_transform(self, img_caption_list):
        """
        Use this function to generate tensor tokens for the text captions
        Args:
            img_caption_list: List of captions for a particular image
        """
        vocab = self.vocab
       
        # Enumerate all the words in the vocab so that they can be indexed directly
        word_to_ix = {word: i for i, word in enumerate(vocab)}
        
        img_caption_token = []
        
        # Generate tensors
        for caption in img_caption_list:
            # Generate a tensor for all the captions by using enumeration
            lookup_tensor = torch.tensor([word_to_ix[w] for w in caption.split()])
            img_caption_token.append(lookup_tensor)
        return img_caption_token
        #return torch.zeros((len(img_caption_list),10))

class ImageCaptionsDataset(Dataset):

    def __init__(self, img_dir, captions_dict, img_transform=None, captions_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            captions_dict: Dictionary with captions list keyed by image ids (integers)
            img_transform (callable, optional): Optional transform to be applied
                on the image sample.

            captions_transform: (callable, optional): Optional transform to be applied
                on the caption sample (list).
        """
        self.img_dir = img_dir
        self.captions_dict = captions_dict
        self.img_transform = img_transform
        self.captions_transform = captions_transform
        self.image_ids = list(captions_dict.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'image_{}.jpg'.format(self.image_ids[idx]))
        image = io.imread(img_name)
        captions = self.captions_dict[self.image_ids[idx]]

        if self.img_transform:
            image = self.img_transform(image)

        if self.captions_transform:
            captions = self.captions_transform(captions)
            return torch.tensor(image), captions
            #return torch.tensor(image), torch.tensor(captions)
        
        return image, captions, self.image_ids[idx]
        #sample = {'image': image, 'captions': captions}

        
def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

def to_device(data, device):
        if isinstance(data,(list,tuple)):
            return [to_device(x,device) for x in data]
        return data.to(device)

class Device_Loader():
        def __init__(self, dl ,device):
            self.dl= dl
            self.device=device
        
        def __iter__(self):
            for batch in self.dl:
                yield to_device(batch,self.device)
            
        def __len__(self):
            return len(self.dl)
'''
def collate_fn(data):
      
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths
'''


'''
Collate function is used to create custom batches in case the user does not want to use the automatic batching
We have used collate function to create 5 different data points from the same image and its 5 captions
Then number of batches effectively would now be
batches = (num_images*5)/batch_size
'''
def collate_fn(data):
    repeated_images=[]
    images, captions = zip(*data)
    captions = [cap_ for cap in captions for cap_ in cap]
    for image in images :
        repeated_images.append(image.repeat(5,1,1,1))
    images_ = torch.cat(repeated_images, 0)
    lengths = [len(cap) for cap in captions]
    # Change the length of all the captions in one batch to the max of all the captions. 
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end] 
    return images_, targets, lengths
 
def get_data(batch):
    '''
    Function to generate training data loader
    Input : Batch size
    Returns : Training data loader, Vocabulary length
    '''
    device = get_device()
    
    # Sequentially compose the transforms using Compose (chains all the transformations to be applied to all images)
    # We normalized the images to the custom normalization used by ImageNet dataset
    # Normalize increased performance slightly
    #img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor()])
    img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225)) ])
    
    captions_preprocessing_obj = CaptionsPreprocessing(CAPTIONS_FILE_PATH)

    # Regenerate the enumeration for the vocabulary words and store the opposite enumeration also
    word_to_ix = {word: i for i, word in enumerate(captions_preprocessing_obj.vocab)}
    ix_to_word = {i: word for i, word in enumerate(captions_preprocessing_obj.vocab)}
    vocab_dict = (word_to_ix, ix_to_word)
    with open('vocab_dict_rasha.pickle', 'wb') as handle:
        pickle.dump(vocab_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    vocab_len = len(captions_preprocessing_obj.vocab)

    # Loading the dataset
    train_dataset = ImageCaptionsDataset(
        IMAGE_DIR, captions_preprocessing_obj.captions_dict, img_transform=img_transform,
        captions_transform=captions_preprocessing_obj.captions_transform
        )
   # train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=2)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                              batch_size=batch,
                                              shuffle=True,
                                              num_workers=2,
                                              collate_fn=collate_fn)
    return train_loader, vocab_len, vocab_dict
   

    
def get_test_data():
    '''
    Function to generate training data loader
    Returns : Test data loader
    
    '''
    captions_test = CaptionsPreprocessing(CAPTIONS_FILE_PATH_TEST)
    img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
   
    test_dataset = ImageCaptionsDataset(
        IMAGE_DIR_TEST, captions_test.captions_dict, img_transform=img_transform,
        captions_transform=None
        )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle= False, num_workers=2)
    return test_loader
 
class ImagePrivateDataset(Dataset):

    def __init__(self, img_dir, img_ids, img_transform=None):
        """
        Args:
            img_dir (string): Directory with all the images.
            img_transform (callable, optional): Optional transform to be applied
                on the image sample.
        """
        self.img_dir = img_dir
        self.img_transform = img_transform
        self.image_ids = img_ids
        self.f = open('private_img_ids.txt', 'w')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, 'image_{}.jpg'.format(self.image_ids[idx]))
        image = io.imread(img_name)
        self.f.write(img_name)
        self.f.write("\n")
        if self.img_transform:
            image = self.img_transform(image)

        return image, self.image_ids[idx]

def get_private_test_data():
    '''
    Function to generate training data loader
    Returns : Test data loader
    
    '''
    img_ids = []
    for file in os.listdir(IMAGE_DIR_PRIVATE_TEST):
        file_= re.findall(r"[\w']+", file)[0].split('_')
        img_ids.append(file_[1])
    #captions_test = CaptionsPreprocessing(CAPTIONS_FILE_PATH_PRIVATE_TEST)
    img_transform = transforms.Compose([Rescale(IMAGE_RESIZE), ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
   
    private_test_dataset = ImagePrivateDataset(
        IMAGE_DIR_PRIVATE_TEST, img_ids, img_transform=img_transform,
        )
    private_test_loader = DataLoader(private_test_dataset, batch_size=1, shuffle= False, num_workers=0)
    return private_test_loader






