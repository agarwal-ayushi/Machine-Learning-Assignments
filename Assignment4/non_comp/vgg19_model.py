import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG19(nn.Module):
    def __init__(self, embed_size):
        super(VGG19, self).__init__()
        self.embed_size = embed_size
        #print(embed_size)
        self.model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(128, 128, kernel_size=2, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(256, 256, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.Conv2d(512, 512, kernel_size=3, padding=1),
                                   nn.ReLU(inplace = True),
                                   nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(                  
           nn.Linear(in_features=512*3*3, out_features=4096),
           nn.ReLU(inplace=True),
           nn.Dropout(p=0.5, inplace=False),
           nn.Linear(in_features=4096, out_features=4096),
           nn.ReLU(inplace=True),
           nn.Linear(in_features=4096, out_features=self.embed_size),
           nn.ReLU(inplace=True),
             )
        
    def forward(self, feat):
        #print(feat.shape)
        features = self.model(feat)
        #print(features.shape)
        features = features.view(features.shape[0],-1)
        #print(features.shape)
        features = self.classifier(features)
        #print(features.shape)
        return features    
