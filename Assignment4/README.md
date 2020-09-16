# Image-Captioning Machine Learning Model in PyTorch

#### The assignment problem statement: Assignmemt4.pdf
#### Report Submitted: Assignment4_report.pdf


##### The assignment was broken into two parts: Non-Competetive and Competitive part.

**Non-Competetive Part:** Encoder - VGG-19 trained from scratch on the dataset given (subset of ImageNet) and Decoder - LSTM (PyTorch based)
**Competitive Part:** Encoder - Pretrained torchvision ResNet-50 model, Decoder - LSTM and Attention based 


Note: Every folder contains the public and private data captions generated with the model and contains an out.txt file which reports the last Bleu Score tested on the checkpoints. 

We have a loss_plot.png file to display the loss curve we obtained during the entire training. The loss curve was fairly similar for all the architectures that we tested. 


**Non-Competitive Part:**

Directory : non_comp/

Contains the code to run VGG-19 from scratch and LSTM from scratch. 

RESNET-50 takes a lot of time to train and hence did not give comparable results with VGG-19.

**Competitive Part:**

1) Directory: comp/code_lstm/

Contains the code to run LSTM decoder architecture with pretrained RESNET-50 model

2) Directory: comp/code_attention/

Contains the code to run Attention Decoder Architecture with pretrained RESNET-50 model. 

The Attention Architecture used in the code is inspired by the paper: Show, Attend and Tell and its implemnetation in PyTorch
