# whatsapp-nlp
![](https://img.shields.io/badge/python-v3.6-blue)

LSTM model for WhatsApp natural language processing given exported conversation .txt file.

## Summary
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage Guide](#usage-guide)

## Introduction

The proposal of this repository, other than studying LSTMs for Natural Language Processing, is to provide a simple automatic solution for training a model and generating text messages based on WhatsApp chats conversation history. It also works for chat groups with more than two people.

There's an interesting aspect to this process regarding the investigation of people's interaction patterns. Examples of this could be who talks more about what, or how they behave in general while talking in that chat. However, there's no guarantee, of course, that the observed learned chatting pattern is actually related to that person or if it's something that usually comes up in the chat. This is due to the fact that a person is interpreted as just another value in the string sequence. 

## Installation

The current code was tested for Python 3.6 and Tensorflow 2.x.

Cloning this repository:
    
    $ git clone https://github.com/colombelli/whatsapp-nlp.git

Installing the necessary python packages:
    
    $ pip install numpy
    $ pip install pandas
    $ pip install tensorflow
    $ pip install tqdm

## Usage Guide

The first thing to do is exporting an WhatsApp chat history: 
1. Open the individual or group chat; 
2. Tap More Options (the three vertical dots) > More > Export chat; 
3. Choose to export the chat without media.

[ref: https://faq.whatsapp.com/android/chats/how-to-save-your-chat-history/?lang=en]

Going back to the whatsapp-nlp package, edit the file ./src/run.py with your input parameters:
* ```file```: the path to the exported chat file
* ```num_training_iterations```: an integer representing the amount of epochs of the learning process
* ```batch_size```: an integer representing the number of observations seen at once before optimizing the loss function (anything whithin 1 and 64 should be fine)
* ```seq_length```: an integer representing how many words should a chat training example have (counting with the name of who sent that message)
* ```learning_rate```: a float in which the bigger the value, the quicker the convergence, but the model could pass-by better local minima
* ```rnn_units```: an integer representing the number of neurons for the LSTM layer
* ```dropout```: a float between 0 and 1 representing the fraction of the units to drop for the linear transformation of the inputs
* ```recurrent_dropout```: a float between 0 and 1 representing the fraction of the units to drop for the linear transformation of the recurrent state
* ```embedding_dim```: an integer representing the embedding dimension to be used

After editing these parameters, go back to de terminal, change your directory to /whatsapp-nlp/src/ and execute the run.py script:
    
    $ python run.py

The script will ask for further information during the execution process.
