# whatsapp-nlp
LSTM model for WhatsApp natural language processing given exported conversation .txt file.

## Summary
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage Guide](#usage-guide)

## Introduction

The proposal of this repository, other than studying LSTMs for Natural Language Processing, is to provide a simple automatic solution for training and generating text messages based on a WhatsApp chat conversation history. It also works for chat groups with more than two people, but it does not work properly if there are any unknown contacts within the conversation.

There's also an interesting aspect to it regarding the investigation of people's interaction patterns, for example, who talks more about what, or how they behave in general.  

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
