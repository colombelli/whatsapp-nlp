import re
import pandas as pd
import numpy as np
import random

class DataProcessing:

    """
    Data processing for whatsapp imported messages.
    It operates aiming to converte all messages into a single string containing the contact sender
    name followed by a ':' and the message. Everything is separated by a ' ' which is used as word
    delimiter separator to build the vocabulary.

    Args:
        file_path (str): path containing the whatsapp imported messages file
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.history = None
        self.all_messages = None

        self.vocabulary = None
        self.word2idx = None
        self.idx2word = None
        self.vector_data_idx = []
        self.vector_data_word = None

        self.__read_history()
        self.__build_all_messages_string()
        self.__generate_vocabulary()
        self.__vectorize_words()


    def __read_history(self):
        f = open(self.file_path, 'r')
        
        # Every text message has the same format: date - sender: message. 
        messages = re.findall(r'(\d+/\d+/\d+, \d+:\d+\d+ [A-Z]*) - (.*?): (.*)', f.read())
        f.close()

        # Convert list to a dataframe and name the columns
        history = pd.DataFrame(messages,columns=['date','name','msg'])
        history['date'] = pd.to_datetime(history['date'],format="%m/%d/%y, %I:%M %p")
        history['date1'] = history['date'].apply(lambda x: x.date())

        # file_path is in the format 'WhatsApp Conversation with XXX.txt'
        history['conv_name'] = self.file_path[19:-4]

        self.history = history
        return



    # Messages format will look like
    # PersonA: hallo message PersonB: yo! PersonA: how are you?
    def __build_all_messages_string(self):

        history = self.history.loc[:, ['name', 'msg']]

        all_messages = ""
        for _, row in history.iterrows():
            name = row['name']
            msg = row['msg']
            all_messages += name + ": " + msg + " "
        

        # Eliminates <PersonA omitted> messages
        names = list(self.history['name'].unique())
        for name in names:
            all_messages = all_messages.replace("<"+name+" omitted>", '')

        # Eliminates 'This message was deleted' messages
        all_messages = all_messages.replace("This message was deleted", '')

        self.all_messages = all_messages
        return


    def __generate_vocabulary(self):

        self.vector_data_word = self.all_messages.split(" ")
        self.vocabulary = sorted(set(self.vector_data_word))
        self.word2idx = {u:i for i, u in enumerate(self.vocabulary)}
        self.idx2word = np.array(self.vocabulary)
        return


    def __vectorize_words(self):
        for word in self.vector_data_word:
            self.vector_data_idx.append(self.word2idx[word])
        return
        