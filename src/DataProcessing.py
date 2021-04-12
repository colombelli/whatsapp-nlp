import re
import pandas as pd
import numpy as np
import random

class DataProcessing:


    """
    Data processing for whatsapp imported messages.

    It operates aiming to converte all messages into a single string containing the contact sender
    name followed by a ':' and the message. Everything is separated by a ' ' which is used as word
    delimiter separator to build the vocabulary and related objects.

    The class also provides methods to get all the possible starting index of the training data, as
    well as X and y batch data using these possible starting indexes.

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

        # Format composed names to use an underline '_' instead of blank space ' '
        formatted_names = [name.replace(' ', '_') for name in history['name']]
        history['name'] = formatted_names

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
        

        # Eliminates <Media omitted> messages
        all_messages = all_messages.replace("<Media omitted>", '')

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

        self.vector_data_idx = np.array(self.vector_data_idx)
        return
    

    # Returns the indexes in the message sequence that can initialize a conversation, in
    # other words, the points in the sequence where the word is something like 'PersonA:'
    def get_possible_starts(self, seq_len):

        n = self.vector_data_idx.shape[0] - 1  # The length of the vectorized data
        th = n - seq_len

        names = list(self.history['name'].unique())
        person_msg_idx = np.concatenate(
            [np.squeeze(np.where(self.vector_data_idx == self.word2idx[name+':'])) for name in names]
        )
        person_msg_idx = np.squeeze(person_msg_idx)

        possible_starts = []
        for idx in person_msg_idx:
            if idx < th:
                possible_starts.append(idx)
        
        return possible_starts


    # Returns X and y data in batch format, ready to train
    def get_batch(self, possible_starts, seq_len, batch_size):

        # randomly choose the starting indices for the examples in the training batch
        idx = np.random.choice(possible_starts, batch_size)

        input_batch = [self.vector_data_idx[i:i+seq_len] for i in idx]
        output_batch = [self.vector_data_idx[i+1:i+1+seq_len] for i in idx]

        # x_batch, y_batch provide the true inputs and targets for network training
        x_batch = np.reshape(input_batch, [batch_size, seq_len])
        y_batch = np.reshape(output_batch, [batch_size, seq_len])
        return x_batch, y_batch

    