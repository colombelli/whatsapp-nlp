import re
import pandas as pd
import numpy as np


class DataProcessing:

    """
    Data processing for whatsapp imported messages.
    It operates aiming to converte all messages into a single string containing the contact sender
    name followed by a ':' and the message. Everything is separated by a ' ' which is used as word
    delimiter separator.

    Args:
        file_path (str): path containing the whatsapp imported messages file
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.history = None
        self.all_messages = None

        self.__read_history()
        self.__build_all_messages_string()


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
        for index, row in history.iterrows():
            name = row['name']
            msg = row['msg']
            all_messages += name + ": " + msg + " "
        
        self.all_messages = all_messages
        return
