import re
import pandas as pd
import numpy as np


class DataProcessing:

    def __init__(self, file_path):
        self.file_path = file_path
        self.history = None
        self.df = None

        self.__read_history()
        self.__build_dataframe()


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


    def __build_dataframe(self):
        return
