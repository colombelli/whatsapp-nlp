import os
from DataProcessing import DataProcessing
from Model import Model


def format_generated_text(generated_text, possible_names):
    formatted_text = ""
    sp_txt = generated_text.split(" ")
    for word in sp_txt:
        if (word[:-1] in possible_names):
            formatted_text += "\n"+word+" "
        else:
            formatted_text += word+" "
    return formatted_text


### Hyperparameter setting and optimization ###
# Optimization parameters:
num_training_iterations = 8000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 100
learning_rate = 5e-3
rnn_units = 600
dropout=0.3
recurrent_dropout=0.3
embedding_dim = 10

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")


file = "/home/colombelli/Documents/datasets/text/haddad.txt"
dp = DataProcessing(file)
model = Model(rnn_units, dropout, recurrent_dropout, learning_rate, 
              batch_size, num_training_iterations, seq_length, 
              checkpoint_prefix, checkpoint_dir, embedding_dim, dp)


model.train_model()
generated_text = model.generate_text('Felipe:')
possible_names = list(dp.history['name'].unique())
formated_text = format_generated_text(generated_text, possible_names)

print("\nSuccess! \n\nGenerated conversation:\n\n", formated_text)