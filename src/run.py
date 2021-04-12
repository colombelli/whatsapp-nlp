import os, sys
from DataProcessing import DataProcessing
from Model import Model


################# INPUT PARAMETERS #################
num_training_iterations = 10#500  # Increase this to train longer
batch_size = 16  # Experiment between 1 and 64
seq_length = 100
learning_rate = 5e-3
rnn_units = 100
dropout=0.3
recurrent_dropout=0.3
embedding_dim = 10
file = "/home/colombelli/Documents/datasets/text/skin.txt"



# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")





def format_generated_text(generated_text, possible_names):
    formatted_text = ""
    sp_txt = generated_text.split(" ")
    for word in sp_txt:
        if (word[:-1] in possible_names):
            formatted_text += "\n"+word+" "
        else:
            formatted_text += word+" "
    return formatted_text


def format_person_input_name(name):
    return name.replace(" ", "_")  + ":"


def console_interaction(model):

    while True:
        do_training = input("\n\nDo you want to train a new model? (y/n) ")

        if do_training == 'y':
            model.train_model()
        elif do_training != 'n':
            print("Choose either 'y' or 'n'\n")
            continue
        

        while True:
            do_text_gen = input("\n\nDo you want to generate text based on the trained model? (y/n) ")

            if do_text_gen == 'n':
                print("Ok, have a nice day! :)")
                sys.exit()
            elif do_text_gen != 'y':
                print("Choose either 'y' or 'n'\n")
                continue

            

            print("\nOk! To generate text, please, enter a contact name (or your whatsapp name)" +
                  " exaclty as it shows on your phone. This person will start the conversation.")

            possible_names = list(model.data_processing.history['name'].unique())
            string_possible_names = ""
            for name in possible_names:
                string_possible_names += name + ", "
            string_possible_names = string_possible_names[:-2]
            print("These are the available options:\n", string_possible_names)

            person_name = input("\nWho's the one to start chatting? ")
            person_name = format_person_input_name(person_name)
            if not (person_name[:-1] in possible_names):
                print("Invalid contact name! Can't generate conversation.")
                continue
            
            try:
                number_of_words = int(input("How many words do you want to generate? "))
            except:
                print("Invalid input. Assuming 1000...")
                number_of_words=1000

            print("\nGenerating Conversation...")

            generated_text = model.generate_text(person_name, number_of_words)
            formated_text = format_generated_text(generated_text, possible_names)

            print("\nSuccess! \n\nGenerated conversation:\n\n", formated_text)
            



if __name__ == '__main__':
    
    dp = DataProcessing(file)
    model = Model(rnn_units, dropout, recurrent_dropout, learning_rate, 
                batch_size, num_training_iterations, seq_length, 
                checkpoint_prefix, checkpoint_dir, embedding_dim, dp)

    console_interaction(model)    