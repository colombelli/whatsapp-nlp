import tensorflow as tf 
from tqdm import tqdm
from DataProcessing import DataProcessing


# Based on MIT's introduction to Deep Learning course
class Model:


    """
    LSTM model for learning from the data and generating a conversation.

    The model recognizes all different possible words and map them to a number,
    which will serve as a value in the sequence of messages. It stacks an LSTM
    layer with a Dense layer working as the output for each next word given a
    sequence of words.

    Args:
        rnn_units (int): number of neurons in the LSTM
        dropout (float): between 0 and 1, representing the fraction of the units 
                        to drop for the linear transformation of the inputs.
        recurrent_dropout (float): between 0 and 1, representing the fraction of 
                        the units to drop for the linear transformation of the 
                        recurrent state.
        learning_rate (float)
        batch_size (int)
        num_training_iterations (int): number of epochs to train
        seq_length (int): the size of the word sequences for each training example
        checkpoint_prefix (str): the file name of the saved checkpoints
        checkpoint_dir (str): the directory where the checkpoints are to be saved
        embedding_dim (int): the embedding dimesion to encode the words to
        
    """


    def __init__(self, rnn_units:int, dropout:float, recurrent_dropout:float, 
                learning_rate:float, batch_size:int, num_training_iterations:int, 
                seq_length:int, checkpoint_prefix:str, checkpoint_dir:str,
                embedding_dim:int, data_processing:DataProcessing): 

        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.rnn_units = rnn_units

        self.optimizer = tf.optimizers.Adam(learning_rate)
        self.batch_size = batch_size
        self.num_training_iterations = num_training_iterations
        self.seq_length = seq_length
        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_dir = checkpoint_dir
        self.embedding_dim = embedding_dim
        self.data_processing = data_processing
        self.possible_starts = data_processing.get_possible_starts(seq_length)
        self.model = self.__build_model()


    # Defining the RNN Model
    def __build_model(self, batch_size=None):
        vocab_size = len(self.data_processing.vocabulary)

        if not batch_size:
            batch_size = self.batch_size

        lstm_layer = tf.keras.layers.LSTM(
                                self.rnn_units, 
                                return_sequences=True, 
                                recurrent_initializer='glorot_uniform',
                                recurrent_activation='sigmoid',
                                stateful=True,
                                dropout=self.dropout, 
                                recurrent_dropout=self.recurrent_dropout
                            )

        model = tf.keras.Sequential([
            # Layer 1: Embedding layer to transform indexes into dense vectors 
            #   of a fixed embedding size
            tf.keras.layers.Embedding(vocab_size, self.embedding_dim, batch_input_shape=[batch_size, None]),

            # Layer 2: LSTM with `rnn_units` number of units. 
            lstm_layer,

            # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
            #   into the vocabulary size.
            tf.keras.layers.Dense(vocab_size)
        ])
        return model


    # Defining the loss function
    def compute_loss(self, labels, logits):
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        return loss


    @tf.function
    def train_step(self, x, y): 
        with tf.GradientTape() as tape:
            y_hat = self.model(x)
            loss = self.compute_loss(y, y_hat)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss



    def train_model(self):

        tr_history = []
        if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists


        for iter in tqdm(range(self.num_training_iterations)):
            # Grab a batch and propagate it through the network
            x_batch, y_batch = self.data_processing.get_batch(  self.possible_starts, 
                                                                self.seq_length, 
                                                                self.batch_size)
            loss = self.train_step(x_batch, y_batch)

            # Update the progress bar
            tr_history.append(loss.numpy().mean())

            # Update the model with the changed weights!
            if iter % 100 == 0:     
                self.model.save_weights(self.checkpoint_prefix)
            
        # Save the trained model and the weights
        self.model.save_weights(self.checkpoint_prefix)


    
    def generate_text(self, start_word, generation_length=1000):

        model = self.__build_model(batch_size=1)

        # Restore the model weights for the last checkpoint after training
        model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        model.build(tf.TensorShape([1, None]))        

        input_eval = [self.data_processing.word2idx[start_word]]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Here batch size == 1
        model.reset_states()
        tqdm._instances.clear()

        for _ in tqdm(range(generation_length)):
            
            predictions = model(input_eval)
            
            # Remove the batch dimension
            predictions = tf.squeeze(predictions, 0)
            
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
            
            # Pass the prediction along with the previous hidden state
            #   as the next inputs to the model
            input_eval = tf.expand_dims([predicted_id], 0)
            
            text_generated.append(" "+self.data_processing.idx2word[predicted_id])
            
        return (start_word + ''.join(text_generated))