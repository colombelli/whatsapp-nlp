import os
import DataProcessing
import Model

### Hyperparameter setting and optimization ###

# Optimization parameters:
num_training_iterations = 8000  # Increase this to train longer
batch_size = 4  # Experiment between 1 and 64
seq_length = 500  
learning_rate = 5e-3
rnn_units = 600

# Checkpoint location: 
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "my_ckpt")

