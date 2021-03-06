from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from scripts.RNN_utils import *



# TODO create class if needed.
def lstm(data_dir = "data/poems_test_small.txt", batch_size = 100, hidden_dim = 500, seq_length = 100, weights = "", mode = "train",
         dropout_rate = 0.2, generate_length = 500, total_epochs = 10, gen_samples = 5, layer_num = 2, save_every = 1, 
         use_subwords = False, poem_end = "\n\n", temp = 1, end_symbol = "$"):

    print("weights:", weights)
    print("mode:", mode)
    print("epochs:", total_epochs)
    print("dropout_rate:", dropout_rate)
    print("layers:", layer_num)
    print("hidden dim:", hidden_dim)
    print("Sequence length:", seq_length)  # If seq_length == -1 use poems
    print("Temperature:", temp)
    
    log_path = ""

    # Load data and vocabulary
    if seq_length != -1:
        VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch, data = load_vocabulary(data_dir, seq_length, batch_size, use_subwords)
    else:
        VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch, data = load_vocabulary_poem(data_dir, batch_size, poem_end, use_subwords, end_symbol)

    # Creating and compiling the Network
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    model.add(Dropout(rate = dropout_rate))
    for i in range(layer_num - 1):
      model.add(LSTM(hidden_dim, return_sequences=True))
      model.add(Dropout(rate = dropout_rate))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # Load model if file for weights is given
    if not weights == '':
      model.load_weights(log_path + weights)
      epoch = int(weights[weights.rfind('_') + 1:weights.rfind('.')])
    else:
      epoch = 0

    # Training
    if mode == 'train':
      while epoch <= total_epochs:
      
        print('\n\nEpoch: {}\n'.format(epoch))
        
        if seq_length != -1:   # Generate sequence by sequence
            data_gen = data_generator(data, seq_length, batch_size, steps_per_epoch)
        else:   # Generate poem by poem
            data_gen = data_generator_poem(data, batch_size, poem_end, use_subwords, end_symbol)
        model.fit_generator(data_gen, \
        steps_per_epoch=steps_per_epoch, verbose = 1, epochs = 1)
        epoch += 1        
        
        if epoch % save_every == 0:
          print("Save weights")
          model.save_weights(log_path + 'checkpoint_layer_{}_hidden_{}_dropout_{}_epoch_{}.hdf5'.format(layer_num, hidden_dim, dropout_rate, epoch))
          print(generate_text(model, generate_length, VOCAB_SIZE, ix_to_char, use_subwords, temp, end_symbol))
          
      # Generate samples
      print("Generating %i sample poems." % gen_samples)
      for i in range(gen_samples):
        print(i, "\n")
        print(generate_text(model, generate_length, VOCAB_SIZE, ix_to_char, use_subwords, temp, end_symbol))


    # Else, loading the trained weights and performing generation only
    elif weights != '':
      # Loading the trained weights
      #model.load_weights(weights)
      print("Translating")
      for i in range(gen_samples):
        print(generate_text(model, generate_length, VOCAB_SIZE, ix_to_char, use_subwords, temp, end_symbol))
        print('\n\n')
    else:
      print('\n\nNothing to do!')