from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from scripts.RNN_utils import *


# TODO create class if needed.
def lstm(data_dir = "../data/poems_test_small.txt", batch_size = 100, hidden_dim = 500, seq_length = 100, weights = "", mode = "train",
         dropout_rate = 0.2, generate_length = 500, total_epochs = 40,gen_samples = 5, layer_num = 2):

    print("weights:", weights)
    print("mode:", mode)
    print("epochs:", total_epochs)
    print("dropout_rate:", dropout_rate)
    print("layers:", layer_num)


    # Find VOCAB_SIZE and dictionaries
    VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch = load_vocabulary(data_dir, seq_length, batch_size)

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
      model.load_weights(weights)
      epoch = int(weights[weights.rfind('_') + 1:weights.find('.')])
    else:
      epoch = 0

    # Training
    if mode == 'train':
      while epoch <= total_epochs:
        print('\n\nEpoch: {}\n'.format(epoch))
        model.fit_generator(data_generator(data_dir, seq_length, batch_size, steps_per_epoch), \
        steps_per_epoch=steps_per_epoch, verbose = 1, epochs = 1)
        epoch += 1
        
        if epoch == 1:  # For h5py test on TensorPort
          model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}_dropout_{}_.hdf5'.format(layer_num, hidden_dim, epoch, dropout_rate))
        
        if epoch % 10 == 0:
          model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}_dropout_{}_.hdf5'.format(layer_num, hidden_dim, epoch, dropout_rate))
          generate_text(model, 100, VOCAB_SIZE, ix_to_char)
          
      # Generate samples
      print("Generating %i sample poems." % gen_samples)
      for i in range(gen_samples):
        print(i, "\n")
        generate_text(model, generate_length, VOCAB_SIZE, ix_to_char)


    # Else, loading the trained weights and performing generation only
    elif weights != '':
      # Loading the trained weights
      #model.load_weights(weights)
      print("Translating")
      for i in range(5):
        print(generate_text(model, generate_length, VOCAB_SIZE, ix_to_char))
        print('\n\n')
    else:
      print('\n\nNothing to do!')