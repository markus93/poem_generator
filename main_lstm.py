from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers.wrappers import TimeDistributed
from RNN_utils import *

def main():

    # TODO: read in from file.
    DATA_DIR = "data/poets_top_100k.txt"
    BATCH_SIZE = 100
    HIDDEN_DIM = 500
    SEQ_LENGTH = 100
    WEIGHTS = ""
    MODE = "train"
    DROPOUT_RATE = 0.2
    GENERATE_LENGTH = 500
    LAYER_NUM = 2
    EPOCHS = 40
    GEN_SAMPLES = 5


    print("Weights:", WEIGHTS)
    print("Mode:", MODE)


    # Find VOCAB_SIZE and dictionaries
    VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch = load_vocabulary(DATA_DIR, SEQ_LENGTH, BATCH_SIZE)

    # Creating and compiling the Network
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    model.add(Dropout(rate = DROPOUT_RATE))
    for i in range(LAYER_NUM - 1):
      model.add(LSTM(HIDDEN_DIM, return_sequences=True))
      model.add(Dropout(rate = DROPOUT_RATE))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    # Load model if file for weights is given
    if not WEIGHTS == '':
      model.load_weights(WEIGHTS)
      epoch = int(WEIGHTS[WEIGHTS.rfind('_') + 1:WEIGHTS.find('.')])
    else:
      epoch = 0

    # Training
    if MODE == 'train':
      while epoch <= EPOCHS:
        print('\n\nEpoch: {}\n'.format(epoch))
        model.fit_generator(data_generator(DATA_DIR, SEQ_LENGTH, BATCH_SIZE, steps_per_epoch), \
        steps_per_epoch=steps_per_epoch, verbose = 1, epochs = 1)
        epoch += 1
        if epoch % 10 == 0:
          model.save_weights('checkpoint_layer_{}_hidden_{}_epoch_{}.hdf5'.format(LAYER_NUM, HIDDEN_DIM, epoch))
          
      # Generate samples
      print("Generating %i sample poems." % GEN_SAMPLES)
      for i in range(GEN_SAMPLES):
        print(i, "\n")
        generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char)
 

    # Else, loading the trained weights and performing generation only
    elif WEIGHTS != '':
      # Loading the trained weights
      #model.load_weights(WEIGHTS)
      print("Translating")
      for i in range(5):
        print(generate_text(model, GENERATE_LENGTH, VOCAB_SIZE, ix_to_char))
        print('\n\n')
    else:
      print('\n\nNothing to do!')

# Call out main function
if __name__ == "__main__":
    main()