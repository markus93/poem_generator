from __future__ import print_function
import numpy as np

# method for generating text, using model
# TODO add randomness to output.
def generate_text(model, length, vocab_size, ix_to_char):
	# starting with random character
	ix = [np.random.randint(vocab_size)]
	y_char = [ix_to_char[ix[-1]]]
	X = np.zeros((1, length, vocab_size))
	for i in range(length):
		# appending the last predicted character to sequence
		X[0, i, :][ix[-1]] = 1
		#ix_to_char[ix[-1]], end=""
		ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
		y_char.append(ix_to_char[ix[-1]])
	return ('').join(y_char)

# Read data and generate vocabulary    
def load_vocabulary(data_dir, seq_length, batch_size):
    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    chars = list(set(data))  # get possible chars
    VOCAB_SIZE = len(chars)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

    ix_to_char = {ix:char for ix, char in enumerate(chars)}  # index to char map
    char_to_ix = {char:ix for ix, char in enumerate(chars)}  # char to index map
    
    steps_per_epoch = len(data)//seq_length//batch_size
    
    return VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch
    
# Read in data by batches, atm only for char-to-char
def data_generator(data_dir, seq_length, batch_size, steps_per_epoch):
    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    chars = list(set(data))  # get possible chars
    VOCAB_SIZE = len(chars)

    print('Data length: {} characters'.format(len(data)))
    print('Vocabulary size: {} characters'.format(VOCAB_SIZE))

    ix_to_char = {ix:char for ix, char in enumerate(chars)}  # index to char map
    char_to_ix = {char:ix for ix, char in enumerate(chars)}  # char to index map
    
    batch_nr = 0
    
    while True:
        
        X = np.zeros((batch_size, seq_length, VOCAB_SIZE))  # input data
        y = np.zeros((batch_size, seq_length, VOCAB_SIZE))
        
        pos_start = batch_nr*batch_size*seq_length  # Continue where left on from patch
        
        for i in range(0, batch_size):        
            
            X_sequence = data[pos_start + i*seq_length:pos_start + (i+1)*seq_length]
            X_sequence_ix = [char_to_ix[value] for value in X_sequence]
            input_sequence = np.zeros((seq_length, VOCAB_SIZE))

            for j in range(len(X_sequence)):  # Last sequence otherwise shorter
                input_sequence[j][X_sequence_ix[j]] = 1.
                X[i] = input_sequence

            y_sequence = data[pos_start+i*seq_length+1:pos_start + (i+1)*seq_length+1]  # next character, as we want to predict next character
            y_sequence_ix = [char_to_ix[value] for value in y_sequence]
            target_sequence = np.zeros((seq_length, VOCAB_SIZE))
            
            for j in range(len(y_sequence)):
                target_sequence[j][y_sequence_ix[j]] = 1.
                y[i] = target_sequence
        
        if batch_nr == (steps_per_epoch-1):  # Because we start from zero
            batch_nr = 0  # Back to beginning - so we could loop indefinitely
        else:
            batch_nr += 1
                
        
        yield(X, y)