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
def load_vocabulary(data_dir, seq_length, batch_size, use_subwords):
    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    
    if use_subwords:  # Split data into subwords
        data = data.split()
    
    chars = list(set(data))  # get possible chars
    VOCAB_SIZE = len(chars)

    print('Data length: {} chars/subwords'.format(len(data)))
    print('Vocabulary size: {} chars/subwords'.format(VOCAB_SIZE))

    ix_to_char = {ix:char for ix, char in enumerate(chars)}  # index to char map  # can also be subwords here
    char_to_ix = {char:ix for ix, char in enumerate(chars)}  # char to index map
    
    steps_per_epoch = int(len(data)/seq_length/batch_size)
    
    return VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch, data
    
# Load vocabulary poem by poem
def load_vocabulary_poem(data_dir, poem_end):

    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    poems = data.split(poem_end)  # list with all the words in data
    words = list(set(poems))  # get possible words
    VOCAB_SIZE = len(words)

    print('Data length: {} poems'.format(len(poems)))
    print('Vocabulary size: {} subwords'.format(VOCAB_SIZE))

    ix_to_word = {ix:subword for ix, subword in enumerate(words)}  # index to char map
    word_to_ix = {subword:ix for ix, subword in enumerate(words)}  # char to index map
    
    steps_per_epoch = len(data)  # One poem per batch
    
    return VOCAB_SIZE, ix_to_word, word_to_ix, steps_per_epoch

    
# Read in data by batches, atm only for char-to-char
def data_generator(data, seq_length, batch_size, steps_per_epoch):

    chars = list(set(data))  # get possible chars
    VOCAB_SIZE = len(chars)

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
        
# Read in data by batches, atm only for char-to-char
def data_generator_poem(data_dir, poem_end):

    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    poems = data.split(poem_end)  # TODO splits also verses (so not actually poems atm)
    words = list(set(data.split()))  # get possible words
    VOCAB_SIZE = len(words)

    print('Data length: {} poems'.format(len(poems)))
    print('Vocabulary size: {} subwords'.format(VOCAB_SIZE))

    ix_to_word = {ix:subword for ix, subword in enumerate(words)}  # index to subword map
    word_to_ix = {subword:ix for ix, subword in enumerate(words)}  # subword to index map
    
    batch_nr = 0
    steps_per_epoch = len(poems)
    
    batch_size = 1 ## Atm only one poem per batch no padding added
    
    while True:
    
        poem = poems[batch_nr]
                    
        subwords = poem.split()  # Split into subwords
        seq_length = len(subwords) - 1  # One less to predict
        
        # TODO should use start and end token?

        X = np.zeros((batch_size, seq_length, VOCAB_SIZE))  # input data
        y = np.zeros((batch_size, seq_length, VOCAB_SIZE))
        
        X_sequence = subwords[:-1]  # Take all but last subword to learn
        X_sequence_ix = [word_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((seq_length, VOCAB_SIZE))

        for j in range(len(X_sequence)):
            input_sequence[j][X_sequence_ix[j]] = 1.
            X[0] = input_sequence  # Batch size 1

        y_sequence = subwords[1:]  # Next subword to predict
        y_sequence_ix = [word_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((seq_length, VOCAB_SIZE))
        
        for j in range(len(y_sequence)):
            target_sequence[j][y_sequence_ix[j]] = 1.
            y[0] = target_sequence
        
        if batch_nr == (steps_per_epoch-1):  # Because we start from zero (in case many epochs learnt together)
            batch_nr = 0  # Back to beginning - so we could loop indefinitely
        else:
            batch_nr += 1
                
        
        yield(X, y)