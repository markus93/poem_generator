from __future__ import print_function
import numpy as np

# method for generating text, using model
def generate_text(model, length, vocab_size, ix_to_char, use_subwords, end_symbol = "$"):
    # starting with random character
    ix = np.random.randint(vocab_size)
    y_char = [ix_to_char[ix]]
    X = np.zeros((1, length, vocab_size))

    if use_subwords:
        end = ' '
    else:
        end = ''
    
    for i in range(length):
        # appending the last predicted character to sequence
        X[0, i, :][ix] = 1
        print(ix_to_char[ix], end=end)
        pred = model.predict(X[:, :i+1, :])[0]
        ix = np.random.choice(np.arange(vocab_size), p = pred[-1])  # Chooses prediction with probability of next char

        if end_symbol == ix_to_char[ix]:
            break

        y_char.append(ix_to_char[ix])    
        
    return (end).join(y_char)

# Read data and generate vocabulary    
def load_vocabulary(data_dir, seq_length, batch_size, use_subwords):
    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    
    if use_subwords:  # Split data into subwords
        data = data.split()
    
    chars = sorted(list(set(data)))  # get possible chars
    VOCAB_SIZE = len(chars)

    print('Data length: {} chars/subwords'.format(len(data)))
    print('Vocabulary size: {} chars/subwords'.format(VOCAB_SIZE))

    ix_to_char = {ix:char for ix, char in enumerate(chars)}  # index to char map  # can also be subwords here
    char_to_ix = {char:ix for ix, char in enumerate(chars)}  # char to index map
    
    steps_per_epoch = int(len(data)/seq_length/batch_size)
    
    return VOCAB_SIZE, ix_to_char, char_to_ix, steps_per_epoch, data
    
# Load vocabulary poem by poem
def load_vocabulary_poem(data_dir, batch_size, poem_end, use_subwords, end_symbol):

    data = open(data_dir, 'r', encoding="utf-8").read()  # Read data
    poems = data.split(poem_end)  # list with all the poems in data
    poems = [s for s in poems if len(s) >= 2]  # Leave out empty poems.
    
    seq_length = len(max(poems, key=len)) + 1 # +1 so the longest poem has also end symbol
    
    if use_subwords:  # Split data into subwords
        data_new = data.split()  # Later initial data needed
        chars = sorted(list(set(data_new)))  # get possible subwords
    else:
        chars = sorted(list(set(data)))  # get possible chars
        
    chars.append(end_symbol)
        
    VOCAB_SIZE = len(chars)

    print('Data length: {} poems'.format(len(poems)))
    print('Vocabulary size: {} chars/subwords'.format(VOCAB_SIZE))

    ix_to_word = {ix:char for ix, char in enumerate(chars)}  # index to char map
    word_to_ix = {char:ix for ix, char in enumerate(chars)}  # char to index map
    
    steps_per_epoch = int(len(poems)/batch_size)  # One poem per batch
    
    print("Steps per epoch:", steps_per_epoch)
    
    return VOCAB_SIZE, ix_to_word, word_to_ix, steps_per_epoch, data

    
# Read in data by batches, atm only for char-to-char
def data_generator(data, seq_length, batch_size, steps_per_epoch):

    chars = sorted(list(set(data)))  # get possible chars
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
        
# Read in data in poem by poem
def data_generator_poem(data, batch_size, poem_end, use_subwords, end_symbol = "$"):
    
    poems = data.split(poem_end)
    poems = [s for s in poems if len(s) >= 2]  # Leave out empty poems.
    
    # Get longest poem to set the sequence length
    seq_length = len(max(poems, key=len)) + 1 # +1 so the longest poem has also end symbol

    print("Subwords:", use_subwords)
    
    if use_subwords:
        data = data.split()
    
    chars = sorted(list(set(data)))  # get possible chars/subwords
    chars.append(end_symbol)
    VOCAB_SIZE = len(chars)   
    
    print('Data length: {} poems'.format(len(poems)))
    print('Vocabulary size: {} chars/subwords'.format(VOCAB_SIZE))

    ix_to_word = {ix:char for ix, char in enumerate(chars)}  # index to char/subword map
    word_to_ix = {char:ix for ix, char in enumerate(chars)}  # char/subword to index map
    
    batch_nr = 0
    steps_per_epoch = len(poems)
    
    # Generate data matrices
    while True:
    
        for i in range(0, batch_size):   
        
            poem = poems[batch_nr*batch_size + i]
            
            if use_subwords:
                elements = poem.split()
                elements = elements + (seq_length-len(elements))*[end_symbol]  # Add end_symbols 
            else:
                elements = poem
                elements = elements + (seq_length-len(elements))*end_symbol  # Add end_symbols 
       
                        
            #seq_length = len(elements) - 1  # One less to predict

            X = np.zeros((batch_size, seq_length-1, VOCAB_SIZE))  # input data
            y = np.zeros((batch_size, seq_length-1, VOCAB_SIZE))
            
            X_sequence = elements[:-1]  # Take all but last subword to learn
            X_sequence_ix = [word_to_ix[value] for value in X_sequence]
            input_sequence = np.zeros((seq_length-1, VOCAB_SIZE))

            for j in range(len(X_sequence)):
                input_sequence[j][X_sequence_ix[j]] = 1.
                X[i] = input_sequence  # Batch size 1

            y_sequence = elements[1:]  # Next subword to predict
            y_sequence_ix = [word_to_ix[value] for value in y_sequence]
            target_sequence = np.zeros((seq_length-1, VOCAB_SIZE))
            
            for j in range(len(y_sequence)):
                target_sequence[j][y_sequence_ix[j]] = 1.
                y[i] = target_sequence
        
        if batch_nr == (steps_per_epoch-1):  # Because we start from zero (in case many epochs learnt together)
            batch_nr = 0  # Back to beginning - so we could loop indefinitely
        else:
            batch_nr += 1
                
        
        yield(X, y)
