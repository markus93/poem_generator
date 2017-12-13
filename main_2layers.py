from __future__ import print_function

from scripts.lstm import *

# Call out main function
if __name__ == "__main__":
    print("Start training")
    lstm(data_dir = "data/poems_clean.txt", layer_num=2, dropout_rate=0.1, save_every=1)
    print("Finished")