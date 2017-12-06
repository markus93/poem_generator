from __future__ import print_function

from scripts.lstm import *

# Call out main function
if __name__ == "__main__":
    print("Start training")
    lstm(data_dir = "../data/poets_top_100k.txt", layer_num=2)
    print("Finished")