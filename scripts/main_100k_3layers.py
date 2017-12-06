from __future__ import print_function

from lstm import *

# Call out main function
if __name__ == "__main__":
    print("Start training")
    lstm(data_dir = "../data/poets_top_100k.txt", layer_num=3)
    print("Finished")