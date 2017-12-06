from __future__ import print_function

from lstm import *

# Call out main function
if __name__ == "__main__":
    print("Start training")
    lstm(data_dir = "../data/poets_top_clean_500.txt", layer_num=2)
    print("Finished")