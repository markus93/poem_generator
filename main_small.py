from __future__ import print_function

from scripts.lstm import *

# Call out main function
if __name__ == "__main__":
    print("Start training")
    lstm(data_dir="data/poems_test_small.txt")
    print("Finished")