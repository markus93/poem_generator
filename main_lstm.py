from __future__ import print_function

from scripts.lstm import *
import argparse


# Call out main function
if __name__ == "__main__":
    print("Start training")
    
    # Parsing arguments for Network definition
    ap = argparse.ArgumentParser()
    ap.add_argument('-data_dir', default='data/poems_clean.txt')
    ap.add_argument('-batch_size', type=int, default=100)
    ap.add_argument('-seq_length', type=int, default=100)
    ap.add_argument('-hidden_dim', type=int, default=500)
    ap.add_argument('-weights', default='')
    ap.add_argument('-mode', default='train')
    ap.add_argument('-dropout_rate', type=float, default=0.2)
    ap.add_argument('-generate_length', type=int, default=300)
    ap.add_argument('-total_epochs', type=int, default=10)
    ap.add_argument('-gen_samples', type=int, default=5)
    ap.add_argument('-layer_num', type=int, default=2)
    ap.add_argument('-save_every', type=int, default=2)
    ap.add_argument('-temp', type=float, default=1)
    ap.add_argument('-use_subwords', type=bool, default=False)

    args = vars(ap.parse_args())
    
    print(args['data_dir'])
    
    ## Run lstm function
    lstm(**args)
    print("Finished")