import argparse
import pandas as pd
from sklearn.model_selection import train_test_split as tts

def parse_arguments():
    parser = argparse.ArgumentParser(description = 'Split file.csv into training and test datasets')
    parser.add_argument('--src', help = 'source file', default = "ratings.csv")
    parser.add_argument('--size', help = 'size of test dataset (number from [0,1], default: 10%)', default = 0.1)
    args = parser.parse_args()

    return args.src, args.size

src, size = parse_arguments()

df = pd.read_csv(src)

train, test = tts(df, test_size = float(size), random_state = 42, shuffle = True, stratify = df['userId'])
name = src.split('.')[0] 

if __name__ == '__main__':
    train.to_csv(name + '_train.csv', index = False)
    test.to_csv(name + '_test.csv', index = False)
