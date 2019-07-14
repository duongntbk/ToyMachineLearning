# -*- coding: utf-8 -*-

'''
Train model to recognize hand written digits using MNIST data set.
'''

import argparse
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import constants
from extractor import Extractor
from utils import is_number


def main():
    '''
    Entry point when executing from commandline.
    '''

    parser=argparse.ArgumentParser()
    parser.add_argument('--max', help='The maximum number of data points to be used while training model. \
        if empty, the whole training set will be used')
    args=parser.parse_args()

    if args.max and not is_number(args.max):
        print(constants.TRAINER_HELP_MSG)
        return

    try:
        extractor = Extractor()

        # Extract train set from archive file
        train_set, _, _ = extractor.extract(constants.MNIST_DATASET_PATH)

        if args.max:
            feature_set = train_set[0][:int(args.max)]
            label_set = train_set[1][:int(args.max)]
        else:
            feature_set = train_set[0]
            label_set = train_set[1]

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(label_set)

        # Use SVC to train a recognition model
        recognizer = SVC(C=1.0, kernel='linear', probability=True)
        recognizer.fit(feature_set, labels)

        # Write trained model and label encoder to file
        with open(constants.MODEL_FILE_PATH, 'wb') as f:
            f.write(pickle.dumps(recognizer))
        with open(constants.LABEL_ENCODER_FILE_PATH, 'wb') as f:
            f.write(pickle.dumps(label_encoder))

        print('Training done')

    except MemoryError:
        # The training dataset is quite big, more than 4GB of RAM and python 64 bit is required
        print('An memory error has occurred, please check if you have enough memory \
                and you are using python 64bit')

    except Exception as e:
        print('An error has occurred')
        print(str(e))

if __name__ == '__main__':
    main()
