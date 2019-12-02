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
        if empty, the whole training set will be used.')
    parser.add_argument('--C', type=float, default=1.0, help='C parameter for SVC algorithm. \
        if empty, C will be set to 1.0.')
    parser.add_argument('--gamma', type=float, default=.01, help='Gamma parameter for SVC algorithm. \
        if empty, gamma will be set to 0.01.')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel for SVC algorithm. \
        if empty, linear kernel will be used. Only linear and rbf kernel are supported at the moment.')
    args=parser.parse_args()

    if (args.max and not is_number(args.max)) or \
            (args.kernel != 'kernel' and args.kernel != 'rbf'):
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
        recognizer = SVC(C=args.C, gamma=args.gamma, kernel=args.kernel, probability=True)
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
