# -*- coding: utf-8 -*-

'''
Driver program.
'''

import argparse
import pickle

import numpy as np

import constants
from recognizer import HandWrittenRecognizer
from utils import is_number


def load_model():
    '''
    This method loads retrained model and label encoder.
    '''

    with open(constants.LABEL_ENCODER_FILE_PATH, 'rb') as f:
        label_encoder = pickle.loads(f.read())
    with open(constants.MODEL_FILE_PATH, 'rb') as f:
        model = pickle.loads(f.read())

    return model, label_encoder

def verify_argument(args):
    '''
    Input check for main method.
    '''

    # If path to test set is not provided, validation failed
    if not args.path:
        return False

    # If max option is provided, check if it is a integer
    if args.max and not is_number(args.max):
        return False

    return True

def main():
    '''
    Entry point when executing from commandline.
    '''

    parser=argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to test set.')
    parser.add_argument('--max', help='The maximum number of data points in test set to be processed. \
        if empty, the whole test set will be processed')
    args=parser.parse_args()

    # Input check
    if not verify_argument(args):
        print(constants.DEMO_HELP_MSG)
        return

    test_file_path = args.path
    test_count = args.max

    try:
        model, label_encoder = load_model()
        recognizer = HandWrittenRecognizer(model, label_encoder)

        with open(test_file_path, 'r') as f:
            # If option max is provided
            if test_count:
                test_count = int(test_count)
                head = [next(f) for _ in range(test_count)]
            # If option max is not provided
            else:
                # Read the whole test data file
                head = [line.rstrip() for line in f]
        
        for line in head:
            num_arr = [float(x) for x in line.split(',')]
            num_arr = np.array(num_arr)
            num_arr = num_arr.reshape(1, -1)
            result = recognizer.recognize(num_arr)
            print('Predict {0} with probability {1}'.format(result["number"], result["probability"]))
        

    except Exception as e:
        print('An error has occurred')
        print(str(e))

if __name__ == '__main__':
    main()
