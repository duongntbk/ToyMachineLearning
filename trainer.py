# -*- coding: utf-8 -*-

'''
Train model to recognize hand written number using MNIST data set.
'''

import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import constants
from extractor import Extractor


def main():
    '''
    Entry point when executing from commandline.
    '''

    try:
        extractor = Extractor()

        # Extract train set from archive file
        train_set, _, _ = extractor.extract(constants.MNIST_DATASET_PATH)

        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(train_set[1])

        # Use SVC to train a recognition model
        recognizer = SVC(C=1.0, kernel='linear', probability=True)
        recognizer.fit(train_set[0], labels)

        # Write trained model and label encoder to file
        with open(constants.MODEL_FILE_PATH, 'wb') as f:
            f.write(pickle.dumps(recognizer))
        with open(constants.LABEL_ENCODER_FILE_PATH, 'wb') as f:
            f.write(pickle.dumps(label_encoder))

        print('Training done')


    except Exception as e:
        print('An error has occurred')
        print(str(e))

if __name__ == '__main__':
    main()
