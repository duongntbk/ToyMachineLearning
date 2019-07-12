# -*- coding: utf-8 -*-

'''
This module retrieves the training, validation and test set from the MNIST data set
and written them to csv file.
'''

import gzip
import os
import pickle

import numpy as np

import constants


class Extractor:
    '''
    Helper class to export data to csv file.
    If data is test set, do not export labels.
    '''

    def extract(self, archive_path):
        '''
        Extract dataset from archive.
        The order is train set, validation set, test set
        '''

        with gzip.open(archive_path, 'rb') as f:
            return pickle.load(f, encoding='latin1')

    def export_to_csv(self, name, dataset, is_test=False):
        '''
        Method to export data to csv file.
        If data is test set, do not export labels.
        '''

        features = [x.tolist() for x in dataset[0]]

        if not is_test:
            folder_path = constants.TRAIN_FOLDER
            # This dataset is not test set, export labels as well
            labels = [x.tolist() for x in dataset[1]]
            csv_data = np.insert(features, 0, labels, axis=1)
        else:
            folder_path = constants.TEST_FOLDER
            # This dataset is test set, do not export labels
            csv_data = features

        file_name = '{0}_data.csv'.format(name)
        file_path = os.path.join(folder_path, file_name)
        np.savetxt(file_path, csv_data, delimiter=',')

def main():
    '''
    Entry point when executing from commandline.
    '''

    try:
        extractor = Extractor()

        # Extract train, validation, test set from archive file
        train_set, validation_set, test_set = extractor.extract(constants.MNIST_DATASET_PATH)
        
        # Write each data set to csv file
        extractor.export_to_csv('train', train_set, is_test=False)
        print('Train set exported')
        extractor.export_to_csv('validation', validation_set, is_test=False)
        print('Validation set exported')
        extractor.export_to_csv('test', test_set, is_test=True)
        print('Test set exported')

    except Exception as e:
        print('An error has occurred')
        print(str(e))

if __name__ == '__main__':
    main()
