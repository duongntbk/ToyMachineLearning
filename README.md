# About

This is a simple machine learning example. It uses Support Vector Classifier to recognize handwritten digits.
The training, validation and test data set is THE MNIST DATABASE of handwritten digits, which can be found at the following link:
http://yann.lecun.com/exdb/mnist/

# Usage

**Note**: please make sure that you have more than 4GB or RAM and is using Python 64bit before trying to run this problem. Otherwise you might run into MemoryError.

### Install required packages 

    pip install -r requirements.txt

### Extract the MNIST DATABASE

    python extractor.py
    
The training, validation and test data set will be extracted from mnist.pkl.gz archive file and written into csv files, which is stored in *train* and *test* folder.

### Train the model

To use the whole training set to train the model:

    python trainer.py

Since the training set has a lot of records, training would takes a long time. For demostration purpose, a simple model trained using a few thousands records will usually suffice.
To limit the number of records used while training the model, add the --max argument. For example, to use 5000 records to train a simple model:

    python trainer.py --max=5000

You can also set C, gamma or kernel to be used in SVC. With max=10000, C=100.0, gamma=0.01, kernel='rbf', an accuracy of 96.58% on test datasets can be reached:

    python trainer.py --max=10000 --C=100.0 --gamma=0.01 --kernel=rbf

The trained model will be stored as pickle files in *model* folder.

### Verify the model

The MNIST DATABASE contains a test data set, we have extracted it to *test* folder in a previous step. We verify the trained model using that test data set:

    python demo.py --path=test/test_data.csv

If you want to verify just a few top records in test data set, add the --max argument. To verify the first 10 records in test data set:

    python demo.py --path=test/test_data.csv --max=10

# License

MIT License

https://opensource.org/licenses/MIT