# -*- coding: utf-8 -*-

'''
This module use trained model to recognize hand written number.
'''

import numpy as np

class HandWrittenRecognizer:
    '''
    This class use pre-trained SVC model to recognize hand written number.
    '''

    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
    
    def recognize(self, input):
        '''
        Use SVM model to recognize hand written number.
        '''

        preds = self.model.predict_proba(input)[0]
        max_class = np.argmax(preds)
        prob = preds[max_class]
        number = self.label_encoder.classes_[max_class]

        return {
            'number': number,
            'probability': prob
        }