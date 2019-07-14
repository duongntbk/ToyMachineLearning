# -*- coding: utf-8 -*-

'''
This module provides small utility methods for other modules.
'''

def is_number(input, check_method=int):
    '''
    Check if input is a number.
    The default type to be check with is integer.
    '''

    # Easier to ask for forgiveness than permission
    try:
        check_method(input)
        return True
    except ValueError:
        return False

