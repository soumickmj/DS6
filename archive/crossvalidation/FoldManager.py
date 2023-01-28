#!/usr/bin/env python
'''

Purpose : Holder for all the folds for different stress testing

'''

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

def getTrainingFolds(stress_no):
    TrainingDictionary = {
        1:[
            ["au70.nii"],  # fold1 images
            ["sc17.nii"],  # fold2 images
            ["um68.nii"],  # fold3 images
            ],
        2: [
            ["au70.nii","xi27.nii"],  #fold1 images
            ["kc73.nii","ww25.nii"],  #fold2 images
            ["me21.nii","vk04.nii"],  #fold3 images
            ],
        3: "March",
        4: [["au70.nii", "kc73.nii", "um68.nii", "sc17.nii"],
            ["me21.nii", "mf79.nii" , "nv85.nii", "vk04.nii"],
            ["kc73.nii", "me21.nii" ,"nv85.nii","pa30.nii"]],
        5: "May",
        6: [['au70.nii', 'kc73.nii','um68.nii', 'sc17.nii', 'me21.nii', 'pa30.nii'],
            ['vk04.nii', 'ww25.nii', 'xi27.nii', 'au70.nii', 'kc73.nii', 'me21.nii'],
            ['me21.nii', 'mf79.nii', 'nv85.nii', 'pa30.nii', 'sc17.nii', 'um68.nii']]
    }
    return TrainingDictionary.get(stress_no)


def getValidationFolds(stress_no):
    ValidationDictionary = {
        1: [
            ["kc73.nii", "pa30.nii"],  # fold1 images
            ["mf79.nii", "nv85.nii"],  # fold2 images
            ["kc73.nii", "mf79.nii"],  # fold3 images
            ],
        2: [
            ["kc73.nii","ww25.nii"], #fold1 images
            ["me21.nii","vk04.nii"], #fold2 images
            ["au70.nii","xi27.nii"], #fold3 images
            ],
        3: "test",
        4: [['me21.nii', 'pa30.nii'],
            ['pa30.nii','ww25.nii'],
            ['au70.nii', 'mf79.nii']],
        5: "May",
        6: [['mf79.nii', 'ww25.nii'],
            ['mf79.nii', 'nv85.nii'],
            ['vk04.nii','ww25.nii']]
    }
    return ValidationDictionary.get(stress_no)


def getTestingFolds(stress_no):
    TestingDictionary = {
        1: [
             ["mf79.nii", "nv85.nii", "um68.nii"],  # fold1 images
             ["au70.nii", "kc73.nii", "me21.nii"],  # fold2 images
             ["me21.nii", "nv85.nii", "pa30.nii"],  # fold3 images
            ],
        2: [
            ["me21.nii","mf79.nii","nv85.nii"], #fold1 images
            ["au70.nii","xi27.nii","sc17.nii"], #fold2 images
            ["mf79.nii","nv85.nii","pa30.nii"], #fold3 images
            ],
        3: "March",
        4: [['vk04.nii', 'ww25.nii', 'xi27.nii'],
            ['um68.nii', 'au70.nii', 'kc73.nii'],
            ['sc17.nii','um68.nii','ww25.nii']],
        5: "May",
        6: [['xi27.nii', 'vk04.nii', 'nv85.nii'],
            ['pa30.nii', 'sc17.nii', 'um68.nii'],
            ['xi27.nii', 'au70.nii', 'kc73.nii']],
    }
    return TestingDictionary.get(stress_no)
