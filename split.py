#!/usr/bin/env python -tt
# coding: utf-8
"""
Code for the Avito fraud detection competition
Adam Kalman, Aleksey Kocherzhenko, Henoch Wong

Takes part of tsv file and makes it a training set. 
Takes another part and makes it a test set, and the answers to the test set in a separate file.
(Kaggle starter code needs it that way, or else we'd have to rewrite it.)
Future improvement: choose data randomly instead of in order.
"""
import csv
import re
import nltk.corpus
from collections import defaultdict
import scipy.sparse as sp
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from nltk import SnowballStemmer
import random as rnd 
import logging
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import math
import APatK
import WordStatsModule as ws

'''change dataFolder to the correct local path'''
dataFolder = "/Users/adamkalman/Desktop/CDIPS/Data"
scratchFolder = "/Users/adamkalman/Desktop/CDIPS/interdata"

scratchfile = "avito_train.tsv"

def file_len(filename):
  print "Determining file length. Counting lines..."
  with open(filename) as f:
    for i, l in enumerate(f):
      if i%1000 == 0:
        print i 
      pass
  return i  

    
def main():        
    trainingPercentage = 80
    testPercentage = 20 #these need not add up to 100. Make them small so code runs fast, until code is done and ready for final huge data set.
#    fileLength = file_len(os.path.join(dataFolder,scratchfile))
    fileLength = 3995000
    trainSize = fileLength//100*trainingPercentage
    testSize = fileLength//100*testPercentage
    print "Training sample size = ", trainSize
    print "Test sample size = ", testSize
    
    f1 = open(os.path.join(dataFolder,scratchfile),'r')
    f1read = csv.reader(f1,dialect='excel-tab')
    f2 = open(os.path.join(dataFolder,'small_train.tsv'),'w')
    f2write = csv.writer(f2, dialect='excel-tab')
    f3 = open(os.path.join(dataFolder,'small_test.tsv'),'w')
    f3write = csv.writer(f3, dialect='excel-tab')
    f4 = open(os.path.join(dataFolder,'small_test_solution.tsv'),'w')
    f4write = csv.writer(f4, dialect='excel-tab')
    f5 = open(os.path.join(dataFolder,'small_test_blockedids.tsv'),'w')
    f5write = csv.writer(f5, dialect='excel-tab')
    print "Writing files..."
    for i,row in enumerate(f1read):
        if i == 0:
            f2write.writerow(row)
            del row[12]
            del row[8]
            del row[7]
            f3write.writerow(row)
            f4write.writerow((row[0],row[8]))
            f5write.writerow((row[0],))
        elif i < testSize:
            f4write.writerow((row[0],row[8]))
            if row[8] == '1':
                f5write.writerow((row[0],))
            del row[12]
            del row[8]
            del row[7]
            f3write.writerow(row)
        elif i < testSize + trainSize:
            f2write.writerow(row)
        if i%10000 == 0:
            print i
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    print "Done"

if __name__=="__main__":            
    main()        
    
    
