#!/usr/bin/env python -tt
# coding: utf-8
"""
Code for the Avito fraud detection competition
Adam Kalman, Aleksey Kocherzhenko, Henoch Wong
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
import string
import APatK
import WordStatsModule as ws

'''change dataFolder to the correct local path'''
dataFolder = "/Users/adamkalman/Desktop/CDIPS/Data"
scratchFolder = "/Users/adamkalman/Desktop/CDIPS/interdata"

def readInFile(filename):
  itemIds = []
  scores = []
  fp = open(filename, 'r')
#  fp.readline() #ignore header
  for line in fp:
    itemId_and_score = line[:-1].split(',')
    itemId = int(itemId_and_score[0])
    score = float(itemId_and_score[-1])
    itemIds.append(itemId)
    scores.append(score)
  fp.close()
  return itemIds, scores

def main():
  
  allIds = []
  allScores = []
  for output_file in os.listdir(scratchFolder):
    if output_file[:19] == 'solution_to_submit_':
      print "Reading", output_file
      itemIds, scores = readInFile(os.path.join(scratchFolder, output_file))
#      print len(itemIds), len(scores)
      print len(itemIds)
    
      if "Nedvijimost" in output_file: 
        scores = list(np.zeros(len(scores))) #makes scores all zero
        
      #print scores  
      
      allIds = allIds + itemIds
      allScores = allScores + scores
#      print len(allIds), len(allScores)

  Dict2rank = {}
  for i in range(len(allIds)):
    Dict2rank[allIds[i]] = allScores[i]

  fp = open(os.path.join(scratchFolder, 'text_solution_to_submit.csv'), 'w')
  RankedList = sorted(Dict2rank.iteritems(), key=lambda x: x[1])
  for element in reversed(RankedList):
    #print element[0], element[1]
    fp.write(" %15d , %30.20f \n" %(element[0], element[1]))
  fp.close()
  
#    item_ids = processData(os.path.join(dataFolder,"avito_train.tsv"), itemsLimit=None)
#    item_ids = processData(os.path.join(dataFolder,"avito_test.tsv"), itemsLimit=None)
    
  logging.info("Done.")
                               
if __name__=="__main__":            
  main()            
    
    
    
