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
  for line in fp:
    itemId_and_score = line[:-1].split(',')
    itemId = int(itemId_and_score[0])
    score = float(itemId_and_score[-1])
    itemIds.append(itemId)
    scores.append(score)
  fp.close()
  return itemIds, scores

def readInGoodFile(filename):
  itemIds = []
  fp = open(filename, 'r')
  for line in fp:
    itemId = int(line.strip())
    itemIds.append(itemId)
  fp.close()
  return itemIds

# get number of entries in a file (number of lines - 1 for header)
def file_len(filename):
  with open(filename) as f:
    for i, l in enumerate(f):
      pass
  return i

def main():
  
  allIds_text = []
  allScores_text = []
  output_file = 'text_solution_to_submit.csv'
  print "Reading", output_file
  itemIds_text, scores_text = readInFile(os.path.join(scratchFolder, output_file))
  allIds_text = allIds_text + itemIds_text
  allScores_text = allScores_text + scores_text

  Dict2rank = {}
  Dict2rank_text = {}
  for i in range(len(allIds_text)):
    Dict2rank_text[allIds_text[i]] = allScores_text[i]

  for key in Dict2rank_text.keys():
    Dict2rank[key] = Dict2rank_text[key]

  RankedList = sorted(Dict2rank.iteritems(), key=lambda x: x[1])

  goodIdsAll = []
  for filename in os.listdir(scratchFolder):
    if "_testgood" in filename:
      print filename
      goodIds = readInGoodFile(os.path.join(scratchFolder, filename))
      goodIdsAll = goodIdsAll + goodIds

  predfile = 'solution_to_submit.csv'
  fp = open(predfile, 'w')
  fp.write('id\n')
  for element in reversed(RankedList):
#    print element[0], element[1]
    fp.write(str(element[0])+'\n')
  for anId in goodIdsAll:
    fp.write(str(anId)+'\n')
  fp.close()

  logging.info("Done.")
                               
if __name__=="__main__":            
  main()            
    
    
    
