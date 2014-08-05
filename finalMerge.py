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

# get number of entries in a file (number of lines - 1 for header)
def file_len(filename):
  with open(filename) as f:
    for i, l in enumerate(f):
      pass
  return i

def main():
  
  allIds_text = []
  allScores_text = []
  allIds_pl = []
  allScores_pl = []
  output_file = 'text_solution_to_submit.csv'
  print "Reading", output_file
  itemIds_text, scores_text = readInFile(os.path.join(scratchFolder, output_file))
  allIds_text = allIds_text + itemIds_text
  allScores_text = allScores_text + scores_text
  output_file = 'pl_solution_to_submit.csv'
  print "Reading", output_file
  itemIds_pl, scores_pl = readInFile(os.path.join(scratchFolder, output_file))
  allIds_pl = allIds_pl + itemIds_pl
  allScores_pl = allScores_pl + scores_pl
  #output_file = 'solution_to_submit_pl.csv'

  Dict2rank = {}
  Dict2rank_text = {}
  Dict2rank_pl = {}
  for i in range(len(allIds_text)):
    Dict2rank_text[allIds_text[i]] = allScores_text[i]
  for i in range(len(allIds_pl)):
    Dict2rank_pl[allIds_pl[i]] = allScores_pl[i]

  for key in Dict2rank_text.keys():
    Dict2rank[key] = Dict2rank_text[key] + 0.0*(Dict2rank_pl[key] - 0.5)**3

  predfile = os.path.join(scratchFolder, 'solution_to_submit.csv')
  fp = open(predfile, 'w')
  fp.write('id\n')
  RankedList = sorted(Dict2rank.iteritems(), key=lambda x: x[1])
  for element in reversed(RankedList):
    #print element[0], element[1]
    fp.write(str(element[0])+'\n')
  fp.close()
  
  solutionfile = os.path.join(dataFolder, 'small_test_blockedids.tsv')
  
#    item_ids = processData(os.path.join(dataFolder,"avito_train.tsv"), itemsLimit=None)
#    item_ids = processData(os.path.join(dataFolder,"avito_test.tsv"), itemsLimit=None)
    
  '''If this is a Kaggle submission, we don't have the answers, so comment out next line.'''
  print 'Average Precison over all positives in test set: ', APatK.APatK(predfile, solutionfile, file_len(solutionfile))    

  logging.info("Done.")
                               
if __name__=="__main__":            
  main()            
    
    
    
