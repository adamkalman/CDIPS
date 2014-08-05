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
import itertools
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
import math
import string
import APatK
import WordStatsModule as ws

'''change dataFolder to the correct local path'''
dataFolder = "/Users/adamkalman/Desktop/CDIPS/Data"
scratchFolder = "/Users/adamkalman/Desktop/CDIPS/interdata"
trainFile = "avito_train.tsv"
testFile = "avito_test.tsv"
#solutionFile = "small_test_solution.tsv"

def getItems(fileName, itemsLimit=None):
    """ Reads data file. """
    
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info("Sampling...")
        if itemsLimit:
            countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
            numItems = 0
            for row in countReader:
                numItems += 1
            items_fd.seek(0)        
            rnd.seed(0)
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))
            
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
            if not itemsLimit or i in sampleIndexes:
                itemNum += 1
                yield itemNum, item
                

def processData(whichfile, attrs2use = None):
  """ Get ads from whichfile, sort them by category, and write to separate files """

  cutoff_number = 100
  cutoff_freq = 0.05

  itemsLimit = None
  symbols = (u"абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-№–, ", u"abvgdeejzijklmnoprstufhzcss_y_euaABVGDEEJZIJKLMNOPRSTUFHZCSS_Y_EUA_____")
  tr = dict( [ (ord(a), ord(b)) for (a, b) in zip(*symbols) ] )
  
  item_ids = []
  categories = {}
#  categoriesSol = {}
  if not attrs2use:
    attributes = {}
  else:
    attributes = attrs2use
  goodAttrFiles = {}
  attributes[''] = [cutoff_number+1, cutoff_number+1]
  itemAttrs = []
    
  print whichfile
  
  if whichfile[-11:] == trainFile[-11:]:
    category_list = ["itemid", "category", "subcategory", "title", "description", "attrs", "price", "is_proved", "is_blocked", "phones_cnt", "emails_cnt", "urls_cnt", "close_hours"]
  elif whichfile[-10:] == testFile[-10:]:
    category_list = ["itemid", "category", "subcategory", "title", "description", "attrs", "price", "phones_cnt", "emails_cnt", "urls_cnt"]
#    solFile = open(os.path.join(dataFolder,solutionFile),'r')
#    solList = list(csv.reader(solFile, delimiter = '\t'))
    #solList = list(itertools.chain.from_iterable(solList))
  else:
    print "error"

  for processedCnt, item in getItems(whichfile, itemsLimit): 

    if len(item['attrs'])>0: 
      towrite = item['attrs']#.encode('utf-8') 
      latin = towrite.translate(tr)
      match = re.search(':"["\w\.,/-]+"', latin)
      res = match.group()
      match = re.search('[\w\.,/-]+', res)
      res = match.group()
      res = res+"_"+item["subcategory"].translate(tr)+"_"+item["category"].translate(tr)
    else:
      res = item["subcategory"].translate(tr)+"_"+item["category"].translate(tr)

    itemAttrs.append(res)

    if not attrs2use:
      if not res in attributes.keys():
        attributes[res] = [1,0]
      else:
        attributes[res][0] += 1
      if "is_blocked" in item:
        if item["is_blocked"] == '1':
          attributes[res][1] += 1
    else:
      if not res in attributes.keys():
        attributes[res] = [cutoff_number+1,cutoff_number+1]
      

    
    if processedCnt%10000 == 0:                 
      print processedCnt, 'ads processed'           


  adnumber = 0
  for processedCnt, item in getItems(whichfile, itemsLimit): 
    if len(item.keys()) != len(category_list):
      print 'Corrupt ad\n'
      continue
  
    if not item["category"] in categories.keys():
      writefile = item["category"]
      writefile = writefile.translate(tr)
      if whichfile[-11:] == trainFile[-11:]:
        fp = open("interdata/"+writefile+"_train.tsv", "w")
        fp1 = open("interdata/good_"+writefile+"_traingood.tsv", "w")
      else:
        fp = open("interdata/"+writefile+"_test.tsv", "w")
        fp1 = open("interdata/good_"+writefile+"_testgood.tsv", "w")
#        fpSol = open("interdata/"+writefile+"_test_solution.tsv", "w")
#        categoriesSol[item["category"]] = fpSol
      categories[item["category"]] = fp
      goodAttrFiles[item["category"]] = fp1
      i = 0
      for element in category_list:
        if i < len(category_list)-1:
          categories[item["category"]].write(element+"\t")
          i += 1
        else:
          categories[item["category"]].write(element+"\n")
#      if whichfile[-10:] == testFile[-10:]:
#        categoriesSol[item["category"]].write("id\n")
    i = 0

    if (attributes[itemAttrs[adnumber]][1]*1./attributes[itemAttrs[adnumber]][0] >= cutoff_freq) or (attributes[itemAttrs[adnumber]][0] < cutoff_number):
      for element in category_list:
        if u'\r' in item[element]:
          item[element] = ''.join(item[element].split('\r'))
        if u'\t' in item[element]:
          item[element] = ' '.join(item[element].split('\t'))
        
        towrite = item[element].encode('utf-8')     
        
        if len(towrite) != 0:
          while (towrite[0] == '"') or (towrite[0] == ' ') or (towrite[0] == "'"):
            if len(towrite) > 1:
              towrite = towrite[1:]
            else:
              towrite = ''
              break
        if i < len(category_list)-1:
          categories[item["category"]].write(towrite+"\t")
          i += 1
        else:
          categories[item["category"]].write(towrite+"\n")
          if i != len(category_list)-1: print i

#      if "Licnye_vesi" in itemAttrs[adnumber]:
#        print 'yes'

    else:
      goodAttrFiles[item["category"]].write(item["itemid"]+'\n')
      
    
#    if whichfile[-10:] == testFile[-10:] and solList[processedCnt][1] == '1':
#      categoriesSol[item["category"]].write(solList[processedCnt][0]+"\n")
    
    if processedCnt%10000 == 0:                 
      print processedCnt, 'ads processed'       
    adnumber += 1     

  for key in categories.keys():
    categories[key].close()
  for key in goodAttrFiles.keys():
    goodAttrFiles[key].close()
  
  if "train" in whichfile:
    for key in attributes.keys():
      print key, attributes[key], attributes[key][1]*1./attributes[key][0]
   
  return item_ids, attributes

def main():
    
    item_ids, attrs2use = processData(os.path.join(dataFolder,trainFile))
    item_ids, attrs2use = processData(os.path.join(dataFolder,testFile), attrs2use)
    
    logging.info("Done.")
                               
if __name__=="__main__":            
    main()            
    
    
    
