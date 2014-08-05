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
import APatK
import WordStatsModule as ws

'''change dataFolder to the correct local path'''
dataFolder = "/Users/adamkalman/Desktop/CDIPS/Data"
scratchFolder = "/Users/adamkalman/Desktop/CDIPS/interdata"

stopwords= frozenset(word.decode('utf-8') for word in nltk.corpus.stopwords.words("russian") if word!="не")    
stemmer = SnowballStemmer('russian')
engChars = [ord(char) for char in u"cCyoOBaAKpPeE"]
rusChars = [ord(char) for char in u"сСуоОВаАКрРеЕ"]
eng_rusTranslateTable = dict(zip(engChars, rusChars))
rus_engTranslateTable = dict(zip(rusChars, engChars))

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)
        
def correctWord (w):
    """ Corrects word by replacing characters with written similarly depending on which language the word. 
        Fraudsters use this technique to avoid detection by anti-fraud algorithms."""

    if len(re.findall(ur"[а-я]",w))>len(re.findall(ur"[a-z]",w)):
        return w.translate(eng_rusTranslateTable)
    else:
        return w.translate(rus_engTranslateTable)

def getItems(fileName, itemsLimit=None):
    """ Reads data file. """
    
    with open(os.path.join(scratchFolder, fileName), 'rU') as items_fd:
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
#            for featureName, featureValue in item.iteritems():
#                if featureName == "itemid" : print
#                print featureName, featureValue
            item = {featureName: featureValue.decode('utf-8') for featureName, featureValue in item.iteritems()}
            if not itemsLimit or i in sampleIndexes:
                itemNum += 1
                yield itemNum, item
                
    
def getWords(text, stemmRequired = False, correctWordRequired = False):
    """ Splits the text into words, discards stop words and applies stemmer. 
    Parameters
    ----------
    text : str - initial string
    stemmRequired : bool - flag whether stemming required
    correctWordRequired : bool - flag whether correction of words required     
    """

    cleanText = re.sub(u'[^a-zа-я0-9]', ' ', text.lower())
    if correctWordRequired:
        words = [correctWord(w) if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(correctWord(w)) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    else:
        words = [w if not stemmRequired or re.search("[0-9a-z]", w) else stemmer.stem(w) for w in cleanText.split() if len(w)>1 and w not in stopwords]
    
    return words

def getAdlength(text):
  textlist = text.split(' ')
  return len(textlist)


def processData(fileName, featureIndexes={}, itemsLimit=None):
  """ Processing data. """
  processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
  logging.info(processMessage+"...")

  wordCounts = defaultdict(lambda: 0)
  goodWordCounts = defaultdict(lambda: 0)
  illegalWordCounts = defaultdict(lambda: 0)
  targets = []
  item_ids = []
  ad_ids = []
  row = []
  col = []
  cur_row = 0
  totalwordsgood = 0
  totalwordsillegal = 0
  totalwords = 0
  adlengthgood = []
  adlengthillegal = []
  adlength = []
  alladlength = []
  prices = []
  isblocked = []
  
  for processedCnt, item in getItems(fileName, itemsLimit):
   
    if not featureIndexes:
      alladlength.append(getAdlength(item["title"]+" "+item["description"])) 
      ad_ids.append(item["itemid"])
      prices.append(item["price"])
      if "is_blocked" in item:
        isblocked.append(item["is_blocked"])
        if item["is_blocked"] == u'1':
          wordsinadillegal = getAdlength(item["title"]+" "+item["description"])
          totalwordsillegal += wordsinadillegal
        else:
          wordsinadgood = getAdlength(item["title"]+" "+item["description"])
          totalwordsgood += wordsinadgood
    else:
      if not "is_blocked" in item:
        alladlength.append(getAdlength(item["title"]+" "+item["description"])) 
        ad_ids.append(item["itemid"])
        prices.append(item["price"])
        wordsinad = getAdlength(item["title"]+" "+item["description"])
        totalwords += wordsinad

    
    for word in getWords(item["title"]+" "+item["description"], stemmRequired = False, correctWordRequired = False):
      if not featureIndexes:
        wordCounts[word] += 1
        if "is_blocked" in item:
          if item["is_blocked"] == u'1':
            illegalWordCounts[word] += 1 
          else:
            goodWordCounts[word] += 1 

      else:
        if word in featureIndexes:
          col.append(featureIndexes[word])
          row.append(cur_row)
    
    '''this next 4 lines is supposed to add a column if the item is free or not
    if featureIndexes:
      if item["price"] < 1:
        col.append(len(featureIndexes))
        row.append(cur_row)
    '''
    
    if featureIndexes:
      cur_row += 1
      if "is_blocked" in item:
        targets.append(int(item["is_blocked"]))
      else:
        adlength.append(wordsinad)
      item_ids.append(int(item["itemid"]))
    else:
      if "is_blocked" in item:
        if item["is_blocked"] == u'1': 
          adlengthillegal.append(wordsinadillegal)
          adlength.append(wordsinadillegal)
        else:
          adlengthgood.append(wordsinadgood)
          adlength.append(wordsinadgood)
                    
    if processedCnt%1000 == 0:                 
      logging.debug(processMessage+": "+str(processedCnt)+" items done")
 
  print totalwordsgood, totalwordsillegal 
  if not featureIndexes:
    index = 0
    totalwordsgood = ws.getTotalWords(goodWordCounts)
    totalwordsillegal = ws.getTotalWords(illegalWordCounts)
    print totalwordsgood, totalwordsillegal

    wordfreqsgood = []
    wordfreqsillegal = []
    no = 0
    fp = open('wordfreqs.txt','a')
    for word, count in wordCounts.iteritems():
      if count>=3:
        no += 1
        if no%1000 == 0: print no
        featureIndexes[word]=index
        if goodWordCounts.has_key(word):
          wordfreqsgood.append(goodWordCounts[word]*1./(totalwordsgood))
        else:
          wordfreqsgood.append(0)
#          print word, "ILLEGAL", count
        if illegalWordCounts.has_key(word):
          wordfreqsillegal.append(illegalWordCounts[word]*1./(totalwordsillegal))
        else:
          wordfreqsillegal.append(0)
        if totalwordsgood == 0: totalwordsgood = 1
        if totalwordsillegal == 0: totalwordsillegal = 1
        fp.write(str(featureIndexes[word])+"  "+word.encode('utf-8')+"  "+str(goodWordCounts[word]*1./totalwordsgood)+"  "+str(illegalWordCounts[word]*1./totalwordsillegal)+"\n")
        index += 1
    fp.close()
    print np.mean(wordfreqsgood), "+-", np.std(wordfreqsgood)
    print np.mean(wordfreqsillegal), "+-", np.std(wordfreqsillegal)
    print np.mean(adlengthgood), "+-", np.std(adlengthgood), len(adlengthgood)
    print np.mean(adlengthillegal), "+-", np.std(adlengthillegal), len(adlengthillegal)
    ws.analyzeadlength(adlengthgood, "adlength_normal.txt")
    ws.analyzeadlength(adlengthillegal, "adlength_illicit.txt")
    ws.analyzeadlength(adlength, "adlength_train.txt")
    fp = open(os.path.join(scratchFolder, fileName[:-10]+"_alladlength_train.csv"), "w")
    fp.write("itemid,adlength,price,isblocked\n")
    for i in range(len(alladlength)):
      fp.write(str(ad_ids[i])+" , "+str(alladlength[i])+" , "+str(prices[i])+" , "+str(isblocked[i])+"\n")
    fp.close()
    # Store the total number of words in good and illegal ads at the end of the word frequency lists
    wordfreqsgood.append(totalwordsgood)
    wordfreqsillegal.append(totalwordsillegal)
    if wordfreqsgood or wordfreqsillegal:
      return featureIndexes, wordfreqsgood, wordfreqsillegal
    else:
      return featureIndexes
  else:
    if "is_blocked" not in item: 
      ws.analyzeadlength(adlength, "adlength_test.txt")
      fp = open(os.path.join(scratchFolder, fileName[:-9]+"_alladlength_test.csv"), "w")
      fp.write("itemid,adlength,price\n")
      for i in range(len(alladlength)):
        fp.write(str(ad_ids[i])+" , "+str(alladlength[i])+" , "+str(prices[i])+"\n")
      fp.close()
      
    features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
    if targets:
      return features, targets, item_ids
    else:
      return features, item_ids


def makeScalingMatrix(wordfreqsgood, wordfreqsillegal):
    freq = []
    totalwordsgood = wordfreqsgood.pop()
    totalwordsillegal = wordfreqsillegal.pop()
    for n in range(len(wordfreqsgood)):
      if wordfreqsgood[n] < wordfreqsillegal[n]:
        freq.append(((totalwordsgood*wordfreqsgood[n]+2)/(totalwordsgood*wordfreqsillegal[n]+2)))
      else:
        freq.append(((totalwordsillegal*wordfreqsillegal[n]+2)/(totalwordsillegal*wordfreqsgood[n]+2)))
    freq = [i for i in freq]
    return freq

# get number of entries in a file (number of lines - 1 for header)
def file_len(filename):
  with open(filename) as f:
    for i, l in enumerate(f):
      pass
  return i


def main():
  '''These global declarations only exist so the variables can be played with at the command line. Feel free to comment or uncomment them.'''
  global featureIndexes #dict with 126229 entries {unicode word: index of that word}
  global trainFeatures #trainSize by 126229 sparse matrix with feature data for each training example
  global trainTargets #a 0-1 list, len=size of training set, of human-provided answers to the training set
  global trainItemIds #a list, len=size of training set, of id numbers for the ads in the training set
  global testFeatures #1351242 by 126229 sparse matrix with feature data for each example in provided (fixed) test set
  global testItemIds #a list, len=1351242, of id numbers for the ads in the provided (fixed) test set 
  global predicted_scores #a list, len=size of test set, of predicted probabilities of is_blocked

  fp1 = open('wordfreqs.txt','w')
  fp1.close()
  fp2 = open('adlength_normal.txt','w')
  fp2.close()
  fp3 = open('adlength_illicit.txt','w')
  fp3.close()

  featureIndexes = {}
  for scratchfile in os.listdir(scratchFolder):
#    print scratchfile, scratchfile[-9:]
    if scratchfile[-9:] == "train.tsv":
      trainfile = scratchfile
      testfile = scratchfile[:-9]+"test.tsv"
      '''If this is a Kaggle submission, comment out the next line'''
#      solutionfile = scratchfile[:-9]+"test_solution.tsv"
    else:
      continue
    
    print trainfile, testfile
    if featureIndexes:
      featureIndexes.clear()
    
    '''uncomment the first 3 and last 2 lines here to rebuild train_data.pkl. It takes a few minutes.'''
    logging.info('Building train_data.pkl from provided training and test tsv files...')
    featureIndexes, wordfreqsgood, wordfreqsillegal = processData(os.path.join(scratchFolder,trainfile), itemsLimit=None)
    trainFeatures, trainTargets, trainItemIds=processData(os.path.join(scratchFolder,trainfile), featureIndexes, itemsLimit=None)
#    freq = makeScalingMatrix(wordfreqsgood, wordfreqsillegal)
#    freqdiag = sp.dia_matrix((freq, [0]), shape=(len(freq), len(freq)))
#    trainFeatures = trainFeatures*freqdiag
    testFeatures, testItemIds=processData(os.path.join(scratchFolder,testfile), featureIndexes)
    joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(scratchFolder,scratchfile[:-9]+"train_data.pkl"))

    logging.info("Loading Data from 'train_data.pkl'...")
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(scratchFolder,scratchfile[:-9]+"train_data.pkl"))
    
    logging.info("Feature preparation done, fitting model...")
    clf = SGDClassifier(    loss="log", 
                            penalty="l2", 
                            alpha=1e-4, 
                            class_weight="auto")
    clf.fit(trainFeatures,trainTargets)
    
    logging.info("Computing predictions for submission...")
    predicted_scores = clf.predict_proba(testFeatures).T[1] 
    output_file = "solution_to_submit_"+scratchfile[:-10]+".csv" #has IDs with probabilities, in order
    predfile = "predictions_"+scratchfile[:-10]+".csv" #just IDs, in order
    logging.info("Writing output to %s" % output_file)
    f = open(os.path.join(scratchFolder,output_file), "w")
#    f.write("id\n")
#    pred_score_prev = 1.
#    n = 1
#    threshold = 0.9
    g = open(os.path.join(scratchFolder,predfile), "w")
    g.write("id\n")
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%15d, %30.20f\n" % (item_id, pred_score))
        g.write(str(item_id)+"\n")
#        if pred_score < threshold and pred_score_prev > threshold:
#          print threshold, ' : ', n
#          threshold -= 0.1 
#        pred_score_prev = pred_score
#        n += 1
    f.close()
    g.close()
#    print 'total: ', n
    
    '''If this is a Kaggle submission, we don't have the answers, so comment out next line.'''
#    print 'Average Precison over all positives in test set: ', APatK.APatK(predfile, solutionfile, file_len(os.path.join(scratchFolder,solutionfile)))

    logging.info("Done.")
                               
if __name__=="__main__":            
    main()            
    
