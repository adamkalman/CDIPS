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
import APatK

'''change dataFolder to the correct local path'''
dataFolder = "/Users/adamkalman/Desktop/CDIPS/Data"

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
    
    with open(os.path.join(dataFolder, fileName)) as items_fd:
        logging.info("Sampling...")
        if itemsLimit:
            countReader = csv.DictReader(items_fd, delimiter='\t', quotechar='"')
            numItems = 0
            for row in countReader:
                numItems += 1
            items_fd.seek(0)        
            rnd.seed()
            sampleIndexes = set(rnd.sample(range(numItems),itemsLimit))
            
        logging.info("Sampling done. Reading data...")
        itemReader=csv.DictReader(items_fd, delimiter='\t', quotechar = '"')
        itemNum = 0
        for i, item in enumerate(itemReader):
            item = {featureName:featureValue.decode('utf-8') for featureName,featureValue in item.iteritems()}
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

def processData(fileName, featureIndexes={}, itemsLimit=None):
    """ Processing data. """
    processMessage = ("Generate features for " if featureIndexes else "Generate features dict from ")+os.path.basename(fileName)
    logging.info(processMessage+"...")

    wordCounts = defaultdict(lambda: 0)
    targets = []
    item_ids = []
    row = []
    col = []
    cur_row = 0
    for processedCnt, item in getItems(fileName, itemsLimit):
        #col = []
        for word in getWords(item["title"]+" "+item["description"], stemmRequired = False, correctWordRequired = False):
            if not featureIndexes:
                wordCounts[word] += 1
            else:
                if word in featureIndexes:
                    col.append(featureIndexes[word])
                    row.append(cur_row)
        
        if featureIndexes:
            cur_row += 1
            if "is_blocked" in item:
                targets.append(int(item["is_blocked"]))
            item_ids.append(int(item["itemid"]))
                    
        if processedCnt%1000 == 0:                 
            logging.debug(processMessage+": "+str(processedCnt)+" items done")
                
    if not featureIndexes:
        index = 0
        for word, count in wordCounts.iteritems():
            if count>=3:
                featureIndexes[word]=index
                index += 1
                
        return featureIndexes
    else:
        features = sp.csr_matrix((np.ones(len(row)),(row,col)), shape=(cur_row, len(featureIndexes)), dtype=np.float64)
        if targets:
            return features, targets, item_ids
        else:
            return features, item_ids

def main():
    '''These global declarations only exist so the variables can be played with at the command line. Feel free to comment or uncomment them.'''
    global featureIndexes #dict with 126229 entries {unicode word: frequency}
    global trainFeatures #trainSize by 126229 sparse matrix with feature data for each training example
    global trainTargets #a 0-1 list, len=trainSize, of human-provided answers to the training set
    global trainItemIds #a list, len=trainSize, of id numbers for the ads in the training set
    global crossvalFeatures #crossvalSize by 126229 sparse matrix with feature data for each cross-validation example
    global crossvalTargets #a 0-1 list, len=crossvalSize, of human-provided answers to the cross-validation set
    global crossvalItemIds #a list, len=crossvalSize, of id numbers for the ads in the cross-validation set
    global testFeatures #1351242 by 126229 sparse matrix with feature data for each example in provided (fixed) test set
    global testItemIds #a list, len=1351242, of id numbers for the ads in the provided (fixed) test set 
    global predicted_scores #a list, len=crossvalSize, of predicted probabilities of is_blocked

    #sum of these two can probably be up to about 4 million
    trainSize = 300000 
    crossvalSize = 100000
    
    '''uncomment the next 5 lines to rebuild train_data.pkl. It takes a few minutes.'''
    #logging.info('Building train_data.pkl from provided training and test tsv files...')
    #featureIndexes = processData(os.path.join(dataFolder,"avito_train.tsv"), itemsLimit=trainSize+crossvalSize)
    #trainFeatures,trainTargets, trainItemIds=processData(os.path.join(dataFolder,"avito_train.tsv"), featureIndexes, itemsLimit=trainSize+crossvalSize)
    #testFeatures, testItemIds=processData(os.path.join(dataFolder,"avito_test.tsv"), featureIndexes)
    #joblib.dump((trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds), os.path.join(dataFolder,"train_data.pkl"))
    logging.info("Loading Data from 'train_data.pkl'...")
    trainFeatures, trainTargets, trainItemIds, testFeatures, testItemIds = joblib.load(os.path.join(dataFolder,"train_data.pkl"))
    
    logging.info('Splitting training data into training set and cross-validation set...')
    crossvalFeatures = trainFeatures[:crossvalSize,:]
    crossvalTargets = trainTargets[:crossvalSize]
    crossvalItemIds = trainItemIds[:crossvalSize]
    trainFeatures = trainFeatures[crossvalSize:,:]
    trainTargets = trainTargets[crossvalSize:]
    trainItemIds = trainItemIds[crossvalSize:]
    
    logging.info("Feature preparation done, fitting model...")
    clf = SGDClassifier(    loss="log", 
                            penalty="l2", 
                            alpha=1e-4, 
                            class_weight="auto")
    clf.fit(trainFeatures,trainTargets)

    logging.info("Predicting results for cross-validation set...")
    
    predicted_scores = clf.predict_proba(crossvalFeatures).T[1] 
    '''blahblahmodel.predict_proba(foobarFeatures) outputs a (crossvalSize,2) matrix, with the probabilities of a yes or no in the two columns.
        Note that the second column is just 1 minus the first column. We set predicted_scores as the first column.'''
    
    logging.info("Writing cross-validation predictions to cvpredictions.csv")
    f = open('cvpredictions.csv','w')
    f.write("id\n") #header
    for pred_score, item_id in sorted(zip(predicted_scores, crossvalItemIds), reverse = True):
        f.write("%d\n" % (item_id))
    f.close()
    
    logging.info('Writing IDs of all confirmed blocked ads in cross-validation set to cvsolution.csv')
    f = open('cvsolution.csv','w')
    f.write("id\n") #header
    for idx in xrange(len(crossvalTargets)):
        if crossvalTargets[idx]:
            f.write(str(crossvalItemIds[idx])+"\n")
    f.close()
    
    print 'Average Precison over all cross-validation positives: ', APatK.APatK( "cvpredictions.csv", "cvsolution.csv", sum(crossvalTargets))

    
    '''uncomment this section when we are ready to submit to Kaggle
    logging.info("Computing predictions for final submission...")
    predicted_scores = clf.predict_proba(testFeatures).T[1] 
    output_file = "solution_to_submit.csv"
    logging.info("Writing submission to %s" % output_file)
    f = open(os.path.join(dataFolder,output_file), "w")
    f.write("id\n")
    for pred_score, item_id in sorted(zip(predicted_scores, testItemIds), reverse = True):
        f.write("%d\n" % (item_id))
    f.close()
    '''
    
    logging.info("Done.")
                               
if __name__=="__main__":            
    main()            
    
    
    
