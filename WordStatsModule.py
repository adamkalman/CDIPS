#!/usr/bin/env python -tt
# coding: utf-8
"""
Word statistics functions
"""

def analyzeadlength(adlength, filename):
  binskeys = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60, 65, 70, 75, 80, 90, 100, 120, 150, 200, "infty"]
  bins = dict.fromkeys(binskeys, 0)
  for el in adlength:
    key = el
    while (key not in bins.keys()) and (key <= 200):
      key += 1
    if key > 200:
      bins["infty"] += 1./len(adlength)
    else:
      bins[key] += 1./len(adlength)
  fp = open(filename, "a")
  for key in sorted(bins.keys()):
    fp.write(str(key)+"     "+str(bins[key])+"\n")
  fp.close()

def getTotalWords(wordCounts):
  totalwords = 0
  for word, count in wordCounts.iteritems():
    if count>=3:
      totalwords += wordCounts[word]
  return totalwords

