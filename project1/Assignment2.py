import numpy  as np 
import sys
import inspect
import math
import operator
from collections import defaultdict

trainfeature = "clintontrump.tweets.train"
trainlabel = "clintontrump.labels.train"
testfeature = "clintontrump.tweets.dev"
testlabel = "clintontrump.labels.dev"
stopwordsfile = "stopwords_en.txt"
vocab = defaultdict(int)
trainLFmap = []     # the label-features map for training
testLFmap = []      # the label-features map for test
stopwords = set()

def parsestopwords(filename):
    global stopwords
    file = open(filename, "r")
    for line in file.readlines():
        stopwords.add(line.strip())
    #print(len(stopwords))

def parsevocab(filename):
    file = open(filename, "r")
    countline = 0
    for line in file.readlines():
        countline += 1
        words = line.strip().split()
        for word in words:
            word = cleanup(word)
            if len(word) > 0:
                vocab[word] += 1
    

#remove word of length 1 and http links
def cleanup(word):
    if len(word) < 2 or word.startswith('http'): 
      return ""
    return word.lower()    

def advancedcleanup(word):
    if len(word) < 2 or word.startswith('http') or word in stopwords: 
      return ""
    return word.lower() 

# parse label-features map from feature file and label file
def parseLFmap(featurefile, labelfile):
    map = []
    lf = open(labelfile, "r")
    countline = 0
    for line in lf.readlines():
        countline += 1
        map.append([line.strip()])
 #   print("label file length: ",countline)
    lf.close
    countline = 0
    ff = open(featurefile, "r")
    for line in ff.readlines():
        countline += 1
        map[countline-1].append(line.strip())
 #   print("feature file length: ", countline)
    ff.close
    return map
 
def bernoullibayes(advanceWordProcess = False):

    #training 
    print("Training.....")
    clintondict = defaultdict(int)
    trumpdict = defaultdict(int)
    pclinton = 0.0  # fraction of tweets about clinton
    ptrump = 0.0    # fraction of tweets about trump
    totalclintonwords = 0.0   #total number of words about clinton
    totaltrumpwords = 0.0     #total number of words about trump
    for labelfeature in trainLFmap:
        wordset = {x.strip() for x in labelfeature[1].split()}
        # if labelfeature[0] is 'HillaryClinton'
        if len(labelfeature[0]) == 14:
            pclinton += 1.0
            for word in wordset:
                if advanceWordProcess:
                   word = advancedcleanup(word)
                else:
                   word = cleanup(word) 
                if(len(word) > 0):
                    clintondict[word] += 1
                    totalclintonwords += 1
        else:
            ptrump +=1.0
            for word in wordset:
                if advanceWordProcess:
                   word = advancedcleanup(word)
                else:
                   word = cleanup(word) 
                if(len(word) > 0):
                    trumpdict[word] += 1
                    totaltrumpwords += 1
    pclinton = pclinton/(pclinton + ptrump)
    ptrump = 1.0 - pclinton
    
    print("10 highly probable words about Clinton")
    sortedwords = sorted(clintondict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0,10):
        print(sortedwords[i])
    print("10 highly probable words about Trump")
    sortedwords = sorted(trumpdict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0,10):
        print(sortedwords[i])
    
    print("Testing.....")
    #print(pclinton)
    predcorrectcnt = 0.0
    totalpred = 0.0
    for labelfeature in testLFmap:
        totalpred += 1
        clintonloglikelihood = math.log(pclinton)
        trumploglikelihood = math.log(ptrump)
        wordset = {x.strip() for x in labelfeature[1].split()}
        for word in wordset:
            if word in clintondict:
                clintonloglikelihood += math.log((clintondict[word]*1.0+1.0)/(totalclintonwords*1.0+2.0))
            if word in trumpdict:
                trumploglikelihood += math.log((trumpdict[word]*1.0+1.0)/(totaltrumpwords*1.0 + 2.0))        
        if clintonloglikelihood > trumploglikelihood and len(labelfeature[0]) == 14:
            #print("Clinton prediction is correct")
            predcorrectcnt += 1.0
        elif clintonloglikelihood < trumploglikelihood and len(labelfeature[0]) > 14:
            #print("Trump prediction is correct")
            predcorrectcnt += 1.0
        #else:
            #print("Prediction is wrong Clinton: ", clintonloglikelihood, " Trump:", trumploglikelihood) 
    print("Prediction accuracy:")
    print(predcorrectcnt/totalpred)

               
def multinomialbayes(advanceWordProcess = False):
     #training 
    print("Training.....")
    clintondict = defaultdict(int)
    trumpdict = defaultdict(int)
    pclinton = 0.0  # fraction of tweets about clinton
    ptrump = 0.0    # fraction of tweets about trump
    totalclintonwords = 0.0   #total number of words about clinton
    totaltrumpwords = 0.0     #total number of words about trump
    for labelfeature in trainLFmap:
        # if labelfeature[0] is 'HillaryClinton'
        if len(labelfeature[0]) == 14:
            pclinton += 1.0
            for word in labelfeature[1].split():
               if advanceWordProcess:
                   word = advancedcleanup(word)
               else:
                   word = cleanup(word) 
               if(len(word) > 0):
                    clintondict[word] += 1
                    totalclintonwords += 1
        else:
            ptrump +=1.0
            for word in labelfeature[1].split():
                if advanceWordProcess:
                   word = advancedcleanup(word)
                else:
                   word = cleanup(word) 
                if(len(word) > 0):
                    trumpdict[word] += 1
                    totaltrumpwords += 1
    pclinton = pclinton/(pclinton + ptrump)
    ptrump = 1.0 - pclinton            
    
    print("10 highly probable words about Clinton")
    sortedwords = sorted(clintondict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0,10):
        print(sortedwords[i])
    print("10 highly probable words about Trump")
    sortedwords = sorted(trumpdict.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0,10):
        print(sortedwords[i])

    print("Testing.....")
    #print(pclinton)
    predcorrectcnt = 0.0
    totalpred = 0.0
    for labelfeature in testLFmap:
        totalpred += 1
        clintonloglikelihood = math.log(pclinton)
        trumploglikelihood = math.log(ptrump)
        for word in labelfeature[1].split():
            if word in clintondict:
                clintonloglikelihood += math.log((clintondict[word]*1.0 + 1.0)/(totalclintonwords*1.0+len(vocab)))
            if word in trumpdict:
                trumploglikelihood += math.log((trumpdict[word]*1.0+1.0)/(totaltrumpwords*1.0+len(vocab)))        
        if clintonloglikelihood > trumploglikelihood and len(labelfeature[0]) == 14:
            #print("Clinton prediction is correct")
            predcorrectcnt += 1.0
        elif clintonloglikelihood < trumploglikelihood and len(labelfeature[0]) > 14:
            #print("Trump prediction is correct")
            predcorrectcnt += 1.0
        #else:
            #print("Prediction is wrong Clinton: ", clintonloglikelihood, " Trump:", trumploglikelihood) 
    print("Prediction accuracy:")
    print(predcorrectcnt/totalpred)    
     
def main():
    print("Parsing vocabulary")
    parsevocab("clintontrump.tweets.train")
    #annotation.append(['Hillary'])
    #annotation[0].append('She is good')  
    #print(annotation[0][1])
    global trainLFmap
    global testLFmap
    print("Parsing training files.....")
    trainLFmap = parseLFmap(trainfeature,trainlabel)
    print("Parsing testing file.....")
    testLFmap = parseLFmap(testfeature, testlabel)
    #print(testFLmap)
    print("Applying Bernoulli Model.....")
    bernoullibayes()
    print("Applying Multinomial Model.....")
    multinomialbayes()
    print("Importing stop-words")
    parsestopwords(stopwordsfile)
    print("Applying Bernoulli Model with stopwords removal.....")
    bernoullibayes(advanceWordProcess = True)
    print("Applying Multinomial Model with stopwords removal.....")
    multinomialbayes(advanceWordProcess = True)
    #print(stopwords)
    #test = 'abc ab abc ab'
    #set = {word.strip() for word in test.split()}
    return 0    























if __name__ == '__main__':
    main();