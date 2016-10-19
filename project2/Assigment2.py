import numpy  as np 
import sys
import inspect
import math
import operator
from collections import defaultdict
import os.path

traintweet = "clintontrump.tweets.train"
trainlabel = "clintontrump.labels.train"
testtweet = "clintontrump.tweets.dev"
testlabel = "clintontrump.labels.dev"
stopwordsfile = "stopwords_en.txt"
vocab = defaultdict(int)

trainLabelTweetList = []     # the label-tweet map for training
testLabelTweetList = []      # the label-features map for test
stopwords = set()

def parseStopwords(filename):
    global stopwords
    file = open(filename, "r")
    for line in file.readlines():
        stopwords.add(line.strip())
    #print(len(stopwords))

def parseVocabulary(filename, advanceWordProcess=False):
    file = open(filename, "r")
    countline = 0
    for line in file.readlines():
        countline += 1
        words = line.strip().split()
        for word in words:
            if advanceWordProcess:
                word = advancedCleanup(word)
            else:
                word = cleanup(word)
            if len(word) > 0:
                vocab[word] += 1



#convert a word to lower case
def cleanup(word):
    return word.lower()    


# remove words of length < 2, httplink and stopwords
def advancedCleanup(word):
    word = word.lower()
    if word in stopwords:
      return ""
    #if len(word) < 2 or word.startswith('http'):
    #  return ""
    return word

# parse label-tweet list from a tweet file and a label file
def parseLabelTweetList(tweetfile, labelfile):
    map = []
    
    lfExists = os.path.isfile(labelfile) 

    if(lfExists):
        lf = open(labelfile, "r")
        countline = 0
        for line in lf.readlines():
            countline += 1
            map.append([line.strip()])
        lf.close
    else:
        print "\n\n\n --------- WARNING --------- \n\n"
        print "can't find ",labelfile,"\n Defaulting to all realDonaldTrump\n\n\n"
    
    countline = 0
    ff = open(tweetfile, "r")
    for line in ff.readlines():
        countline += 1

        if(lfExists == False):
            map.append(["realDonaldTrump"])

        map[countline - 1].append(line.strip())
 #   print("feature file length: ", countline)
    ff.close


    return map
 





def leastUsedStopWords(topCount = 1000):
    
    # the number of times any word appears in clintons tweets
    clintonWordCounts = defaultdict(int)

    # the number of times any word appears in clintons tweets
    trumpWordCounts = defaultdict(int)
    
    # the number of tweet in the training about Clinton
    clintonTweetCount = 0.0  
    # the number of tweet in the training about Trump
    trumpTweetCount = 0.0    
    
    for labelTweet in trainLabelTweetList:
        label = labelTweet[0]
        tweet = labelTweet[1]
        
        tweetWordSet = {x.strip() for x in tweet.split()}

        if label == "HillaryClinton":

            clintonTweetCount += 1.0
            for word in tweetWordSet:
                word = cleanup(word) 

                if len(word) > 0:
                   clintonWordCounts[word] += 1
        else:

            trumpTweetCount +=1.0
            for word in tweetWordSet:
                word = cleanup(word) 
                if(len(word) > 0):
                    trumpWordCounts[word] += 1

    sortedwords1 = sorted(clintonWordCounts.items(), key=operator.itemgetter(1), reverse=True)
    sortedwords2 = sorted(trumpWordCounts.items(), key=operator.itemgetter(1), reverse=True)
     

    topWords1 = sortedwords1[0:min(topCount, len(sortedwords1))]
    topWords2 = sortedwords2[0:min(topCount, len(sortedwords1))]
    
    topSet = set()
    
    for item in topWords1:
        topSet.add(item[0])
    for item in topWords2:
        topSet.add(item[0])

    fullSet = set()
    
    for item in sortedwords1:
        fullSet.add(item[0])
    for item in sortedwords2:
        fullSet.add(item[0])

    print "fullset  ",len(fullSet)

    global stopwords
    stopwords = fullSet.difference(topSet)

    print len(fullSet) - len(topSet), len(stopwords), len(topSet)




def bernoulliBayes(verbose=True, advanceWordProcess=False, prior=1.0, outputFile = "BernoulliResults.txt"):
    """
    Training the classifier with Bernoulli Bayes model
    
    Parameters:
        verbose : True/False 
                if true will print out training information
                if false only print out the prediction accuracy
        advancedWordProcess : True/False
                if true will allow link and stopword removal in the dictionaray
                if false will keep everything
        prior : float
                the prior of the model
    """

    global trainLabelTweetList
    global testLabelTweetList
    #training
    if verbose:
        print("Training.....")

    # the number of times any word appears in clintons tweets
    clintonWordCounts = defaultdict(int)

    # the number of times any word appears in clintons tweets
    trumpWordCounts = defaultdict(int)
    
    # the number of tweet in the training about Clinton
    clintonTweetCount = 0.0  
    # the number of tweet in the training about Trump
    trumpTweetCount = 0.0    
    
    for labelTweet in trainLabelTweetList:
        label = labelTweet[0]
        tweet = labelTweet[1]
        
        tweetWordSet = {x.strip() for x in tweet.split()}

        if label == "HillaryClinton":

            clintonTweetCount += 1.0
            for word in tweetWordSet:
                if advanceWordProcess:
                   word = advancedCleanup(word)
                else:
                   word = cleanup(word) 
                if len(word) > 0:
                   clintonWordCounts[word] += 1
        else:

            trumpTweetCount +=1.0
            for word in tweetWordSet:
                if advanceWordProcess:
                   word = advancedCleanup(word)
                else:
                   word = cleanup(word) 
                if(len(word) > 0):
                    trumpWordCounts[word] += 1

    # the probability of seeing a tweet about Clinton
    pClinton = clintonTweetCount / (clintonTweetCount + trumpTweetCount)  
    # the probability of seeing a tweet about Trump
    pTrump = 1.0 - pClinton                                            
   
   
    # Clinton dictionary
    clintonWordSet = set(clintonWordCounts.keys())                      
    # Trump dictionary
    trumpWordSet = set(trumpWordCounts.keys())                                
    
    # words in Clinton dictionary but not in Trump dictionary
    clintonUniqueWordSet = clintonWordSet.difference(trumpWordSet)        
    # words in Trump dictionary but not in Clinton dictionary
    trumpUniqueWordSet = trumpWordSet.difference(clintonWordSet)          
    

    # the dictionary in which a key is a word and a value is the probability
    # that the word appears in Clinton tweets
    clintonWordProb = dict()   
    for word, wordCount in clintonWordCounts.iteritems():
        #          wordCount + prior
        # Pr(w) = ------------------------
        #          #tweets + 2 * prior
        p = (wordCount * 1.0 + prior) / (clintonTweetCount * 1.0 + 2.0 * prior)
        clintonWordProb.update({word : p})


    # for all words that are unique to trump, give hillary the prior prob.
    for word in trumpUniqueWordSet:
        #             prior
        # Pr(w) = ------------------------
        #          #tweets + 2 * prior
        clintonWordProb.update({word : prior / (clintonTweetCount * 1.0 + 2.0 * prior)})
    


    # the dictionary in which a key is a word and a value is the probability
    # that the word appears in Trump tweets
    trumpWordProb = dict()   
    for word, wordCount in trumpWordCounts.iteritems():
        #            wordCount + prior
        # Pr(w) = ------------------------
        #           #tweets + 2 * prior
        trumpWordProb.update({word : (wordCount * 1.0 + prior) / (trumpTweetCount * 1.0 + 2.0 * prior)})

    for word in clintonUniqueWordSet:
        #             prior
        # Pr(w) = ------------------------
        #          #tweets + 2 * prior
        trumpWordProb.update({word : prior / (trumpTweetCount * 1.0 + 2.0 * prior)})


    if verbose:
        print("10 highly probable words about Clinton")
        sortedwords = sorted(clintonWordCounts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(0,min(len(sortedwords),10)):
            print(sortedwords[i])
        print("10 highly probable words about Trump")
        sortedwords = sorted(trumpWordCounts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(0,min(len(sortedwords),10)):
            print(sortedwords[i])
        print("Testing.....")










        
    if outputFile != "":
        outfile = open(outputFile,"w")
        
    clintonCorrect = 0
    clintonIncorrect = 0
    trumpCorrect = 0
    trumpIncorrect = 0

    # the number of correct predictions
    predCorrectCount = 0.0    
    # the total number of predictions
    totalpred = 1.0 * len(testLabelTweetList)         

    cnt = 0
    
    # compute the Laplace Smoothing factor for both classes
    logClintonLaplaceSmooth = math.log(prior / (clintonTweetCount * 1.0 + 2.0 * prior))
    logTrumpLaplaceSmooth = math.log(prior / (trumpTweetCount * 1.0 + 2.0 * prior)) 
   
    for labelTweet in testLabelTweetList:
        label = labelTweet[0]
        tweet = labelTweet[1]

        #
        # l(class | tweet)
        #      = log(Pr(class)
        #      + \sum_{w in tweet} log( Pr(w | class))
        #      + \sum_{w !  in tweet} log( 1 - Pr(w | class))
        #
        clintonloglikelihood = math.log(pClinton)
        trumploglikelihood = math.log(pTrump)


        tweetWordSet = {x.strip() for x in tweet.split()}
        for word in tweetWordSet:
            if len(word) > 0:
                if word in clintonWordProb:
                    clintonloglikelihood += math.log(clintonWordProb[word])
                else:
                    # this is a word that was not in the training set
                   clintonloglikelihood += logClintonLaplaceSmooth

                if word in trumpWordProb:
                    trumploglikelihood += math.log(trumpWordProb[word])
                else:
                    # this is a word that was not in the training set
                    trumploglikelihood += logTrumpLaplaceSmooth 
               
        for word in  clintonWordProb :
            if word not in tweetWordSet:
               clintonloglikelihood += math.log(1 - clintonWordProb[word])

        for word in  trumpWordProb:
            if word not in tweetWordSet:
               trumploglikelihood += math.log(1 - trumpWordProb[word])


        if outfile != None:
            if clintonloglikelihood > trumploglikelihood:
                outfile.write("HillaryClinton\n")
            else:
                outfile.write("realDonaldTrump\n")

        if label == "HillaryClinton":
            #print("Clinton prediction is correct")
            if clintonloglikelihood > trumploglikelihood:
                predCorrectCount += 1.0
                clintonCorrect += 1
            else:
                clintonIncorrect += 1
        else:

            if clintonloglikelihood < trumploglikelihood:
                predCorrectCount += 1.0
                trumpCorrect += 1
            else:
                trumpIncorrect += 1
            #print("Trump prediction is correct")


        cnt += 1
        if verbose and (cnt % 50) == 0:
            print('finish ' + str(cnt) + ' predictions')

    print("Prediction accuracy:")
    print(predCorrectCount / totalpred)
    
    print "Hillary    ", clintonCorrect, trumpIncorrect
    print "Trump      ", clintonIncorrect, trumpCorrect
    
    
    
    
    
    
    
    
    










    
    
    
     
def multinomialbayes(verbose=True, advanceWordProcess=False, prior=1.0, outputFile = "multinomialResults.txt"):
    """
    Training the classifier with Multinomial Bayes model
    
    Parameters:
        verbose : True/False 
                if true will print out training information
                if false only print out the prediction accuracy
        advancedWordProcess : True/False
                if true will allow link and stopword removal in the dictionaray
                if false will keep everything
        prior : float
                the prior of the model
    """
    if verbose:
        print("Training.....")


    clintonWordCounts = defaultdict(int)
    trumpWordCounts = defaultdict(int)
    
    clintonTweetCount = 0.0  # the number of tweet in the training about Clinton
    trumpTweetCount = 0.0    # the number of tweet in the training about Trump

    pClinton = 0.0  # fraction of tweets about clinton
    pTrump = 0.0    # fraction of tweets about trump
    totalClintonWords = 0.0   #total number of words about clinton
    totalTrumpWords = 0.0     #total number of words about trump

    for labelTweet in trainLabelTweetList:
        label = labelTweet[0]
        tweet = labelTweet[1]

        if label == "HillaryClinton":
            clintonTweetCount += 1.0
            for word in labelTweet[1].split():
               if advanceWordProcess:
                   word = advancedCleanup(word)
               else:
                   word = cleanup(word) 
               if(len(word) > 0):
                    clintonWordCounts[word] += 1
                    totalClintonWords += 1
        else:
            trumpTweetCount +=1.0
            for word in labelTweet[1].split():
                if advanceWordProcess:
                   word = advancedCleanup(word)
                else:
                   word = cleanup(word) 
                if(len(word) > 0):
                    trumpWordCounts[word] += 1
                    totalTrumpWords += 1
    

    #                  # Hillary Tweets
    # Pr[Hillary] = ----------------------
    #                  total # tweets
    pClinton = clintonTweetCount / (clintonTweetCount + trumpTweetCount)
    pTrump = 1.0 - pClinton    
    
    if verbose:
        print("10 highly probable words about Clinton")
        sortedwords = sorted(clintonWordCounts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(0,10):
            print(sortedwords[i])
        print("10 highly probable words about Trump")
        sortedwords = sorted(trumpWordCounts.items(), key=operator.itemgetter(1), reverse=True)
        for i in range(0,10):
            print(sortedwords[i])
        print("Testing.....")
    


    if outputFile != "":
        outfile = open(outputFile,"w")


                
    clintonCorrect = 0
    clintonIncorrect = 0
    trumpCorrect = 0
    trumpIncorrect = 0


    predCorrectCount = 0.0
    totalpred = 0.0
    for labelTweet in testLabelTweetList:
        label = labelTweet[0]
        tweet = labelTweet[1]
        totalpred += 1

        #
        # l(class | tweet)
        #      = log(Pr(class)
        #      + \sum_{w in tweet} log( Pr(w | class))
        #      + \sum_{w !  in tweet} log( 1 - Pr(w | class))
        #
        clintonloglikelihood = math.log(pClinton)
        trumploglikelihood = math.log(pTrump)

        for word in tweet.split():
            word = cleanup(word)
        
            if len(word) > 0:

                if word in clintonWordCounts:
                    #                              # times Hillary said word +
                    #                              prior
                    # Pr[word | Hillary] = log
                    # -----------------------------------------------------
                    #                              total Hillary words + (total
                    #                              uniquie words * prior)
                    clintonloglikelihood += math.log((clintonWordCounts[word] * 1.0 + prior) / (totalClintonWords * 1.0 + len(vocab) * prior))
                else:
                    #                                    prior
                    # Pr[word | Hillary] = log
                    # -----------------------------------------------------
                    #                              total Hillary words + (total
                    #                              uniquie words * prior)
                    clintonloglikelihood += math.log(prior / (totalClintonWords * 1.0 + len(vocab) * prior))
                

                if word in trumpWordCounts:
                    trumploglikelihood += math.log((trumpWordCounts[word] * 1.0 + prior) / (totalTrumpWords * 1.0 + len(vocab) * prior))    
                else:
                    trumploglikelihood += math.log(prior / (totalTrumpWords * 1.0 + len(vocab) * prior))  
                    


        if outfile != None:
            if clintonloglikelihood > trumploglikelihood:
                outfile.write("HillaryClinton\n")
            else:
                outfile.write("realDonaldTrump\n")
            
                        
        if label == "HillaryClinton":
            #print("Clinton prediction is correct")
            if clintonloglikelihood > trumploglikelihood:
                predCorrectCount += 1.0
                clintonCorrect += 1
            else:
                clintonIncorrect += 1
                 
        else: # label == "realDonaldTrump"

            if clintonloglikelihood < trumploglikelihood:
                predCorrectCount += 1.0
                trumpCorrect += 1
            else:
                trumpIncorrect += 1          

            if outfile != None:
                outfile.write("realDonaldTrump\n")

    print("Prediction accuracy:")
    print(predCorrectCount / totalpred)    
     


    print "Hillary    ", clintonCorrect, trumpIncorrect
    print "Trump      ", clintonIncorrect, trumpCorrect
    











def main():
    global vocab
    print("Parsing vocabulary")
    parseVocabulary(traintweet)
    global trainLabelTweetList
    global testLabelTweetList
    print("Parsing training files.....")
    trainLabelTweetList = parseLabelTweetList(traintweet,trainlabel)
    print("Parsing testing file.....")
    testLabelTweetList = parseLabelTweetList(testtweet, testlabel)
    #print(testFLmap)
    print "\n======================================================="
    print " Part 1:(a) Applying Bernoulli Model"
    print "=======================================================\n"
    bernoulliBayes(verbose = True)
    print "\n======================================================="
    print " Part 1(b): Applying Multinomial Model"
    print "=======================================================\n"
    multinomialbayes(verbose = True)
    print ''
    print ''
    print "\n======================================================="
    print " Part 2: Prior vs prediction accuracy"
    print "=======================================================\n"
    
    
    for log_alpha in range(-5,1):
        alpha = math.pow(10, log_alpha)
        print("Applying Bernoulli Model with prior =" + str(alpha))
        bernoulliBayes(verbose = False, prior = alpha, outputFile = ("BernoulliResult_alpha"+str(alpha)+".txt"))
    
        
    print ''
    print ''


    for log_alpha in range(-5,1):
        alpha = math.pow(10, log_alpha)
        print("Applying Multinomial Model with prior =" + str(alpha))
        multinomialbayes(verbose = False, prior = alpha, outputFile = ("MultinomialResult_alpha"+str(alpha)+".txt"))




    print ''
    print ''
    print("Importing stop-words")

    #parseStopwords(stopwordsfile)
    leastUsedStopWords(10)
    vocab.clear()
    #parseVocabulary(traintweet, advanceWordProcess = True)
    
    
    print "\n======================================================="
    print "              Part 3(a): Applying Bernoulli Model with Stopwords Removal....."
    print "=======================================================\n"
    
    
    bernoulliBayes(verbose = True, advanceWordProcess = True,outputFile = ("BernoulliResult_smallVocab.txt"))
    print "\n======================================================="
    print "              Part 3(b): Applying Multinomial Model with stopwords removal....."
    print "=======================================================\n"
    
    
    multinomialbayes(verbose = False, advanceWordProcess = False,outputFile = ("MultinomialResult_smallVocab.txt"))
    return 0    























if __name__ == '__main__':
    main();