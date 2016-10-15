import sys
import operator
from collections import defaultdict
import numpy as np

# Read in all four files
def readin(filename):
    return [x.strip() for x in open(filename, "r").readlines()]
X = readin(sys.argv[1])
devX = readin(sys.argv[2])
testX = None
if len(sys.argv) > 3:
    testX = readin(sys.argv[3])

# Convert the tweets into a bag of words representation
labels,vocabulary = defaultdict(int),defaultdict(int)
for x in X:
    x = x.split()
    for word in x:
        vocabulary[word] += 1
for x in devX:
    x = x.split()
    for word in x:
        vocabulary[word] += 1
if not testX is None:
    for x in testX:
        x.split()
        for word in x:
            vocabulary[word] += 1


f = open("clintontrump.vocabulary", "w")
mappingTo,mappingFrom = {},{}
for i,word in enumerate(vocabulary):
    mappingTo[word] = i
    mappingFrom[i] = word
    f.write(str(i) + "\t" + word + "\n")
f.close()


def bagofwords(word):
    return mappingTo.get(word, len(vocabulary))

def lookup(index):
    return mappingFrom.get(index, "<UNK>")



f = open("clintontrump.bagofwords.train","w")
for x in X:
    line = " ".join([str(bagofwords(word)) for word in x.split()])
    f.write(line + "\n")
f.close()

f = open("clintontrump.bagofwords.dev","w")
for x in devX:
    line = " ".join([str(bagofwords(word)) for word in x.split()])
    f.write(line + "\n")
f.close()

if not testX is None:
    f = open("clintontrump.bagofwords.test","w")
    for x in testX:
        line = " ".join([str(bagofwords(word)) for word in x.split()])
        f.write(line + "\n")
    f.close()
    
#if __name__ == '__main__':
#    main();