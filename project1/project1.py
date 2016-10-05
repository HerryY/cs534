import numpy  as np 
import sys
import inspect

def getDefaultValue(func, field):
    
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return dict(zip(reversed(args), reversed(defaults)))[field]


def parse(filePath):
    """
    Parses a CSV file containing floats
    
    Parameters
    -------
    filePath : str
        Full or relative path to the CSV file to be parsed
    
    Returns
    -------
    output : numpy array
        An array containing the parsed file
    """
    return np.genfromtxt(filePath, delimiter=',')

def computeNormalizingParams(data):
    """
    Computes the standard deviation and mean for each column of the data 
    
    Parameters
    -------
    data : numpy array
        The data from which the sd and mean is computed
    
    Returns
    -------
    output : n by 2 numpy array
        An array with the standard deviation and mean for each column. sd contained in first column, mean the second
    """

    numRows, numCols = data.shape

    params = np.ones((numCols,2))

    for i in range(numCols): 
        params[i,0] = np.nanstd(data[:,i])
        params[i,1] = np.nanmean(data[:,i]) 
    return params


def normalize(data, norms):
    """
    Returns normaled data based on the provided parameters. If the first column is a vector of ones, it is unchanged.
    
    Parameters
    -------
    data : m by n numpy array
        The data to be normalized
    norms : n by 2 numpy array
        The standard divantion and means which the data should be normalized with.
    
    Returns
    -------
    output : m by n numpy array
        data normalized with the provided params
    """
    numRows, numCols = data.shape
    colStart = 0


    # if we have a normalizing column, just skip it...
    if(norms[0,0] == 0 and norms[0,1] == 1):
        colStart = 1

    for j in range(colStart, numCols):
        if(norms[j,0]):
            for i in range(numRows):
                data[i,j] = (data[i,j] - norms[j,1]) / norms[j,0]
        else: 
            for i in range(numRows):
                data[i,j] = (data[i,j] - norms[j,1])
    return data


def score(data, model, reg):
    """
    Returns normaled data based on the provided parameters. If the first column is a vector of ones, it is unchanged.
    
    Parameters
    -------
    data : m by n numpy array
        The data to be scored
    norms : n by 2 numpy array
        The model for the data
    
    Returns
    -------
    output : 2-tuple of floats
        The avg L2, avg L2, and regularized L2 (SSE) score of the data on the model
    """

    X = data[:,0:-2]
    y = data[:,-1]

    numRows, numCols = X.shape


    # get our prediction error
    yp = np.dot(X ,model)
    error = yp - y.reshape((numRows,1))


    l1Error = np.linalg.norm(error, ord = 1)
    l2Error = pow(np.linalg.norm(error, ord = 2),2)

    rSSE = l2Error + reg * pow(np.linalg.norm(model, ord = 2),2)

    l1Error = l1Error / numRows
    l2Error = l2Error / numRows

    return l1Error, l2Error, rSSE


def train(data, reg = 0.01, stepSize = 0.1, epsilon = 0.0001, printScores = False, testData = None):
    """
    Performed batch gradient decent with L2 regularization. 
    
    Parameters
    ----------
    data : m by n numpy array
        m labeled training examples. Features are the first n-1 columns, the label is the last column.
    reg : float
        The L2 regularization hyper parameter. Should be in the range [0,inf).
    stepSize : float
        The normal gradient decent hyper parameter to control convergence rate.
    epsilon : float
        Determines when the gradient decent algorithm has gotten close enough. When the avg L2 error 
        improves less than epsilon, we terminate.
    printScores : bool
        if true, print the scores (L1,L2) of the training data after each iteration
    testData : m by n numpy array
        m labeled test examples. Features are the first n-1 columns, the label is the last column.

    Returns
    -------
    model : 1 by n numpy array
        The model computed by the gradient decent algorithm


    """

    if( reg < 0):
        print "reg (regularization hyper parameter)  must be non-negative. Defualting to 0"
        reg = 0

    # make sure its a float
    stepSize = stepSize * 1.0

    # the feature values are all the data except the last column
    X = data[:,0:-2]

    # the labels are in the last column
    y = data[:,-1]

    yMean = np.nanmean(np.absolute(y))

    numRows, numCols = X.shape 
    converged = False
    diverged = False


    # initialize the model as the one vector
    w = np.ones((numCols, 1))

    startL1, startL2, _ = score(data, w, reg)
    prevL2 = startL2


    iterationIdx =0
    
    # Print the current scores if instructed to
    #if printScores:
    #    print iterationIdx,  startL1, startL2 
    
    while ((diverged or converged) == False):
        iterationIdx = iterationIdx + 1;

        # initialize the gradient as the zero vector
        deltaE = np.zeros((numCols,1))

        # compute the batched gadient
        for i in range(numRows):

            # compute the estimate for this training example
            yp = np.dot(w.transpose(), X[i,:]);

            # compute the error with respect to the true label
            error = yp - y[i];

            # compute the error scaled by this examples features. reshare(...) isn't conceptually import
            update = error * X[i,:].reshape(deltaE.shape);

            #update the overall gradient
            deltaE = deltaE +  update 

 
        # update the model. Scale it by the step size and number of examples. 
        # scaling by the number of exmaples make sure that we dont need to change
        # the step size if we have more or less training data
        w = w -  stepSize * ( 1.0/numRows * deltaE + (reg * 2.0 * w));


        # compute the L1,L2 score of the training data
        L1, L2, rSSE = score(data, w, reg);


        # Print the current scores if instructed to
        if printScores:
            print iterationIdx, "   L1 = ", L1, "     L1/|mean| = ", (L1 / yMean), "    L2 = ", L2 ,"    rSSE = ", rSSE
        
        # if we are provided test data, print their current scores too
        if testData is not None:
            testL1, testL2, testrSSE  = score(testData, w, reg)
            print iterationIdx, "  tL1 = ", testL1, "    tL1/|mean| = ", (testL1 / yMean), "   tL2 = ", testL2 ,"   trSSE = ", testrSSE

        # compute how much of an improvement over the last model we have
        L2Change = prevL2 - L2 
        prevL2 = L2

        # see if we are getting a lot worse. If so, just quit...
        if(startL2  < L2 /1000):
            print "diverged!"
            diverged = True;

        # decide if we have converged by seeing if the L1 norm of the gradient id close to zero
        converged =  L2Change  < (epsilon)
        

    return w

def exmaple(trainingData, testData):
    
    print "\n======================================================="
    print "                    example"
    print "=======================================================\n"

    # run gradient decent with L2 regression
    model = train(trainingData, printScores = True)


    reg = getDefaultValue(train, "reg");

    # compute the score of the resulting model usign the test data
    print score(testData, model, reg)

def part1(trainingData, testData):
        
    print "\n======================================================="
    print "                    part1"
    print "=======================================================\n"


    print "learning rate,     avg l1 error,   avg l2 error,      regularized SSE"
    for pw in range(-8,6):

        learningRate = pow(2,pw)
        
        # run gradient decent with L2 regression
        model = train(trainingData,stepSize = learningRate)
        reg = getDefaultValue(train, "reg");

        # compute the score of the resulting model usign the test data
        print learningRate, score(testData, model, reg)


def part2(trainingData, testData):
        
    print "\n======================================================="
    print "                    part2"
    print "=======================================================\n"
    print "reg,     avg l1 error,   avg l2 error,      regularized SSE"

    regs = [0];
    for pw in range(-12,6):
        regs.append(pow(2,pw))

    for reg in regs:

        # run gradient decent with L2 regression
        model = train(trainingData,reg = reg)

        # compute the score of the resulting model usign the test data
        print reg, score(testData, model, reg)


def part3(d1, d2):
        
    print "\n======================================================="
    print "                    part3"
    print "=======================================================\n"
    print "reg,     k-fold additive regularized SSE"

    data = np.concatenate((d1, d2), axis=0)
    numRows, numCols = data.shape

    numFolds = 10

    regs = [0];
    for pw in range(-12,6):
        regs.append(pow(2,pw))


    for reg in regs:

        sumrSSE = 0
        for foldIdx in range(numFolds):

            trainingData = np.concatenate(
                (data[0: foldIdx * numRows / numFolds,:],
                data[(foldIdx+1) * numRows / numFolds:-1,:]), axis=0)

            testData = data[foldIdx * numRows / numFolds : (foldIdx+1) * numRows / numFolds,:];

            
            # run gradient decent with L2 regression
            model = train(trainingData,reg = reg)

            # compute the score of the resulting model usign the test data
            _,_2, rSSE = score(testData, model, reg)

            sumrSSE = sumrSSE + rSSE

        print reg ,sumrSSE

def main():

    
    reg = getDefaultValue(train, "reg");
    
    # parse the prodived data files
    trainingData = parse("trainP1-16.csv")
    testData = parse("testP1-16.csv")
    
    # compute the normaizing parameters based on the training data
    normalizingParams = computeNormalizingParams(trainingData)

    # normalize both the training data and test data using these params
    trainingData = normalize(trainingData, normalizingParams)
    testData = normalize(testData, normalizingParams)


    exmaple(trainingData, testData)
    
    part1(trainingData, testData)
    part2(trainingData, testData)
    part3(trainingData, testData)


    return 0    























if __name__ == '__main__':
    main();