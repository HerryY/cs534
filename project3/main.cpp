
#include "Util/CLP.h"
#include <stdint.h>
#include <iostream>
#include <fstream>

#include "MLTree/BoostedMLTree.h"
#include "MLTree/TreeNode.h"
#include "MLTree/MLTree.h"
#include <algorithm>
#include "Common/PRNG.h"

void loadIris(
    std::vector<DbTuple >& rows)
{
    std::string filePath("./iris.data");

    std::fstream in;
    in.open(filePath, in.in);

    // if we failed to find it at relative path ./iris.trainingData
    // try using the absolution path that goes to the folder that
    // this file lives in.
    if (in.is_open() == false)
    {
        filePath = std::string(SOLUTION_DIR) + "project3/MLTree/" + filePath;
        in.open(filePath, in.in);
    }

    // if we failed again, just give up.
    if (in.is_open() == false)
    {
        std::cout << "cant open " << filePath << std::endl;
        throw std::runtime_error("");
    }




    //sepal length: 4.3  7.9   5.84  0.83    0.7826   
    //sepal width: 2.0  4.4   3.05  0.43   -0.4194
    //petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    //petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

    // trainingData parameters
    double sepalLengthMin(4.2);
    double sepalLengthMax(7.9);
    double sepalLengthRange = sepalLengthMax - sepalLengthMin;

    double sepalWidthMin(2.0);
    double sepalWidthMax(4.4);
    double sepalWidthRange = sepalWidthMax - sepalWidthMin;

    double petalLengthMin(1.0);
    double petalLengthMax(6.9);
    double petalLengthRange = petalLengthMax - petalLengthMin;

    double petalWidthMin(0.1);
    double petalWidthMax(2.5);
    double petalWidthRange = petalWidthMax - petalWidthMin;

    // the resolution of the predicates, the greater the value, the more predicates we
    // will have and the slower the learning will be. Note that more predicates
    // doesn't always mean more accurate models...
    u64 stepCount = 40;

    // the total number of boolean features that we want. 4 because the input
    // trainingData is 4 floating foint values
    u64 numPredicates = 4 * stepCount;

    std::string line;

    rows.clear();

    std::vector<std::string> words;

    // get the next line of the file. When there are no more lines, this will
    // return false and leave the loop.
    while (std::getline(in, line))
    {
        if (line.size())
        {


            // add anther row to the dataset
            rows.emplace_back();

            // get a reference to this row. 
            auto& row = rows.back();

            // split the row by their commas.
            std::vector<std::string> tok;
            split(line, ',', tok);

            // convert the string to doubles
            double sepalLength = std::stod(tok[0]);
            double sepalWidth = std::stod(tok[1]);
            double petalLength = std::stod(tok[2]);
            double petalWidth = std::stod(tok[3]);

            // copy the trainingData into row. This trainingData isn't touched by the learning algorithm
            row.mPlain.resize(4);
            row.mPlain[0] = sepalLength;
            row.mPlain[1] = sepalWidth;
            row.mPlain[2] = petalLength;
            row.mPlain[3] = petalWidth;

            // resize this rows predicates. This are the boolean features that
            // we will extract from the actual input trainingData.
            row.mPreds.resize(numPredicates);

            // now compute the predicates
            for (u64 i = 0; i < stepCount; ++i)
            {
                // a fraction in (0,1) that will determine the current split value
                // within each the the ranges that the features can take.
                double frac = (i + 1.0) / (stepCount + 1);


                row.mPreds[0 * stepCount + i] = ((sepalLength - sepalLengthMin)/ sepalLengthRange > frac ) ? 1 : 0;

                //std::cout << sepalLength << " > " << ((frac * sepalLengthRange) + sepalLengthMin) << "  " << (u32)row.mPreds[0 * stepCount + i] <<std::endl;;

                row.mPreds[1 * stepCount + i] = ((sepalWidth - sepalWidthMin) / sepalWidthRange > frac) ? 1 : 0;
                row.mPreds[2 * stepCount + i] = ((petalLength - petalLengthMin) / petalLengthRange > frac) ? 1 : 0;
                row.mPreds[3 * stepCount + i] = ((petalWidth - petalWidthMin) / petalWidthRange > frac) ? 1 :0;
            }

            // compute the class value
            if (tok[4] == "Iris-setosa")
            {
                row.mValue = 0;
            }
            else if (tok[4] == "Iris-versicolor")
            {
                row.mValue = 1;
            }
            else
            {
                row.mValue = 3;
            }
        }
    }

}


int main(int argc, char** argv)
{

    std::vector<DbTuple > fullData;
    loadIris(fullData);

    PRNG prng(2345);

    // shuffle the data so that when we split it into test and training
    // we dont get all of one class
    std::shuffle(fullData.begin(), fullData.end(), prng);
    u64 foldCount = 10;

    //for (u64 i = 0; i < fullData.size(); ++i)
    //{
    //    std::cout
    //        << fullData[i].mPlain[0] << ", "
    //        << fullData[i].mPlain[1] << ", "
    //        << fullData[i].mPlain[2] << ", "
    //        << fullData[i].mPlain[3] << " => "
    //        << fullData[i].mValue << std::endl;

    //}

    double
        learningRate{ 1 },
        numTrees{ 1 },
        minSplitSize{ 16};


    BoostedMLTree tree;

    std::vector<DbTuple> testData, trainingData, d2;
    trainingData.clear();

    u64 i = 0;

    trainingData.insert(
        trainingData.end(),
        fullData.begin(),
        fullData.begin() + (i * fullData.size() / foldCount));

    trainingData.insert(
        trainingData.end(),
        fullData.begin() + ((i + 1) * fullData.size() / foldCount),
        fullData.end());

    testData.clear();
    testData.insert(
        testData.begin(),
        fullData.begin() + (i * fullData.size() / foldCount),
        fullData.begin() + ((i + 1) * fullData.size() / foldCount));


    tree.learn(fullData, numTrees, learningRate, minSplitSize, &fullData);


    return 0;
}

