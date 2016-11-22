
#include "Util/CLP.h"
#include <stdint.h>
#include <iostream>
#include <fstream>

#include "MLTree/BoostedMLTree.h"
#include "MLTree/TreeNode.h"
#include "MLTree/MLTree.h"
#include <algorithm>
#include "Common/PRNG.h"
#include "MLTree/RandomForest.h"

void loadIris(
    std::vector<DbTuple >& rows, std::string filePath)
{
    //("./iris.data");

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

    std::array<double, 4> mins{ sepalLengthMin ,sepalWidthMin ,petalLengthMin , petalWidthMin };
    std::array<double, 4> ranges{ sepalLengthRange ,sepalWidthRange ,petalLengthRange , petalWidthRange };

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
    u64 idx = 0;
    // get the next line of the file. When there are no more lines, this will
    // return false and leave the loop.
    while (std::getline(in, line))
    {
        if (line.size())
        {


            // add anther row to the dataset
            rows.emplace_back();

            rows.back().mIdx = idx++;
            // get a reference to this row. 
            auto& row = rows.back();

            // split the row by their commas.
            std::vector<std::string> tok;
            split(line, ';', tok);

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

            row.mPredsGroup.resize(4);
            row.mPredsGroup[0].resize(stepCount);
            row.mPredsGroup[1].resize(stepCount);
            row.mPredsGroup[2].resize(stepCount);
            row.mPredsGroup[3].resize(stepCount);

            // now compute the predicates
            for (u64 i = 0; i < stepCount; ++i)
            {
                // a fraction in (0,1) that will determine the current split value
                // within each the the ranges that the features can take.
                double frac = (i + 1.0) / (stepCount + 1);

                //std::cout 
                //    << sepalLength << " > " << ((frac * sepalLengthRange) + sepalLengthMin) << "  " << (u32)row.mPreds[0 * stepCount + i] << "  -  "
                //    << sepalWidth << " > " << ((frac * sepalWidthRange) + sepalWidthMin) << "  " << (u32)row.mPreds[1 * stepCount + i] << "  -  "
                //    << petalLength << " > " << ((frac * petalLengthRange) + petalLengthMin) << "  " << (u32)row.mPreds[2 * stepCount + i] << "  -  "
                //    << petalWidth << " > " << ((frac * petalWidthRange) + petalWidthMin) << "  " << (u32)row.mPreds[3 * stepCount + i] << "  -  "
                //    <<std::endl;;


                row.mPredsGroup[0][i] = ((sepalLength - sepalLengthMin) / sepalLengthRange > frac) ? 1 : 0;
                row.mPredsGroup[1][i] = ((sepalWidth - sepalWidthMin) / sepalWidthRange > frac) ? 1 : 0;
                row.mPredsGroup[2][i] = ((petalLength - petalLengthMin) / petalLengthRange > frac) ? 1 : 0;
                row.mPredsGroup[3][i] = ((petalWidth - petalWidthMin) / petalWidthRange > frac) ? 1 : 0;
            }

            row.mValue = std::stoi(tok[4]);
        }
    }

}


int main(int argc, char** argv)
{

    std::vector<DbTuple > trainingData, testData;
    loadIris(trainingData, "./iris-train.csv");
    loadIris(testData, "./iris-test.csv");

    PRNG prng(2345);


    double
        learningRate{ 1 },
        numTrees{ 5 },
        minSplitSize{ 1 };



    std::cout << "single tree using minSplitSize = " << minSplitSize << std::endl;


    u64 trials = 10;

    for (u64 i = 1; i < 100; i += 2)
    {

        double testAcc = 0;
        double trainAcc = 0;
        for (u64 j = 0; j < trials; ++j)
        {
            BoostedMLTree tree;

            tree.learn(trainingData, 1, 1, i);//, &testData

            testAcc += tree.test(testData, 0);
            trainAcc += tree.test(trainingData, 0);

        }
        testAcc = testAcc / trials;
        trainAcc = trainAcc / trials;

        std::cout << "minSp " << i << "  test   " << testAcc << "%  train   " << trainAcc << "%" << std::endl;
    }


    for (u64 minSplitSize = 1; minSplitSize < 0; ++minSplitSize)
    {


        std::cout << "\n\nrandom forest with minSplitSize = " << minSplitSize << " and " << numTrees << " trees." << std::endl;



        for (u64 i = 5; i < 101; i += 5)
        {

            double testAcc = 0;
            double trainAcc = 0;
            for (u64 j = 0; j < trials; ++j)
            {


                RandomForest forest;

                forest.learn(trainingData, i, minSplitSize);//, &testData

                testAcc += forest.test(testData, 0);
                trainAcc += forest.test(trainingData, 0);

            }
            testAcc = testAcc / trials;
            trainAcc = trainAcc / trials;

            std::cout << "trees " << i << "  test   " << testAcc << "%  train   " << trainAcc << "%" << std::endl;


        }

    }
    return 0;
}

