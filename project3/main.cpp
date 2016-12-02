
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
#include <thread>
#include <string>
#include <future>

void loadFromFile(
    std::istream & in,
    std::vector<DbTuple>& mRows,
    bool header,
    std::vector<std::vector<std::function<bool(const std::vector<double>&)>>>& preds,
    std::function<double(const std::vector<double>&)>& classMap,
    u64 maxRows = -1)
{
    std::string line;

    if (header)
        std::getline(in, line);

    mRows.clear();

    std::vector<std::string> words;

    u64 i = 0;
    while (std::getline(in, line) && maxRows--)
    {

        mRows.emplace_back();
        auto& row = mRows.back();


        std::vector<std::string> tok;
        split(line, ',', tok);

        for (auto iter = tok.begin(); iter != tok.end(); ++iter)
        {
            auto word = *iter;

            double d = std::stod(word);

            row.mPlain.push_back(d);
        }

        row.mPredsGroup.resize(preds.size());

        for (u64 i = 0; i < preds.size(); ++i)
        {

            row.mPredsGroup[i].resize(preds[i].size());

            for (u64 j = 0; j < preds[i].size(); ++j)
            {
                row.mPredsGroup[i][j] = preds[i][j](row.mPlain);
            }
        }

        row.mValue = classMap(row.mPlain);
        row.mIdx = i++;
        //std::cout<< std::endl;
    }

}



void loadCTData2(
    std::vector<DbTuple >& fullData,
    u64 maxRows = -1)
{
    std::string filePath("./slice_localization_data.csv");
    u64 numColumns = 385;
    u64 numDataColumns = numColumns - 2;

    std::fstream in;
    in.open(filePath, in.in);

    if (in.is_open() == false)
    {
        filePath = std::string(SOLUTION_DIR) + "project3/MLTree/" + filePath;
        in.open(filePath, in.in);
    }

    if (in.is_open() == false)
    {
        std::cout << "cant open " << filePath << std::endl;
        throw std::runtime_error("");
    }



    u64 stepCount = 4;
    std::vector<std::vector<std::function<bool(const std::vector<double>&)>>> preds(numDataColumns);


    for (u64 j = 0; j < numDataColumns; ++j)
    {
        preds[j].resize(stepCount);

        for (u64 i = 0; i < stepCount; ++i)
        {
            preds[j][i] = [j, i, stepCount](const std::vector<double>& t)
            {
                return t[j + 1] >= ((i) / double(stepCount));
            };
        }
    }

    std::function<double(const std::vector<double>&)> classMap = [](const std::vector<double>& t)
    {
        return t.back();// -50;
    };

    loadFromFile(in, fullData, true, preds, classMap, maxRows);

    //// center the data
    //for (auto& row : fullData)
    //    row.mValue -= 50;

}


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



    //std::vector<DbTuple > trainingData, testData;
    //loadIris(trainingData, "./iris-train.csv");
    //loadIris(testData, "./iris-test.csv");


    std::vector<DbTuple > fullData;
    loadCTData2(fullData);

    PRNG prng(0);
    //std::shuffle(fullData.begin(), fullData.end(), prng);
    u64 foldCount = 10;
    u64 foldStart = 0, foldEnd = 10;



    std::vector<std::vector<DbTuple>>  trainingData(foldCount), testData(foldCount);
    //u64 i = 0;

    for (u64 i = foldStart; i < foldEnd; ++i)
    {
        auto testStart = fullData.begin() + (i * fullData.size() / foldCount);
        while (testStart != fullData.begin() && testStart->mPlain[0] == (testStart - 1)->mPlain[0])
            ++testStart;
        auto testend = fullData.begin() + ((i + 1) * fullData.size() / foldCount);
        while (testend != fullData.end() && testend->mPlain[0] == (testend - 1)->mPlain[0])
            ++testend;

        if (testStart == testend)
        {
            testStart = fullData.begin() + (i * fullData.size() / foldCount);
            testend = fullData.begin() + ((i + 1) * fullData.size() / foldCount);
            std::cout << "WARNING: train and test sets overlap in patent id" << std::endl;
        }


        trainingData[i].insert(
            trainingData[i].end(),
            fullData.begin(),
            testStart);

        trainingData[i].insert(
            trainingData[i].end(),
            testend,
            fullData.end());

        testData[i].insert(
            testData[i].begin(),
            testStart,
            testend);
    }


    //valData.insert(
    //    valData.begin(),
    //    fullData.end() - (fullData.size() / foldCount),
    //    fullData.end());

    //fullData.resize(fullData.size() - valData.size());



    //std::cout << "test  " << testData.front().mIdx << "  " << testData.back().mIdx << std::endl;
    //std::cout << "train " << trainingData.front().mIdx << "  " << trainingData.back().mIdx << std::endl;



    std::vector<double>
        learningRates{ 0.15/*, 0.1,0.075, 0.05, 0.025 */},
        epsilons{ 1, 0.1,0.05, 0.01, 0.005, 0.001 ,0.0005,0.0001, 0.00005},
        dropRates{ 0.1, 0.05, 0.01, 0.05, 0.0001};
    std::vector<i64>
        numTreess{/* 100,*/ 300 },
        minSplitSizes{ 10/*, 10, 100*//*, 1000*/ },
        maxDepths{ /*100, */-1 },
        maxLeafCounts{ 1000 };


    std::vector<SplitType> types
    {
        //SplitType::L2,
        //SplitType::Dart,
        SplitType::DLart//,
        //SplitType::L2Laplace//,
        //SplitType::Random
    };


    std::vector<std::function<void()>> funcs;
    std::fstream master;
    master.open(SOLUTION_DIR "master.txt", std::ios::trunc | std::ios::out);
    if (master.is_open() == false)
    {
        std::cout << "failed to open file: " << SOLUTION_DIR "master.txt" << "  " /*<< std::strerror_s(errno)*/ << std::endl;
        throw std::runtime_error(LOCATION);
    }
    std::mutex masterMtx;

    std::atomic<u64>completedTrees(0);
    u64 totalTrees = 0;
    u64 jj = 0;
    for (u64 i = foldStart; i < foldEnd; ++i)
    {
        for (auto learningRate : learningRates)
        {
            for (auto numTrees : numTreess)
            {
                for (auto minSplitSize : minSplitSizes)
                {
                    for (auto maxDepth : maxDepths)
                    {
                        for (auto maxLeafCount : maxLeafCounts)
                        {
                            for (auto epsilon : epsilons)
                            {
                                for (auto dropRate : dropRates)
                                {
                                    for (auto type : types)
                                    {
                                        // only laplace based learners use epsilon
                                        if (type != SplitType::L2Laplace &&
                                            type != SplitType::DLart &&
                                            epsilon != epsilons[0]) continue;

                                        // only L2 style trees use a learning rate...
                                        if (type != SplitType::L2Laplace &&
                                            type != SplitType::L2 &&
                                            learningRate != learningRates[0]) continue;

                                        // only dart style learners use drop rate
                                        if (type != SplitType::Dart &&
                                            type != SplitType::DLart &&
                                            dropRate!= dropRates[0]) continue;

                                        totalTrees += numTrees;

                                        auto j = jj++;
                                        //futures.emplace_back(std::async(
                                        funcs.emplace_back(
                                            [&, j, i,dropRate, learningRate, numTrees, minSplitSize, maxDepth, maxLeafCount, epsilon, type]()
                                        {
                                            u64 nt = numTrees;

                                            //if (type == SplitType::Random)
                                            //{
                                            //    nt *= 10;
                                            //}

                                            BoostedMLTree tree;

                                            tree.completedTrees = &completedTrees;

                                            std::fstream out;
                                            std::string name =
                                                "/eval" 
                                                + (type == SplitType::L2Laplace || type == SplitType::L2
                                                    ?"lr" + std::to_string(learningRate) : "")
                                                //+ "_nT" + std::to_string(nt)
                                                //+ "_mS" + std::to_string(minSplitSize)
                                                //+ "_mD" + std::to_string(maxDepth)
                                                //+ "_mL" + std::to_string(maxLeafCount)
                                                + (type == SplitType::L2Laplace || type == SplitType::DLart
                                                    ? "_e" + std::to_string(epsilon) : "")
                                                + (type == SplitType::Dart || type == SplitType::DLart 
                                                    ? "_dr" + std::to_string(dropRate) : "")
                                                + "_" + toString(type)
                                                + "_" + std::to_string(i)
                                                + "_.txt";

                                            out.open(SOLUTION_DIR + name, std::ios::trunc | std::ios::out);

                                            if (out.is_open() == false)
                                            {
                                                std::cout << "failed to open file: " << SOLUTION_DIR + name << "  " /*<< std::strerror_s(errno)*/ << std::endl;
                                                throw std::runtime_error(LOCATION);
                                            }

                                            tree.mOut = &out;

                                            tree.mPrng.SetSeed(prng.get<u64>());

                                            tree.learn(
                                                trainingData[i], 
                                                nt, 
                                                learningRate, 
                                                minSplitSize, 
                                                maxDepth, 
                                                maxLeafCount, 
                                                type, 
                                                epsilon, 
                                                dropRate, 
                                                &testData[i]);

                                            masterMtx.lock();
                                            master << name << " " << tree.mBest << std::flush;
                                            masterMtx.unlock();

                                        });
                                    }
                                }

                            }
                        }
                    }
                }
            }
        }
    }

    std::mutex mtx;
    u64 numThrds = std::thread::hardware_concurrency() - 3;

    std::vector<std::thread> thrds(numThrds);
    std::cout << numThrds << " threads working on " << funcs.size() << " tasks" << std::endl;
    std::atomic<u64>j(0);

    for (u64 i = 0; i < numThrds; ++i)
    {
        thrds[i] = std::thread([&]()
        {
            std::function<void()> func;

            while (true)
            {
                mtx.lock();
                if (funcs.size())
                {
                    func = funcs.back();
                    funcs.pop_back();
                    mtx.unlock();

                    func();
                    //std::cout << "\r" << j++ << "done            " << std::flush;

                }
                else
                {
                    mtx.unlock();
                    return;
                }
            }

        });
    }

    double rem = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::time_point::clock::now();

    u64 cc = 0;
    while (completedTrees != totalTrees)
    {
        u64 c = completedTrees;
        auto d = (c * 1.0 / totalTrees);
        auto percent = d * 100.0;

        if (c != cc)
        {
            cc = c;
            auto now = std::chrono::system_clock::time_point::clock::now();
            auto diff = now - start;
            auto cur = std::chrono::duration_cast<std::chrono::seconds>(diff).count();
            rem = (cur * (1 - d) / d);
        }

        u64 min = rem / 60;
        u64 sec = (u64)rem % 60;

        std::cout << "\r" << std::setprecision(2) << std::setw(4) << percent << "%     "
            << completedTrees << " / " << totalTrees << " sub-trees    "
            << min << ":" << std::setw(2) << std::setfill('0') << sec << " min remaining            " << std::flush;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    //std::cout << std::max(futures.size(), thrds.size()) << " threads at work" << std::endl;
    for (auto& thrd : thrds)
    {
        thrd.join();
    }
    std::cout << "\r" << funcs.size() << " done" << std::endl;

    return 0;
}

