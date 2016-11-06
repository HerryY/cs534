#include "MLTreeMain.h"



#include "TreeNode.h"
#include "MLTree.h"
#include <fstream>

#include "BoostedMLTree.h"
#include "DiffPrivMedian.h"

#include "PlainQueryOracle.h"
#include "DiffPrivQueryOracle.h"

#include <string>
#include <sstream>
#include "Laplace.h"

#include "Crypto/PRNG.h"



void MLTreeMain(int ii)
{
    MLTreeCTScanDP(ii);
    std::cout << "done" << std::endl;
}


void loadFromFile(
    std::istream & in,
    std::vector<DbTuple>& mRows,
    bool header,
    std::vector<std::function<bool(const std::vector<double>&)>>& preds,
    u64 maxRows = -1)
{
    std::string line;

    if (header)
        std::getline(in, line);

    mRows.clear();

    std::vector<std::string> words;

    while (std::getline(in, line) && maxRows--)
    {

        mRows.emplace_back();
        auto& row = mRows.back();
        row.mPreds.resize(preds.size());


        std::vector<std::string> tok;
        split(line, ',', tok);

        for (auto iter = tok.begin(); iter != tok.end(); ++iter)
        {
            auto word = *iter;

            double d = std::stod(word);

            row.mPlain.push_back(d);
        }


        for (u64 i = 0; i < preds.size(); ++i)
        {
            row.mPreds[i] = preds[i](row.mPlain);
        }

        row.mValue = row.mPlain.back();

        //std::cout<< std::endl;
    }

}



void loadCTData2(
    std::vector<DbTuple >& fullData,
    u64 maxRows = -1,
    u64 maxPreds = -1)
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
    std::vector<std::function<bool(const std::vector<double>&)>> preds(std::min(stepCount * numDataColumns, maxPreds));

    auto  predIter = preds.begin();
    for (u64 i = 0; i < stepCount; ++i)
    {

        for (u64 j = 0; j < numDataColumns && predIter != preds.end(); ++j)

        {
            *predIter++ = [j, i, stepCount](const std::vector<double>& t)
            {
                return t[j + 1] >= (i / double(stepCount));
            };
        }
    }



    loadFromFile(in, fullData, true, preds, maxRows);

    // center the data
    for (auto& row : fullData)
        row.mValue -= 50;

}

void MLTreeCTScanDP(int ii)
{
    std::vector<DbTuple > fullData;
    loadCTData2(fullData);


    u64 foldCount = 10;

    std::vector<double>
        learningRates{/* 0.3,*/ 0.1/*, 0.1, 0.05 */ },
        nodeEpsilon{ /*1,0*/0.01/*,0.05,0.025,0.01,0.005,0.0025,0.001 */ },
        leafEpsilon{ /*1,0.1,*/0.01,/*0.025,0.01,0.005,0.0025,0.001 *//*,0.01*/ },
        numTrees{ /*50,100,*/200/*,500 */ },
        depths{ /*2,4,*/20/*,100*/ },
        minSplits{/* 2000,1000,*/1/*, 250 */ };

    YType maxValue = 50,
        minYValue = 50;

    int i = 0;
    for (auto learningRate : learningRates)
        for (auto epsilon : nodeEpsilon)
            for (auto epsilonLeaf : leafEpsilon)
                for (auto numTree : numTrees)
                    for (auto depth : depths)
                    {
                        for (auto minSplitSize : minSplits)
                        {

                            if (i++ == ii || ii == -1)
                            {
                                //DiffPrivQueryOracle qq(epsilon, epsilonLeaf, maxValue, minYValue, depth, minSplitSize, numTree);
                                PlainQueryOracle qq(depth, minSplitSize);
                                BoostedMLTree tree;

                                std::vector<DbTuple> evalData, data, d2;
                                data.clear();

                                u64 i = 0;

                                data.insert(
                                    data.end(),
                                    fullData.begin(),
                                    fullData.begin() + (i * fullData.size() / foldCount));

                                data.insert(
                                    data.end(),
                                    fullData.begin() + ((i + 1) * fullData.size() / foldCount),
                                    fullData.end());

                                evalData.clear();
                                evalData.insert(
                                    evalData.begin(),
                                    fullData.begin() + (i * fullData.size() / foldCount),
                                    fullData.begin() + ((i + 1) * fullData.size() / foldCount));
                                auto w = std::setw(8);
                                std::stringstream ss;
                                ss << w << learningRate << w << epsilon
                                    << w << epsilonLeaf << w << numTree
                                    << w << depth << w << minSplitSize;


                                tree.learn(data, qq, numTree, learningRate, &evalData);
                            }
                        }
                    }
}


