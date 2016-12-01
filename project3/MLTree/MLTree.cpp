#include "MLTree.h"
#include "Common/Defines.h"

#include <array>
#include "Common/Timer.h"
#include <atomic>
#include "Laplace.h"

MLTree::MLTree()
    :mDepth(0)
    , mLeafCount(0)
    , mMuteFactor(1)
{
    // seed the random number generalor with the address of this class
    // Should be pretty random.
    mPrng.SetSeed(0);

}


MLTree::~MLTree()
{

    //deleteNode(root.mLeft);
    //deleteNode(root.mRight);

}


MLTree::MLTree(MLTree &&c)
{
    throw std::runtime_error(LOCATION);
}

// private function for report
double indexFractionToAttributeValue(int i)
{
    // order: sepal length, sepal width, petal length, petal width
    double minAttValues[] = { 4.2, 2.0, 1.0, 0.1 };
    double attRanges[] = { 3.7, 2.4, 5.9, 2.4 };
    int stepCount = 40;

    double v;
    int att = i / stepCount;		// 40 is stepcount
    int frac = (i + 1) % (stepCount + 1);
    v = minAttValues[att] + frac * attRanges[att] / stepCount;

    return v;
}

void MLTree::learn(std::vector<DbTuple>& db, u64 minSplitSize, u64 maxDepth,
    u64 maxLeafCount, SplitType type, double nodeEpsilon)
{

    // initialize some member variables
    mDepth = 0;
    mNodeCount = 1;

    // This will be the root node of this tree.
    // mIdx is just debug info.
    root.mIdx = 1;
    root.mDepth = 0;

    // This update vector holds information that we need 
    // to compute the best split value. we will have 
    // two splitUpdate s for each predicate, one for each
    // of the two nodes that each prodicate will preoduce
    u64 predSize = 0;
    for (u64 i = 0; i < db[0].mPredsGroup.size(); ++i)
        predSize += db[0].mPredsGroup[i].size();

    // select the features that this tree should use.
    // the caller can override this by presetting
    // this array. For a random tree, we select
    // each feature with Pr[1/2]. Otherwise we select all.
    if (mFeatureSelection.size() == 0)
    {
        selectFeatures(db, type);
    }

    // copy the full datset into the root node. It
    // will later be partioned into the leaves through
    // recursive splits
    root.mRows.resize(db.size());
    for (u64 i = 0; i < db.size(); ++i)
    {
        root.mRows[i] = &db[i];
    }

    switch (type)
    {
    case SplitType::Entropy:
        entropySplit(&root, predSize, minSplitSize);
        break;
    case SplitType::Random:
        // currently assumes 3 classes...
        randomSplit(&root, minSplitSize);
        break;
    case SplitType::L2:
    case SplitType::Dart:
        L2Split(&root, minSplitSize);
        break;
    case SplitType::L2Laplace:
        L2LaplaceSplit(&root, minSplitSize, nodeEpsilon);
        break;
    default:
        throw std::runtime_error(LOCATION);
        break;
    }
    //std::cout << "push loss " << root.mLoss << std::endl;

    // add it to the next list, this indicates
    // that this node has not been split yet.
    nextList.push(QueueItem(&root, root.mLoss));




    // while we still have more nodes that need splitting
    while (nextList.size())
    {

        u64 leafCount = mLeafNodes.size() + nextList.size();

        // the pointer to the current node being split
        TreeNode*cur = nextList.top().mNode;

        //std::cout << "Pop  loss " << cur->mLoss << std::endl;

        // remove it from the list, marking it as being processed
        nextList.pop();


        // if we found a predicate (at least one was of min split size), then
        // lets use it and copy our data into the new codes.
        if (cur->mPredIdx[0] != -1 && leafCount < maxLeafCount)
        {
            
            // these are the two new nodes that were prodiced by this split
            std::array<TreeNode*, 2> nodes = { new TreeNode(),new TreeNode() };

            // connect them to the parent
            cur->mLeft.reset(nodes[0]);
            cur->mRight.reset(nodes[1]);

            // do some book keeping, used for debug
            nodes[0]->mIdx = cur->mIdx << 1;
            nodes[1]->mIdx = cur->mIdx << 1 | 1;
            nodes[0]->mDepth = nodes[1]->mDepth = cur->mDepth + 1;


            // copy the rows that were in cur to the corresponding 
            // child node.
            for (auto& row : cur->mRows)
            {
                auto& y = row->mValue;

                u8 px = row->mPredsGroup[cur->mPredIdx[0]][cur->mPredIdx[1]];

                nodes[px]->mRows.emplace_back(std::move(row));
            }

            // we dont need to keep track of intermidate mappings so lets
            // clean the memory
            cur->mRows.clear();
            cur->mRows.shrink_to_fit();


            for (auto& node : nodes)
            {
                if (node->mDepth < maxDepth)
                {
                    switch (type)
                    {
                    case SplitType::Entropy:
                        entropySplit(node, predSize, minSplitSize);
                        break;
                    case SplitType::Random:
                        // currently assumes 3 classes...
                        randomSplit(node, minSplitSize);
                        break;
                    case SplitType::L2:
                    case SplitType::Dart:
                        L2Split(node, minSplitSize);
                        break;
                    case SplitType::L2Laplace:
                        L2LaplaceSplit(node, minSplitSize, nodeEpsilon);
                        break;
                    default:
                        throw std::runtime_error(LOCATION);
                        break;
                    }
                }
            }

            //std::cout << "push loss " << nodes[0]->mLoss << std::endl;
            //std::cout << "push loss " << nodes[1]->mLoss << std::endl;

            // make these two nodes as being real to be split
            nextList.push(QueueItem(nodes[0], nodes[0]->mLoss));
            nextList.push(QueueItem(nodes[1], nodes[1]->mLoss));

            // update the total depth of the tree, used for debugging 
            mDepth = std::max(mDepth, nodes[0]->mDepth);

            // update the total number of node, used for debugginh
            mNodeCount += 2;
        }
        else
        {
            // if we couldn't find a good split, then make this
            // node as a leaf node. We will compute its value in a second
            mLeafNodes.push_back(cur);
            // this node should not be split any more...
        }
    }


    mLeafCount = mLeafNodes.size();

    // for each leaf node, compute its average label and use that as a prediction
    // the majority label could work too if we are not doing ABA boosting.
    while (mLeafNodes.size())
    {
        TreeNode* cur = mLeafNodes.back();
        mLeafNodes.pop_back();

        double  sum = 0;

        // sum the labels for all records mapped to this node
        for (auto i = 0; i < cur->mRows.size(); ++i)
        {
            sum += cur->mRows[i]->mValue;
        }

        // compute the average.
        cur->mValue = sum / cur->mRows.size();
    }
}

void MLTree::selectFeatures(std::vector<DbTuple> & db, SplitType type)
{
    if (type == SplitType::Random)
    {
        mFeatureSelection.resize(db[0].mPredsGroup.size());
        bool bb = false;
        for (u64 i = 0; i < mFeatureSelection.size(); ++i)
        {
            // select the feature with prob. 1/2
            mFeatureSelection[i] = mPrng.getBit();
            bb |= mFeatureSelection[i];
        }

        // make sure at least one feature is selceted
        if (bb == false)
        {
            mFeatureSelection[mPrng.get<u64>() % mFeatureSelection.size()] = 1;
        }
    }
    else
    {
        mFeatureSelection.resize(0);
        mFeatureSelection.resize(db[0].mPredsGroup.size(), 1);
    }
}

void MLTree::randomSplit(TreeNode * cur, const u64 &minSplitSize)
{
    u64 predSize = 0;
    for (u64 k = 0; k < mFeatureSelection.size(); ++k)
    {
        if (mFeatureSelection[k])
            predSize += cur->mRows[0]->mPredsGroup[k].size();
    }

    // validNodes holds a list of node that are of sufficient size
    // counts hold the size of the nodes.
    std::vector<std::array<u64, 2>> validNodes, counts(predSize);
    validNodes.reserve(predSize);

    // list through each rows of db in the current treenode
    for (auto j = 0; j < cur->mRows.size(); ++j)
    {
        // for each record that is in the current node
        // see what node it would be mapped to if we used 
        // the j'th split

        auto& row = cur->mRows[j];
        auto y = row->mValue; // the label

        u64 i = 0;
        for (u64 k = 0; k < mFeatureSelection.size(); ++k)
        {
            // see if this feature has been selected for this random tree.
            if (mFeatureSelection[k])
            {
                for (u64 l = 0; l < row->mPredsGroup[k].size(); ++l, ++i)
                {
                    // px is the index of what node this record would be mapped to.
                    // its either 0 or 1 (left or right node)
                    u8 px = row->mPredsGroup[k][l];

                    // add this records data to the running total
                    counts[i][px]++;

                    // if this split i just became large enough to consider, 
                    // then add it to the list of valid nodes. Later we 
                    // will ranodmly pick one....
                    if (counts[i][px] == minSplitSize &&
                        counts[i][1 - px] >= minSplitSize)
                    {
                        validNodes.emplace_back(std::array<u64, 2>{k, l});
                    }
                }
            }
        }
    }

    if (validNodes.size())
    {
        u64 idx = mPrng.get<u64>() % validNodes.size();

        // ensurestemp that the largest nodes get split....
        cur->mLoss = -double(cur->mRows.size());

        cur->mPredIdx = validNodes[idx];
    }

}

void MLTree::L2Split(TreeNode * cur, const u64 & minSplitSize)
{

    struct splitUpdate
    {
        splitUpdate() : mYSum(0), mSize(0) {}
        YType mYSum;
        u64 mSize;
    };

    std::vector<std::vector<std::array<splitUpdate, 2>>>
        updates(mFeatureSelection.size());

    for (u64 i = 0; i < updates.size(); ++i)
    {
        // the L2 loss function need the size of each node
        // and the sum of the labels for this node. 
        // This pair is for the right node of a canidate split

        updates[i].resize(cur->mRows[0]->mPredsGroup[i].size());
    }


    for (auto j = 0; j < cur->mRows.size(); ++j)
    {
        // for each record that is in the current node
        // see what node it would be mapped to if we used 
        // the j'th split

        auto& row = cur->mRows[j];

        auto y = row->mValue;


        for (u64 k = 0; k < mFeatureSelection.size(); ++k)
        {
            // see if this feature has been selected for this random tree.
            if (mFeatureSelection[k])
            {
                for (u64 l = 0; l < row->mPredsGroup[k].size(); ++l)
                {
                    // px is the index of what node this record would be mapped to.
                    // its either 0 or 1 (left or right node)
                    u8 px = row->mPredsGroup[k][l];

                    // add this records data to the running total
                    updates[k][l][px].mYSum += y;
                    updates[k][l][px].mSize++;
                }
            }
        }
    }

    // now lets compute which split is the best using the L2 loss function

    cur->mLoss = 9999999999999;;

    for (u64 i = 0; i < updates.size(); ++i)
    {
        for (u64 l = 0; l < updates[i].size(); ++l)
        {
            // for each potential split, make sure that it is of minimal size
            if (updates[i][l][0].mSize >= minSplitSize &&
                updates[i][l][1].mSize >= minSplitSize)
            {
                // compute the L2 loss fucntion. (its been slightly modified since we only care about relative improvement)
                i64 variance
                    = -(i64)updates[i][l][0].mYSum * (i64)updates[i][l][0].mYSum / (i64)updates[i][l][0].mSize
                    + -(i64)updates[i][l][1].mYSum * (i64)updates[i][l][1].mYSum / (i64)updates[i][l][1].mSize
                    ;

                // if the loss using this split is less that the 
                // loss incured by the other splits, make this one 
                // as the best
                if (variance < cur->mLoss)
                {
                    cur->mLoss = variance;
                    cur->mPredIdx = { i,l };
                }
            }
        }
    }
}

void MLTree::L2LaplaceSplit(TreeNode * cur, const u64 & minSplitSize, double nodeEpsilon)
{

    struct splitUpdate
    {
        splitUpdate() : mYSum(0), mSize(0) {}
        YType mYSum;
        u64 mSize;
    };

    std::vector<std::vector<std::array<splitUpdate, 2>>>
        updates(mFeatureSelection.size());

    for (u64 i = 0; i < updates.size(); ++i)
    {
        // the L2 loss function need the size of each node
        // and the sum of the labels for this node. 
        // This pair is for the right node of a canidate split

        updates[i].resize(cur->mRows[0]->mPredsGroup[i].size());
    }

    double average = 0;
    u64 count = 0;

    for (auto j = 0; j < cur->mRows.size(); ++j)
    {
        // for each record that is in the current node
        // see what node it would be mapped to if we used 
        // the j'th split

        auto& row = cur->mRows[j];

        auto y = row->mValue;


        for (u64 k = 0; k < mFeatureSelection.size(); ++k)
        {
            // see if this feature has been selected for this random tree.
            if (mFeatureSelection[k])
            {
                for (u64 l = 0; l < row->mPredsGroup[k].size(); ++l)
                {
                    // px is the index of what node this record would be mapped to.
                    // its either 0 or 1 (left or right node)
                    u8 px = row->mPredsGroup[k][l];

                    // add this records data to the running total
                    updates[k][l][px].mYSum += y;
                    updates[k][l][px].mSize++;
                    average += std::abs(y);
                    ++count;
                }
            }
        }
    }

    average = average / count;

    Laplace lap(mPrng.get<u64>(), average / nodeEpsilon);

    // now lets compute which split is the best using the L2 loss function
    cur->mLoss = 9999999999999;;

    for (u64 i = 0; i < updates.size(); ++i)
    {
        for (u64 l = 0; l < updates[i].size(); ++l)
        {
            // for each potential split, make sure that it is of minimal size
            if (updates[i][l][0].mSize >= minSplitSize &&
                updates[i][l][1].mSize >= minSplitSize)
            {
                double noise = lap.get();

                // compute the L2 loss fucntion. (its been slightly modified since we only care about relative improvement)
                double variance
                    = -updates[i][l][0].mYSum * updates[i][l][0].mYSum / updates[i][l][0].mSize
                    + -updates[i][l][1].mYSum * updates[i][l][1].mYSum / updates[i][l][1].mSize
                    ;

                double noisyVar = noise + variance;

                // if the loss using this split is less that the 
                // loss incured by the other splits, make this one 
                // as the best
                if (noisyVar <  cur->mLoss)
                {
                    cur->mLoss = noisyVar;
                    cur->mPredIdx = { i,l };
                }
            }
        }
    }
}

void MLTree::entropySplit(TreeNode * cur, const u64 &predSize, const u64 &minSplitSize)
{
    struct splitUpdate
    {
        splitUpdate() : mYSum(0), mSize(0) {}
        YType mYSum;
        u64 mSize;
        std::array<u64, 3> classFreq;
    };

    std::vector<std::vector<std::array<splitUpdate, 2>>> updates(mFeatureSelection.size());
    for (u64 i = 0; i < updates.size(); ++i)
    {
        updates[i].resize(cur->mRows[0]->mPredsGroup[i].size());
    }


    double nodeEntropy = 0.0; // the uncertainty of the current node
    std::array<u32, 3> nodeClassFreq; // the frequency (# of records) for each class label
    nodeClassFreq.fill(0);
    // list through each rows of db in the current treenode
    for (auto j = 0; j < cur->mRows.size(); ++j)
    {
        // for each record that is in the current node
        // see what node it would be mapped to if we used 
        // the j'th split

        auto& row = cur->mRows[j];

        auto y = row->mValue; // the label

        int label = std::round(y);
        //if (label < 0 || label > 2) printf("break here\n");
        nodeClassFreq[label] ++;

        for (u64 k = 0; k < mFeatureSelection.size(); ++k)
        {
            // see if this feature has been selected for this random tree.
            if (mFeatureSelection[k])
            {
                for (u64 l = 0; l < row->mPredsGroup[k].size(); ++l)
                {
                    // px is the index of what node this record would be mapped to.
                    // its either 0 or 1 (left or right node)
                    u8 px = row->mPredsGroup[k][l];

                    // add this records data to the running total
                    updates[k][l][px].mYSum += y;
                    updates[k][l][px].classFreq[label]++;
                    updates[k][l][px].mSize++;
                }
            }
        }
    }

    // compute entropy 

    double nodep0 = 1.0*nodeClassFreq[0] / (1.0*cur->mRows.size());
    double nodep1 = 1.0*nodeClassFreq[1] / (1.0*cur->mRows.size());
    double nodep2 = 1.0 - nodep0 - nodep1;
    double epsilon = 0.0000001;

    if (nodep0 > epsilon) nodeEntropy += -nodep0*std::log2(nodep0);
    if (nodep1 > epsilon) nodeEntropy += -nodep1*std::log2(nodep1);
    if (nodep2 > epsilon) nodeEntropy += -nodep2*std::log2(nodep2);
    //printf("node entropy: %f\n", nodeEntropy);


    // now lets compute which split is the best using the L2 loss function

    // compute information gain
    double bestIG = 0.0;
    for (u64 i = 0; i < updates.size(); ++i)
    {

        for (u64 l = 0; l < updates[i].size(); ++l)
        {

            // for each potential split, make sure that it is of minimal size
            if (updates[i][l][0].mSize >= minSplitSize &&
                updates[i][l][1].mSize >= minSplitSize)
            {
                double epsilon = 1e-7;

                // entropy of the left child
                double p00 = 1.0*updates[i][l][0].classFreq[0] / (1.0*updates[i][l][0].mSize);
                double p01 = 1.0*updates[i][l][0].classFreq[1] / (1.0*updates[i][l][0].mSize);
                double p02 = 1.0 - p00 - p01;
                double entropy0 = 0.0;
                if (p00 > epsilon) entropy0 += -p00*std::log2(p00);
                if (p01 > epsilon) entropy0 += -p01*std::log2(p01);
                if (p02 > epsilon) entropy0 += -p02*std::log2(p02);

                // entropy of the right child
                double p10 = 1.0*updates[i][l][1].classFreq[0] / (1.0*updates[i][l][1].mSize);
                double p11 = 1.0*updates[i][l][1].classFreq[1] / (1.0*updates[i][l][1].mSize);
                double p12 = 1.0 - p10 - p11;
                double entropy1 = 0.0;
                if (p10 > epsilon) entropy1 += -p10*std::log2(p10);
                if (p11 > epsilon) entropy1 += -p11*std::log2(p11);
                if (p12 > epsilon) entropy1 += -p12*std::log2(p12);

                // expected entropy of children
                double p0 = 1.0*updates[i][l][0].mSize / (1.0*(updates[i][l][0].mSize + updates[i][l][1].mSize));
                double p1 = 1.0 - p0;
                double childrenEntropy = p0*entropy0 + p1*entropy1;

                double IG = nodeEntropy - childrenEntropy;	// information gain
                                                            //printf("%d, %5f, %5f\n", i, indexFractionToAttributeValue(i), IG);
                if (IG >= bestIG) {
                    bestIG = IG;
                    cur->mPredIdx = { i, l };
                }
            }
        }
    }

    //printf("Information Gain: i \t %5f \t %5f ()\n", cur->mPredIdx, indexFractionToAttributeValue(cur->mPredIdx), bestIG);

}






double MLTree::evaluate(const DbTuple & row)
{
    TreeNode* cur = &root;

    // traverse the tree until we get to a leaf. 
    while (cur->hasChildren())
    {

        auto px = row.mPredsGroup[cur->mPredIdx[0]][cur->mPredIdx[1]];
        cur = px ? cur->mRight.get() : cur->mLeft.get();

    }

    // this is a leaf node, use its prediction
    return cur->mValue;
}

u64 MLTree::getDepth()
{
    return mDepth;
}

u64 MLTree::leafCount()
{
    return mLeafCount;
}

//void MLTree::deleteNode(TreeNode *& node)
//{
//    if (node)
//    {
//        deleteNode(node->mLeft);
//        deleteNode(node->mRight);
//
//
//        delete node;
//        node = nullptr;
//    }
//
//}
