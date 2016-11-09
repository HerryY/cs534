#include "MLTree.h"
#include "Common\Defines.h"

#include <array>
#include "Common/Timer.h"
#include <atomic>

MLTree::MLTree()
    :mDepth(0)
{
}


MLTree::~MLTree()
{

    deleteNode(root.mLeft);
    deleteNode(root.mRight);

}



void MLTree::learn(std::vector<DbTuple>& db, u64 minSplitSize)
{
    // initialize some member variables
    mDepth = 0;
    mNodeCount = 1;

    // This will be the root node of this tree.
    // mIdx is just debug info.
    root.mIdx = 1;
    root.mDepth = 0;

    // copy the full datset into the root node. It
    // will later be partioned into the leaves through
    // recursive splits
    root.mRows.resize(db.size());
    for (u64 i = 0; i < db.size(); ++i)
    {
        root.mRows[i] = &db[i];
    }

    // add it to the next list, this indicates
    // that this node has not been split yet.
    nextList.push_back(&root);

    // This update vector holds information that we need 
    // to compute the best split value. we will have 
    // two splitUpdate s for each predicate, one for each
    // of the two nodes that each prodicate will preoduce
    u64 predSize = db[0].mPreds.size();
    std::vector<std::array<splitUpdate, 2>> updates(predSize);


    // while we still have more nodes that need splitting
    while (nextList.size())
    {
        // the pointer to the current node being split
        TreeNode*cur = nextList.back();

        // remove it from the list, marking it as being processed
        nextList.pop_back();

        // clear the old values in the update list
        for (u64 i =0; i < updates.size(); ++i)
        {
            // the L2 loss function need the size of each node
            // and the sum of the labels for this node. 
            // This pair is for the right node of a canidate split
            updates[i][0].mSize = 0;
            updates[i][0].mYSum = 0;
            
            // and this one is for the left node of a canidate split
            updates[i][1].mSize = 0;
            updates[i][1].mYSum = 0;
        }


        for (auto j = 0; j < cur->mRows.size(); ++j)
        {
            // for each record that is in the current node
            // see what node it would be mapped to if we used 
            // the j'th split
            
            auto& row = cur->mRows[j];

            auto y = row->mValue;

            for (u64 i = 0; i < predSize; ++i)
            {
                // px is the index of what node this record would be mapped to.
                // its either 0 or 1 (left or right node)
                u8 px = row->mPreds[i];

                // add this records data to the running total
                updates[i][px].mYSum += y;
                updates[i][px].mSize++;

            }
        }

        // now lets compute which split is the best using the L2 loss function
        cur->mPredIdx = -1;
        i64 bestLoss = 99999999999999;

        for (u64 i = 0; i < updates.size(); ++i)
        {
            // for each potential split, make sure that it is of minimal size
            if (updates[i][0].mSize > minSplitSize &&
                updates[i][1].mSize > minSplitSize)
            {
                // compute the L2 loss fucntion. (its been slightly modified since we only care about relative improvement)
                i64 loss
                    = -(i64)updates[i][0].mYSum * (i64)updates[i][0].mYSum / (i64)updates[i][0].mSize
                    + -(i64)updates[i][1].mYSum * (i64)updates[i][1].mYSum / (i64)updates[i][1].mSize
                    ;

                // if the loss using this split is less that the 
                // loss incured by the other splits, make this one 
                // as the best
                if (loss < bestLoss)
                {
                    bestLoss = loss;
                    cur->mPredIdx = i;
                }
            }
        }

        // if we found a predicate (at least one was of min split size), then
        // lets use it and copy our data into the new codes.
        if (cur->mPredIdx != -1)
        {

            // these are the two new nodes that were prodiced by this split
            std::array<TreeNode*, 2> nodes = { new TreeNode(),new TreeNode() };

            // connect them to the parent
            cur->mLeft = nodes[0];
            cur->mRight = nodes[1];

            // do some book keeping, used for debug
            nodes[0]->mIdx = cur->mIdx << 1;
            nodes[1]->mIdx = cur->mIdx << 1 | 1;
            nodes[0]->mDepth = nodes[1]->mDepth = cur->mDepth + 1;


            // copy the rows that were in cur to the corresponding 
            // child node.
            for (auto& row : cur->mRows)
            {
                auto& y = row->mValue;

                u8 px = row->mPreds[cur->mPredIdx];

                nodes[px]->mRows.emplace_back(std::move(row));
            }

            // we dont need to keep track of intermidate mappings so lets
            // clean the memory
            cur->mRows.clear();
            cur->mRows.shrink_to_fit();

            // make these two nodes as being real to be split
            nextList.push_back(nodes[0]);
            nextList.push_back(nodes[1]);

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






double MLTree::evaluate(const DbTuple & row)
{
    TreeNode* cur = &root;

    // traverse the tree until we get to a leaf. 
    while (cur->mPredIdx != -1)
    {

        auto px = row.mPreds[cur->mPredIdx];
        cur = px ? cur->mRight : cur->mLeft;

    }

    // this is a leaf node, use its prediction
    return cur->mValue;
}

u64 MLTree::getDepth()
{
    return mDepth;
}

void MLTree::deleteNode(TreeNode *& node)
{
    if (node)
    {
        deleteNode(node->mLeft);
        deleteNode(node->mRight);


        delete node;
        node = nullptr;
    }

}
