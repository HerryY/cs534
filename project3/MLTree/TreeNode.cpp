#include "TreeNode.h"

#include <string>
#include <sstream>

TreeNode::TreeNode()
    :mRight(nullptr),
    mLeft(nullptr),
    mPredIdx(-1), mValue(-1), mIdx(1)
{
}


TreeNode::~TreeNode()
{
}

double TreeNode::hash()
{

    double sum = 0;

    for (auto row : mRows)
    {
        for(auto p : row->mPreds)
            sum += p;

        sum += row->mValue;
    }

    return sum;
}

void TreeNode::toFile(std::ostream & out)
{
    if (mPredIdx == -1)
    {
        out << "l " << mValue << std::endl;
    }
    else
    {
        out << mPredIdx << std::endl;
        mLeft->toFile(out);
        mRight->toFile(out);
    }


}
