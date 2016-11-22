#include "TreeNode.h"

#include <string>
#include <sstream>

TreeNode::TreeNode()
    :mRight(nullptr),
    mLeft(nullptr),
    mPredIdx({ u64(-1),u64(-1) }), mValue(0), mIdx(1)
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
        for(auto p : row->mPredsGroup)
            sum += p[0] * p[1];

        sum += row->mValue;
    }

    return sum;
}

void TreeNode::toFile(std::ostream & out)
{
    if (mPredIdx[0] == -1)
    {
        out << "l " << mValue << std::endl;
    }
    else
    {
        out << mPredIdx[0] <<  "-"<< mPredIdx[1] << std::endl;
        mLeft->toFile(out);
        mRight->toFile(out);
    }


}
