#include "BitIterator.h"



    BitReference::operator u8() const
    {
        return (*mByte & mMask) >> mShift;
    }

