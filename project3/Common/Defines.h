#pragma once
// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use. 

#include <cinttypes>
#include <iomanip>
#include <vector>
#include <sstream>
#include <iostream>
#include <memory>
#include "Common/Timer.h"
#ifdef GetMessage
#undef GetMessage
#endif



#ifdef _MSC_VER 
#define __STR2__(x) #x
#define __STR1__(x) __STR2__(x)
#define TODO(x) __pragma(message (__FILE__ ":"__STR1__(__LINE__) " Warning:TODO - " #x))
#define ALIGNED(__Declaration, __alignment) __declspec(align(__alignment)) __Declaration 
#else
//#if defined(__llvm__)
#define TODO(x) 
//#else
//#define TODO(x) DO_PRAGMA( message ("Warning:TODO - " #x))
//#endif

#define ALIGNED(__Declaration, __alignment) __Declaration __attribute__((aligned (16)))
#endif

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)
#define LOCATION __FILE__ ":" STRINGIZE(__LINE__)


    template<typename T> using ptr = T*;
    template<typename T> using uPtr = std::unique_ptr<T>;
    template<typename T> using sPtr = std::shared_ptr<T>;

    typedef uint64_t u64;
    typedef int64_t i64;
    typedef uint32_t u32;
    typedef int32_t i32;
    typedef uint16_t u16;
    typedef int16_t i16;
    typedef uint8_t u8;
    typedef int8_t i8;

    enum Role
    {
        First = 0,
        Second = 1
    };

    extern Timer gTimer;

    template<typename T>
    static std::string ToString(const T& t)
    {
        return std::to_string(t);
    }


    inline u64 roundUpTo(u64 val, u64 step)
    {
        return ((val + step - 1) / step) * step;
    }

    void split(const std::string &s, char delim, std::vector<std::string> &elems);
    std::vector<std::string> split(const std::string &s, char delim);


    u64 log2ceil(u64);
    u64 log2floor(u64);


