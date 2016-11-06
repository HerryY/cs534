#pragma once
// This file and the associated implementation has been placed in the public domain, waiving all copyright. No restrictions are placed on its use. 

#include <exception>
#include <string>
#include <sstream>
#include <iostream>



    class not_implemented : public std::exception
    {
        virtual const char* what() const throw()
        {
            return "Case not implemented";
        }
    };

