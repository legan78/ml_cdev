#include "dbgLogger.h"
#include <sstream>

#include <cstdarg>
#include <cstdio>

namespace tinf
{
    /**
    * @brief Output verbose
    * @param TAG Tag of the class dumping the logs
    * @param msg Log Msg
    */
    void DbgLogger::LOGI(const std::string& TAG, const std::string& state, const std::string& msg, ...){
        /// Retrieve variable arguments
        
        unsigned int length = msg.size()*2;
        char buffer[length];
        
        std::va_list arg;
        va_start(arg, msg);
        std::vsnprintf(buffer, length, msg.c_str(), arg);
        va_end(arg);

        std::cout << "[INFO] "
                  << TAG
                  << "( " 
                  << state 
                  << " )"
                  << ": "
                  << buffer
                  << std::endl;
        
    }
    
    
    const std::string  DbgLogger::bad_input_exception::TAG = "BadInput";
    
    DbgLogger::bad_input_exception::bad_input_exception
    ( const std::string& dumperTag, const std::string& funcName, const std::string& _msg){
        std::stringstream ss;
        ss << "[" << TAG << "] "
           << "(" << dumperTag << "@" 
           << funcName  << "): "
           << _msg << std::endl;
        
        ss >> msg;
    }

    const std::string&  DbgLogger::bad_input_exception::what()const{
        return msg;
    } 
    
}