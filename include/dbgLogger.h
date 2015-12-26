#ifndef _DBG_LOGGER_H_
#define _DBG_LOGGER_H_ 

#include <sstream>

#include <cstdarg>
#include <cstdio>

namespace tinf
{

	class DbLogger{
    /**
    * @brief Output verbose
    * @param TAG Tag of the class dumping the logs
    * @param msg Log Msg
    */
    void LOGI(const std::string& TAG, const std::string& state, 
			const std::string& msg, ...);    
    
    const std::string  DbgLogger::bad_input_exception::TAG = "BadInput";
    
   bad_input_exception::bad_input_exception
    ( const std::string& dumperTag, const std::string& funcName, const std::string& _msg){
        std::stringstream ss;
        ss << "[" << TAG << "] "
           << "(" << dumperTag << "@" 
           << funcName  << "): "
           << _msg << std::endl;
        
        ss >> msg;
    }
	const std::string  DbgLogger::bad_input_exception; 
    const std::string&  bad_input_exception::what();    
	}

	
    const std::string  DbgLogger::bad_input_exception::TAG = "BadInput";

}

#endif
