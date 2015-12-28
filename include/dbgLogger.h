#ifndef _DBG_LOGGER_H_
#define _DBG_LOGGER_H_

#include <cstdlib>
#include <string>
#include <iostream>



namespace tinf{
    /**
     * @brief Class to implement a debugging tool for verbosing string outputs
     * when needed for debugging a method or function. This class is a singleton.
     * Cant create an object. Only call by static methods.
     */
    class DbgLogger{
    public:
        
        /**
         * @brief Output to standard output a message of a given funcition/class.
         * @param TAG Class or function tag or name.
         * @param state State of processing in the class or function.
         * @param msg Message to be printed in the standard output.
         */
        static void LOGI(const std::string& TAG, const std::string& state, const std::string& msg, ...);
        
        class bad_input_exception{
        public:
            bad_input_exception( const std::string& dumper_tag, 
                                 const std::string& function,
                                 const std::string& msg = "Invalid input data to function.");
            
            const std::string& what()const;
        protected:
            const static std::string TAG;
            std::string msg;
        };
        
    private:
        
        DbgLogger();
        DbgLogger(const DbgLogger& dl);
        DbgLogger& operator=(const DbgLogger& dl);
        
    };
}

#endif /*_DBG_LOGGER_H_*/