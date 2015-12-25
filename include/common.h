#ifndef _COMMON_H_
#define _COMMON_H_

#include <opencv2/opencv.hpp>

#include <stdlib.h>
#include <fstream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "stats.h"
#include "../libCommon/dbgLogger.h"

// Package to generate faster combinations
#include <gsl/gsl_combination.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>


namespace tinf
{
    
/**
 * @brief Minimum standard deviation
 */
#define TINF_MINIMUM_STD_DEV    0.004
    
    
    /**
     * @brief Write a matrix to a text file.
     * @param mat Matrix to save.
     * @param fileName Full path and name of the file
     * @param writeSize Flag to write also the size of the matrix
     */
    void saveMatrix(const cv::Mat& mat, const char* fileName, bool writeSize=false);
    
    /**
     * @brief Write a matrix to a text file.
     * @param mat Matrix to save.
     * @param fileName Full path and name of the file
     */    
    void saveMatrixCSV(const cv::Mat& mat, const char* fileName);
    
    /**
     * @brief Write a matrix to a text file.
     * @param mat Matrix to save.
     * @param fileName Full path and name of the file
     */    
    void saveMatrixCSV2(const cv::Mat& mat, const char* fileName)    ;
    
    /**
     * @brief Load matrix from file
     * @param file Full path to the file
     * @return Loaded matrix
     */
    cv::Mat loadMatrix(const char* file)throw();
    
    /**
     * @brief Conver a matrix into a given string. The matrix rows are concatenated
     * into a one big string row containing all elements in the matrix.
     * @param mat input matrix to convert.
     * @return Resultant row string of elements in the matrix.
     */
    template<typename T = double>
    std::string mat2String(const cv::Mat& mat){
        std::stringstream ss;
        if(!mat.rows || !mat.cols) return ss.str();
        
         for(unsigned int i = 0; i< mat.rows; i++){
             for(unsigned int j=0; j<mat.cols; j++){
                 ss << mat.at< T >(i, j) 
                    << " ";
             }
         }
        
        return ss.str();
    }
}


#endif
