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
     * @brief Recovers path of all files contained in some directory.
     * @param dirPath Full path of folder.
     * @return Vector of strings containing names of files in directory.
     */
    std::vector<std::string> getFilesFromDir( const std::string& dirPath );
    
    /**
     * @brief Verifies if a given character is a digit
     * @param c Character to check
     * @return True if the given character is a digit.
     */
    bool is_not_digit(char c);

    /**
     * @brief Compares two numeric strings.
     * @param s1 String 1.
     * @param s2 String 2.
     * @return True if numeric value of s1 is lower than s2.
     */
    bool numeric_string_compare(const std::string& s1, const std::string& s2);
    
       
    /**
     * @brief Transforms an image 2D array into a 1D array in column form.
     * @param img Image to be transformed.
     * @return 1D vector.
     */
    cv::Mat img2Column(const cv::Mat& img);

    /**
     * @brief Transforms a 1D vector into a 2D image of corresponding size.
     * @param column Vector be transformed.
     * @param rows Number of rows in the new 2D array
     * @param cols Number of columns in the new 2D array
     * @return 2D matrix.
     */
    cv::Mat column2Img(const cv::Mat& column, int rows, int cols);    
    
    /**
     * @brief Save squence of a given set of images in a given directory.
     * @param seq Vector of images.
     * @param directory Full path of directory.
     * @param sufix Sufix of images.
     */
    void saveSequence
    ( const std::vector<cv::Mat>& seq, 
      const char* directory, const char* sufix );
      
    /**
     * @brief save image in a given format.
     * @param img Image to save.
     * @param directory Full path directory in which to save.
     * @param sufix Sufix of the name.
     * @param i Number of image
     */
	void saveImgFormat
	( const cv::Mat& img, const char* directory, const char* sufix, int i );
      
    /**
     * @brief Loads a set of images that represents a sequence.
     * @param directory Directory of storage.
     * @return Container with the set of images.
     */
    std::vector<cv::Mat> load_frame_sequence(const char* directory);
    
    /**
     * @brief Construct all psible combinations of a given set v of size k.
     * @param v Set of data.
     * @param k Size of combinations.
     * @return Matrix of combinations.
     */ 
    template<typename T>
    std::vector< std::vector<T> > myNChooseK(const std::vector<T>& v,unsigned int k){
        unsigned int sampleSize = v.size();
        /// No combination to make
        if(sampleSize == 0 || k == 0 || sampleSize < k)
            return std::vector< std::vector< T > >();

        gsl_combination * c = NULL;
        std::vector<std::vector<T> > comb_list;

        for (size_t i = 0; i<=k; i++){
            /// Generate the combination 
		    c = gsl_combination_calloc (sampleSize, i);
		    do{
                std::vector< T > temp_comb;
                /// Retrive and store combination
                for(unsigned int p = 0; p<i; p++)
					temp_comb.push_back( v[ c->data[p] ] );

                comb_list.push_back( temp_comb );

		    } while (gsl_combination_next (c) == GSL_SUCCESS);
		    gsl_combination_free (c);
		}

        return comb_list;
    }
    
    /**
     * @brief Compute the hanning function from a given number of samples
     * @param nSamples Number of samples to compute the hanning funtion
     * @return Hanning function in a Mat object
     */
    cv::Mat hanning(int nSamples);
    
    /**
     * @brief Data type for pixel map of regions. Each vector contains the pixel location
     * of the pixels assigned to the region. The region is labeled as the index of 
     * the vector in the out-most container.
     */
    typedef std::vector< std::vector< cv::Point > > regionPixelMap;
    
    /**
     * @brief Save the label for each pixel. The labels are the output from the
     * scene segmentation algorithm using the spectral clustering method. 
     * @param pixLab Map of pixel labels.
     * @param outPutFile Output file name or path to save the pixel labels.
     * @param cleanRegions Number of regions resultant from the region clustering.
     */
    void saveLabelPixels
    ( const regionPixelMap& pixLab, const char* outPutFile, int cleanRegions );
    
    /**
     * @brief Reads and load the pixel label map from the given file.
     * @param inputFile Input file or path to the file containing the pixel map.
     * @return The structure of pixel label map.
     */
    regionPixelMap loadPixelLabels(const char* inputFile);
    
    /**
     * @brief Loads combinations saved as a text files. Each file in the directory
     * corresponds to the set of combinations of size k of an array of size n. The
     * sizes are formated in the files named.
     * @param inputPath Directory where the combinations are stored.
     * @return Set of combination sets.
     */
    std::vector< std::vector< int > > loadCombinations( const char* inputPath );
    
    /**
     * @brief Compute the sine of the given Mat.
     * @param m Input Mat.
     * @param p Power for each entry
     * @return The output sinusoidal matrix.
     */
    cv::Mat sinMat(const cv::Mat& m, int p=1);

    /**
     * @brief Compute the cosine of the given Mat.
     * @param m Input Mat.
     * @param p Power for each entry
     * @return The output sinusoidal matrix.
     */    
    cv::Mat cosMat(const cv::Mat& m, int p=1);
    
    /**
     * @brief Convert a given string containing numbers into a matrix of a fixed
     * size.
     * @param str String containing numbers.
     * @param rows Number of desired rows in the matrix.
     * @param cols Number of desired columns in the matrix.
     * @return The matrix resultant from the given string.
     */
    cv::Mat string2Mat(const std::string& str, int rows, int cols);
    
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
//             ss << mat.at< T >(i, mat.cols-1);
         }
        
        return ss.str();
    }

    typedef std::pair<double, double> OptFlowDensePoint;

    std::vector<OptFlowDensePoint* > parseStringToOptFlowData(const std::string& str, unsigned int img_rows, unsigned int img_cols);

}


#endif
