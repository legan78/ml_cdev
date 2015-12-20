
#include <vector>
#include "common.h"

namespace tinf {

  /**
   * @brief Write a matrix to a text file.
   * @param mat Matrix to save.
   * @param fileName Full path and name of the file
   * @param writeSize Flag to write also the size of the matrix
   */    
  void saveMatrix(const cv::Mat& mat, const char* fileName,  bool writeSize)
  {
    std::ofstream file(fileName);
    
    if(writeSize)
      file<<mat.rows<<"\t"<<mat.cols<<std::endl;
    
    for(size_t i=0; i<mat.rows; i++) {
      for(size_t j=0; j<mat.cols; j++) {
        file << mat.at<double>(i,j);
          if(j != mat.cols-1) 
            file << "\t";
      }

      file<<std::endl;
    }       

    file.close();
  }
  
  /**
   * @brief Write a matrix to a text file.
   * @param mat Matrix to save.
   * @param fileName Full path and name of the file
   */    
  void saveMatrixCSV(const cv::Mat& mat, const char* fileName) {
    std::ofstream file(fileName);
    
    for(size_t i=0; i<mat.rows; i++)
      file <<";"<<i;

    file << std::endl;

    for(size_t i=0; i<mat.rows; i++) {
      file << i;
      for(size_t j=0; j<mat.cols; j++)
        file << ";"<< mat.at<double>(i,j);

      file<<std::endl;
    }
    
    file.close();
  }
  
  /**
   * @brief Write a matrix to a text file.
   * @param mat Matrix to save.
   * @param fileName Full path and name of the file
   */    
  void saveMatrixCSV2(const cv::Mat& mat, const char* fileName) {
    std::ofstream file(fileName);
    char str[10];
    
    file <<"X1";
    for(size_t i=1; i<mat.cols; i++) {
      sprintf(str, "%d", i+1);
      file <<",X"<<str;
    }
    
    file << std::endl;
    
    for(size_t i=0; i<mat.rows; i++) {
      file << mat.at<double>(i,0);

      for(int j=1; j<mat.cols; j++)
        file << ","<< mat.at<double>(i,j);

      file<<std::endl;
    }       
    
    file.close();
  }    
  
  /**
   * @brief Load matrix from file
   * @param file Full path to the file
   * @return Loaded matrix
   */    
  cv::Mat loadMatrix(const char* fileName) throw() {
    std::ifstream inputFile(fileName);
    if(inputFile.fail()) {
      std::string msg =  "Could not open file: " + std::string( fileName );
      throw std::runtime_error(msg);
    }

    std::string line;
    double val;

    std::vector< std::vector< double > > bufferMat;
    std::vector< double > bufferRow;

    unsigned int lineCounter = 1;
    while( std::getline(inputFile, line)) {
      if(!line.size()) 
        continue;

      std::stringstream ss;
      bufferRow.clear();
      ss.str( line.c_str() );

      while(ss >> val)
        bufferRow.push_back( val );

      bufferMat.push_back( bufferRow );
      lineCounter++;
    }

    inputFile.close();

    if(!bufferMat.size()) 
      return cv::Mat();

    int rows = bufferMat.size();
    int cols = bufferMat[0].size();    

    cv::Mat mat = cv::Mat::zeros(rows, cols, CV_64FC1);

    for(int i = 0; i< rows; i++)
      for(int j = 0; j<cols; j++)
        mat.at< double >(i,j) = bufferMat[i][j];    

    return mat;
  }                
  
}
