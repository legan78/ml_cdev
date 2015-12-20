
#include <vector>
#include "common.h"

namespace tinf
{
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
        
        for(int i=0; i<mat.rows; i++)
        {
            for(int j=0; j<mat.cols; j++){
                file << mat.at<double>(i,j);
                if(j != mat.cols-1) file << "\t";
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
    void saveMatrixCSV(const cv::Mat& mat, const char* fileName)
    {
        std::ofstream file(fileName);
        
        for(int i=0; i<mat.rows; i++)
            file <<";"<<i;
        
        file << std::endl;
        
        for(int i=0; i<mat.rows; i++)
        {
            file << i;
            
            for(int j=0; j<mat.cols; j++)
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
    void saveMatrixCSV2(const cv::Mat& mat, const char* fileName)
    {
        std::ofstream file(fileName);
        char str[10];
        
        file <<"X1";
        for(int i=1; i<mat.cols; i++)
        {
            sprintf(str, "%d", i+1);
            file <<",X"<<str;
        }
        
        file << std::endl;
        
        for(int i=0; i<mat.rows; i++)
        {
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
    cv::Mat loadMatrix(const char* fileName) throw(){
        std::ifstream inputFile(fileName);
        if( inputFile.fail() ){
            std::string msg =  "Could not open file: " + std::string( fileName );
            throw std::runtime_error(msg);
        }

        std::string line;
        double val;

        std::vector< std::vector< double > > bufferMat;
        std::vector< double > bufferRow;

        unsigned int lineCounter = 1;
        while( std::getline(inputFile, line)){
            if(!line.size()) continue;
            std::stringstream ss;
            bufferRow.clear();
            ss.str( line.c_str() );
            while( ss >> val ){
                //ss >> val;
                bufferRow.push_back( val );
            }
            bufferMat.push_back( bufferRow );
            lineCounter++;
        }

        inputFile.close();

        if( !bufferMat.size() ) return cv::Mat();

        int rows = bufferMat.size();
        int cols = bufferMat[0].size();    

        cv::Mat mat = cv::Mat::zeros(rows, cols, CV_64FC1);

        for(int i = 0; i< rows; i++)
            for(int j = 0; j<cols; j++)
                mat.at< double >(i,j) = bufferMat[i][j];    

        return mat;
    }                
    
    /**
     * @brief Recovers path of all files contained in some directory.
     * @param dirPath Full path of folder.
     * @return Vector of strings containing names of files in directory.
     */    
    std::vector<std::string> getFilesFromDir
    ( const std::string& dirPath )
    {
        DIR *dp;
        struct dirent *dirp;
        std::vector<std::string> imgFiles;

        // Opening the source directory
        dp = opendir( dirPath.c_str() );

        while( (dirp = readdir(dp)) != NULL ){
            if( dirp->d_name[0] != '.' ){
                // Get all files existing in the directory and generate 
                // the output chunk name files
                imgFiles.push_back( std::string( dirPath + "/" + dirp->d_name ) );
            }
        }    

        return imgFiles;
    }
    

    std::vector<OptFlowDensePoint* > parseStringToOptFlowData(const std::string& str, unsigned int img_rows, unsigned int img_cols) {
        std::stringstream ss;
        ss.str(str.c_str());

        std::vector<OptFlowDensePoint*> optFlowData(img_cols*img_rows, NULL);

        std::pair<int,int> pixel;
        std::pair<double, double> flowData;

       // Read row, column, direction, magnitude 
        while( (ss >> pixel.first >> pixel.second >> flowData.first >> flowData.second) ){
            optFlowData[pixel.first*img_cols + pixel.second]  = new std::pair<double, double>(flowData);
        }

        return optFlowData;
    }



    /**
     * @brief Verifies if a given character is a digit
     * @param c Character to check
     * @return True if the given character is a digit.
     */    
    bool is_not_digit(char c)
    {
        return !std::isdigit(c);
    }

    /**
     * @brief Compares two numeric strings.
     * @param s1 String 1.
     * @param s2 String 2.
     * @return True if numeric value of s1 is lower than s2.
     */    
    bool numeric_string_compare(const std::string& s1, const std::string& s2)
    {
        // handle empty strings...

        int s1Size = s1.size();
        int s2Size = s2.size();
        
        std::string ss1, ss2;
        
        for(int i=s1Size-5; i>=0; i--)
        {
            if(s1[i]=='_')break;
            ss1.push_back(s1[i]);
        }
        
        for(int i=s2Size-5; i>=0; i--)
        {
            if(s2[i]=='_')break;
            ss2.push_back(s2[i]);
        }

        std::string subS1 = std::string(ss1.rbegin(), ss1.rend());
        std::string subS2 = std::string(ss2.rbegin(), ss2.rend());
        
        int a = atoi(subS1.c_str());
        int b = atoi(subS2.c_str());
        
        return (a<b);
        
    }
    
    
    /**
     * @brief Transforms an image 2D array into a 1D array in column form.
     * @param img Image to be transformed.
     * @return 1D vector.
     */
    cv::Mat img2Column(const cv::Mat& img)
    {
        // Column matrix
        cv::Mat col = cv::Mat::zeros(img.rows*img.cols, 1, CV_64FC1);

        // Fill column matrix
        for(unsigned int i=0, k=0; i<img.rows;i++)
            for(unsigned int j=0; j<img.cols; j++, k++)
                col.at<double>(k,0) = img.at<double>(i, j);

        return col;
    }

    /**
     * @brief Transforms a 1D vector into a 2D image of corresponding size.
     * @param column Vector be transformed.
     * @param rows Number of rows in the new 2D array
     * @param cols Number of columns in the new 2D array
     * @return 2D matrix.
     */
    cv::Mat column2Img(const cv::Mat& column, int rows, int cols)
    {
        // Matrix
        cv::Mat img = cv::Mat::zeros(rows, cols, CV_64FC1);

        // Fill matrix with colum data
        for(unsigned int i=0, k=0; i<rows; i++)
            for(unsigned int j=0; j<cols; j++, k++)
                img.at<double>(i,j) = column.at<double>(k, 0);

        return img;        
    }
    
    
    /**
     * @brief Save squence of a given set of images in a given directory.
     * @param seq Vector of images.
     * @param directory Full path of directory.
     * @param sufix Sufix of images.
     */
    void saveSequence
    ( const std::vector<cv::Mat>& seq, 
      const char* directory, const char* sufix )
    {
        std::string path = directory;
        char str[10];
        
        for(int i=0; i<seq.size(); i++)
        {
            sprintf(str, "%d", i);
            //cv::imwrite( (path + sufix + "_0000" + str + ".jpg").c_str(), seq[i] );
        }
    }

    /**
     * @brief Loads a set of images that represents a sequence.
     * @param directory Directory of storage.
     * @return Container with the set of images.
     */    
    std::vector<cv::Mat> load_frame_sequence(const char* directory)
    {
         std::vector<cv::Mat> seq;
         cv::Mat tmpImg;
         
        // Read images from a stored files
        std::vector<std::string> imgs = tinf::getFilesFromDir(directory);
        // Sort images using a numeric comparison
        std::sort(imgs.begin(), imgs.end(), tinf::numeric_string_compare);
        
        // Load image sequence and store in a container
        for(unsigned int i=0; i<imgs.size();i++)
        {
            // Load images
            //tmpImg = cv::imread(imgs[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE);
//
//            // Convert to a floating point of double precision
//            tmpImg.convertTo( tmpImg, CV_64FC1 );        
            seq.push_back( tmpImg.clone() );        
        }
         
         return seq;
    }

    /**
     * @brief save image in a given format.
     * @param img Image to save.
     * @param directory Full path directory in which to save.
     * @param sufix Sufix of the name.
     * @param i Number of image
     */    
	void saveImgFormat
	( const cv::Mat& img, const char* directory, const char* sufix, int i )
	{
		std::string path = directory;
		char str[10];
		    
		sprintf(str, "%d", i);
		//cv::imwrite( (path + sufix + "_0000" + str + ".jpg").c_str(), img );
	}

   cv::Mat hanning(int nSamples)
   {
       cv::Mat w = cv::Mat::zeros(nSamples, 1, CV_64FC1);
       double pi = 3.1416;
       // for hamming filter c = 0.54
       double c=0.54;
       
       for(int i=0; i<nSamples; i++)
           w.at<double>(i,0) = c+(c-1.0)*cos( 2.0*pi*double(i)/(double(nSamples-1)) );      
       
       return w;
   }
   
   void saveLabelPixels(const std::vector<std::vector<cv::Point> >& pixLab, const char* filePath, int cleanRegions)
   {
       std::ofstream outFile(filePath);
       
        outFile << cleanRegions <<std::endl;
       
       for(int i=0; i<pixLab.size(); i++)
       {
           if(pixLab[i].size()==0)continue;
           
           outFile << pixLab[i].size() << "\t";
           
           for(int j=0; j<pixLab[i].size(); j++)
               outFile << pixLab[i][j].x << "\t" << pixLab[i][j].y << "\t";
           
           outFile << std::endl;
       }
       
       outFile.close();
   }
   
   std::vector<std::vector<cv::Point> > loadPixelLabels( const char* filePath )
   {
       std::ifstream inFile(filePath);
              
       int cleanRegions, nPixels, x, y;
       
       inFile >> cleanRegions;
       
//       std::cout<<"INFO: Loading "<<cleanRegions<<" regions."<<std::endl;
       
       std::vector<std::vector<cv::Point> > pixelLabelMap(cleanRegions);
       
       for(int i=0; i<cleanRegions; i++)
       {
           inFile >> nPixels;
//           std::cout<<"INFO: Region "<<i<<" with "<<nPixels<<" pixels."<<std::endl;
           
           pixelLabelMap[i] = std::vector< cv::Point >(nPixels);
           
           for(int j=0; j<nPixels; j++)
           {
               inFile >> x >> y;
               pixelLabelMap[i][j] = cv::Point(x, y);
           }
           
       }
       
       
       inFile.close();
       
       return pixelLabelMap;
   }
    
   
   
    std::vector< std::vector< int > > 
    loadCombinations(const char* fileName)
    {
        std::ifstream file(fileName);

        int numLines = 0;
        std::string unUsed;

        std::vector< std::vector< int > > comb;

        while( std::getline(file, unUsed) )
        {
            numLines++;

            std::stringstream ss;

            ss.str(unUsed.c_str());

            int s;

            std::vector< int > nodes;

            while( ss >> s )
            {
//                std::cout << s 
//                          << "\t";
                nodes.push_back(s);
            }
    //        std::cout << std::endl;

            comb.push_back( nodes );
        }

    //    std::cout << numLines << std::endl;

        file.close();

        return comb;
    }   
    
    
    
    cv::Mat sinMat(const cv::Mat& m, int p)
    {
        cv::Mat outPut = m.clone();
        
        for(int i=0; i< m.rows; i++)
            for(int j=0; j<m.cols; j++)
                outPut.at< double >(i, j) = pow(sin(m.at<double>(i,j)), p);
        
        return outPut;        
    }
    
    cv::Mat cosMat(const cv::Mat& m, int p)
    {
        cv::Mat outPut = m.clone();
        
        for(int i=0; i< m.rows; i++)
            for(int j=0; j<m.cols; j++)
                outPut.at< double >(i, j) = pow(cos(m.at<double>(i,j)), p);
        
        return outPut;
    }
    
   
    cv::Mat string2Mat(const std::string& str, int rows, int cols){
        
    }
    
}
