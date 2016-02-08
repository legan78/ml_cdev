#include "../include/stats.h"
#include "mxTypeModel.h"


namespace tinf
{
/****************************************************************************************\
 *                                  SPHERICAL K-MEANS                                   *
\****************************************************************************************/



    /**
     * @brief Default constructor.
     */
    SPHKMeans::SPHKMeans()
    {  }

    /**
     * @brief Copy Constructor
     * @param sph Instance to copy.
     */
    SPHKMeans::SPHKMeans(const SPHKMeans& sph)
    : labels(sph.labels), centers(sph.centers.clone())
    { }

    /**
     * @brief Constructor by performing the Spherical K-means.
     * @param X Training sample of unit vectors, each row a unit vector.
     * @param K Number of clusters
     */
    SPHKMeans::SPHKMeans(const cv::Mat& X, int K)
    {

        // Init variables
        int maxSteps = 100;
        int N        = X.rows;   

        double  _pi  = 3.1416;

        labels  = std::vector<int>( N );
        centers = cv::Mat::zeros(K, X.cols, CV_64FC1);


        /// Init centers by randomly take points        
        std::vector<int> sample;        
        sample = samplingWithoutReplacement(N, K);

        /// Initial points
        for(int k = 0; k < K; k++)
        {
            X.row( sample[k]).copyTo( centers.row( k ) );
        }
        
        /// To evaluate the dot product of each data in the sample
        cv::Mat yn = cv::Mat::zeros(1, K, CV_64FC1);

        /// Main Loop for clustering
        for(int t = 0; t < maxSteps; t++)
        {
            yn = cv::Scalar::all(0.f);

            // Data assingment for each vector
            for(int n = 0; n <N; n++)
            {
                const cv::Mat& xn = X.row(n);

                // Evaluate cosine similarity
                for(int k = 0; k<K; k++)
                    yn.at< double >(0, k) = xn.dot( centers.row(k) );

                /// Get the cluster with maximum cosine similarity
                cv::Point p;
                cv::minMaxLoc(yn, 0, 0,0, &p);
                
//                std::cout << yn << std::endl;

                labels[n] = p.x;            
            }

            centers = cv::Scalar::all(0.f);

            /// Centroid estimation
            for(int n = 0; n <N; n++)
            {
                /// Compute sum of vectors
                int k = labels[n];
                cv::Mat row = centers.row(k);

                row = row + X.row(n);

                row.copyTo( centers.row(k) );
            }

            /// Compute normalization of vectors
            for(int k = 0; k<K; k++)
            {
                cv::Mat row = centers.row(k);
                double norm = sqrt( row.dot( row ) );

                /// Avoid zero division
                double norm1 = (norm == 0.0)?1.0: norm; 
                row = row / norm1;

                row.copyTo( centers.row(k) );
            }
        }               


    }

    /**
     * @brief Assignation operator.
     * @param sph Instance to copy.
     * @return Reference to this instance.
     */
    SPHKMeans& SPHKMeans::operator=(const SPHKMeans& sph)
    {
        labels = sph.labels;
        centers = sph.centers.clone();

        return  *this;
    }

    /**
     * @brief Get the labels of the training data.
     * @return Constant reference to the labels of the training data.
     */
    const std::vector< int >& SPHKMeans::getTrainingLabels()const
    {
        return labels;
    }

    /**
     * @brief Get the centers of the clusters of the training data
     * @return Constant reference to the centers found.
     */
    const cv::Mat& SPHKMeans::getCenters()const
    {
        return centers;
    }    
    
}
