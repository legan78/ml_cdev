#ifndef _STATS_H_
#define _STATS_H_

#include "distributions.h"


namespace tinf{
    // Typedef for gaussian mixture models
    typedef std::pair<cv::EM, cv::Mat> _GMM;
    
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec){
        os << "(";
        for(int i=0;i<int(vec.size())-1;i++)
            os << vec[i] << ", ";
        if(vec.size()) os << vec[vec.size()-1];
        os << ")";
        
        return os;
    }

    /**
     * Class for mixture model selection using the Bayesian information criteria.
     * The process find the best gaussian mixture for a set of patterns.
     */
    class mixtureModelSelector{
    public:
        
        /**
         * @brief Constructor
         * @param patterns Dataset of observations.
         * @param featSize Dimension of observations.
         * @param _bicLower Lower limit to use in BIC.
         * @param _bicUpper Upper limit to use in BIC.
         */
        mixtureModelSelector(const cv::Mat& patterns, int featSize, int _bicLower, int _bicUpper, int _covModel);
        
        /**
         * @brief Do model selecion by Bayesian information criteria.
         */
        void process();
        
        /**
         * @brief Get the bic estimation in the procedure of selection.
         * @return A vector containing the BIC estimation in the range.
         */
        cv::Mat& getBIC();
        
        /**
         * @brief Get the best label found by BIC.
         * @return A vector of best label correspondance.
         */
        cv::Mat& getLabels();
        
        /**
         * @brief Get the likelihood of all the estimated in BIC.
         * @return A vector containing the likelihood of the BIC process.
         */
        cv::Mat& getLogLikelihood();
        
        /**
         * @brief Get the set of mixture models.
         * @return Container of mixture models.
         */
        std::vector<cv::EM>& getRegionModels();
        
    private:
        
        std::vector<cv::EM> selected_model;     // Container of mixture models
        
        cv::Mat             BIC;                // Matrix of BIC of process
        cv::Mat             logLikelihood;      // Matrix of log likelihood of process
        cv::Mat             labels;             // Matrix of best labels for data
        cv::Mat             trainData;          // Observations
        
        int                 bicLower;           // Lower limit to use in bic    
        int                 bicUpper;           // Upper limit to use in bic
        int                 totalOfRegions;     // Total number of set of patterns to model
        int                 featureDimension;   // Feature dimension
        int                 covModel;           // Covariance model to use in EM
        
    };
    
    /*
     * Class to implement the artificial clustering. This is done mainly for 
     * saving a model with an artificial cluster. This is why it inherites
     * from the class of EM of OpenCV.
     */
    class mixtureOfGaussians: public cv::EM
    {
        public:
            
            /**
             * @brief Set a covariance matrices to the model.
             * @param otherCovs New covariances.
             */
            void setCovs(const std::vector<cv::Mat>& otherCovs);
            
            /**
             * @brief Set a set of means
             * @param otherMeans New means
             */
            void setMeans(const cv::Mat& otherMeans);
            
            /**
             * @brief Set new responsibilities.
             * @param otherWeigths New responsibilities.
             */
            void setWeights(const cv::Mat& otherWeigths);
            
            /**
             * @brief Get the covariances from the current trained model.
             * @return A container of covariances.
             */
            std::vector<cv::Mat>& getCovs();
            
            /**
             * @brief Get the estimated means.
             * @return Matrix of current estimated means.
             */
            cv::Mat& getMeans();
            
            /**
             * @brief Get the estimated weights.
             * @return Matrix of current estimated weights.
             */
            cv::Mat& getWeights();
            
            /**
             * @brief Get number of current clusters.
             * @return Number of current clusters.
             */
            int& getNClusters();
            
    };

    /**
     * 
     * @param X
     * @param n_states
     * @return 
     */
    cv::Mat discretize_data(const cv::Mat& X, const std::vector<unsigned int>& n_states)throw();
    
    /**
     * @brief Method to compute sample correlation between two signals.
     * @param X First signal.
     * @param Y Second signal.
     * @param meanX Mean of first signal.
     * @param meanY Mean of second signal.
     * @param stddevX Standard deviation of first signal.
     * @param stddevY Standard deviation of second signal.
     * @return Value of sample correlation of signal X and Y.
     */
    double computeCorrelation
    ( const cv::Mat& X, const cv::Mat& Y, double meanX, 
      double meanY, double stddevX, double stddevY );
    
    /**
     * @brief Method to compute sample correlation coefficient of two signals.
     * @param X First signal.
     * @param Y Second signal.
     * @return Value of sample correlation of signal X and Y.
     */
    double computeCorrelation( const cv::Mat& X, const cv::Mat& Y );
    
    /**
     * @brief Method to calculate a covariance matrix of a data colection.
     * @param X Data matrix. Each row is an observation and a column is a feature.
     * @return Covariance matrix
     */
    cv::Mat covariance(const cv::Mat& X);
    
    /**
     * @brief Method to implement the canonical correlation between 2 set of observations.
     * @param X Data matrix. Each row is an observation.
     * @param Y Data matrix. Each row is an observation.
     * @return Correlation coeficients. The number of coeficients equals to the 
     * rank of data matrices.
     */
    cv::Mat canonicalCorrelation(const cv::Mat& X, const cv::Mat& Y);
    
    /**
     * @brief Computes mutual information considering a time delay of patterns.
     * @param X Discrete data corresponding to a cluster label.
     * @param Y Discrete data corresponding to a cluster label.
     * @param Kx Number of clusters for data X.
     * @param Ky Number of clusters for data Y.
     * @param T Time delay.
     * @return Mutual information function for signals and time delay.
     */
    std::vector<double>
    timeDelayMutualInformation
    ( const cv::Mat& X, const cv::Mat& Y, int Kx, int Ky, int T, 
            cv::Mat* binX = NULL, cv::Mat* binY = NULL );
    
    
   /**
    * @brief Generates a histogram given a configuration of its parents.
    * @param data  States data to construct the histograms.
    * @param currentConf   Vector of parents configuration.
    * @param parents   Vector of parent index.
    * @param child     Index of child.
    * @param nChildStates  Number of child states
    * @return Hitogram of child given one parent configuration
    */
   void cond_histogram(const cv::Mat& data
                      ,const std::vector<unsigned int>& currentConf
                      ,const std::vector<unsigned int>& parents
                      ,cv::Mat& hist
                      ,cv::Mat& counter
                      ,unsigned int child
                      ,unsigned int nChildStates )throw();
    
     /**
      * @brief Constructs a set of histograms for the child node given each parents configuration.
      * @param data Discrete data.
      * @param states Number of states for each variable.
      * @param parentIndex Index for the parents.
      * @param hist Resultant distribution represented as a histogram.
      * @param count Frequency cummulator for child states.
      * @param childIndex Index of child. 
      */
     void build_cond_dist(const cv::Mat& data
                          ,const std::vector<unsigned int>& states
                          ,const std::vector<unsigned int>& parentIndex
                          ,cv::Mat& confDist, cv::Mat& count
                          ,unsigned int childIndex)throw();
   
     
    /**
     * @brief Computes Bayessian information criteria score.
     * @param data Data of parents and child cinfiguration.
     * @param states Number of states variables in the graph.
     * @param setOfParents Set of parent index.
     * @param childIndex Index of child.
     * @return Bic score for the child given its parents configuration.
     */
    double bicScore(const cv::Mat& data
                    ,const std::vector<unsigned int>& states
                    ,const std::vector<unsigned int>& parents
                    ,unsigned int childIndex, cv::Mat* cond_dist = NULL, cv::Mat* cond_count=NULL);

   /**
    * @brief Computes BIC score for a given data and its parents.
    * @param data Data for child node.
    * @param states Number of states for each node in the graph.
    * @param setOfParents Set of parent index.
    * @return Bic score of child givent its parents
    */
    double bicScore(const cv::Mat& data
                    ,const std::vector<unsigned int>& states
                    ,std::vector<std::vector<unsigned int> >& setOfParents );

     /**
      * @brief Construct histogram given a set of data.
      * @param data Discrete variable data.
      * @param states Number of variable staes.
      * @param hist Resulting histogram.
      * @param count Frequency of child states.
      */
     void histogram(const cv::Mat& data, int states, cv::Mat& hist, cv::Mat& count);
     
    
    /**
     * @brief Selects the best model of a mixture of Gaussians given from a set of models.
     * @param data Data to use to perform mixture generation.
     * @param bicLowRange Lower number of clusters to consider.
     * @param bicUpperRange Upper number of clusters to consider.
     * @param path2Save  Full path in where to save results of clustering process.
     * @param BIC Bic function resultant.
     * @param selected_model Selected mixture of gaussians model.
     * @param labels Resultant labels of clustering process.
     */    
    void computeBICSelection
    ( const cv::Mat& data, int bicLowRange, 
      int bicUpperRange, cv::Mat& BIC, 
      cv::EM& selected_model, cv::Mat& outLogLikelihood, 
      cv::Mat& labels, int covModel = cv::EM::COV_MAT_DEFAULT );
    
    
    /**
     * @brief Estimates the density of data using mixture of gaussians.
     * @param regionMovingPatterns Moving foreground patterns.
     * @param regionStaticPatterns Statuc foreground patterns.
     * @param bicUpperRange Low number of cluster to be considered.
     * @param confLabels Configurations of variables in time.
     * @param bicLowRange Great number of clusters to be considered.
     * @param path2Save Path in where to save results.
     * @return Container of mixture of models.
     */    
    std::vector<_GMM> regionMOGDensityEstimation
    ( const cv::Mat& regionMovingPatterns, 
      const cv::Mat& regionStaticPatterns, 
      cv::Mat& confLabels,
      int bicUpperRange, int bicLowRange,
      const std::string& path2Save );
    
    
    /**
     * @brief Finds within cluster variance given a set of clusters and a dataset.
     * @param data Clustered data.
     * @param centers Set of centers of the clusters.
     * @param labels Label for each data point.
     * @return Within cluster variance.
     */
    double withinClusterVariance
    ( const cv::Mat& data, const cv::Mat& centers, const std::vector<int>& labels );    
    
    /**
     * @brief Computes mean and standar deviation for a univariate data
     * @param data Data.
     * @param mean Resultant mean.
     * @param std Resultant standard deviation
     */
    void computeMeanStd(const cv::Mat& data, double& mean, double& std );
    
    /**
     * @brief computes normalized correlation of two images.
     * @param m1 First image.
     * @param m2 Second image.
     * @return Normalized correlation
     */
    double computeNormalizedCorrelation(const cv::Mat& m1, const cv::Mat& m2);
    
    /**
     * @brief Computes normalized cross correlation for two images.
     * @param patch Image patch.
     * @param templ Template to match.
     * @return Normalized cross correlation.
     */
    double normalizeCrossCorrelation(const cv::Mat& patch, const cv::Mat& templ);
    
    /**
     * @brief Computes the Kullback-Leiber divergence for two distributions.
     * @param p First distribution.
     * @param q Second distribution.
     * @return Kullback divergence.
     */
    double kullbackLeiblerDivergence( const cv::Mat& p, const cv::Mat& q );
    
    /**
     * @brief Computes the histogram of an image.
     * @param img Image.
     * @return Histogram of image.
     */
    cv::Mat imgHistogram(const cv::Mat& img);
    

    /**
     * @brief Generates a sample of size n from a Population of size N by indicator
     * variables indicating if a data point is selected.
     * @param N Size of population.
     * @param n Size of sample.
     * @return Indicator vector with 0 at the selected entries.
     */
    std::vector<int> samplingNoReplace(int N, int n);    
    
    /**
     * @brief Generates a sample of size n from a Population of size N by index selection.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Vector containing the index of the selected samples.
     */
    std::vector<int> samplingWithoutReplacement(int populationSize, int sampleSize);
    
        
    /**
     * @brief Get a sample of data from a set of data.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Matrix of samples.
     */
    cv::Mat getSample(const cv::Mat& population, int sampleSize);
    
    
    /**
     * @brief Compute the Mutual information of two random variables with fixed number
     * of bins.
     * @param x Signal.
     * @param y Signal.
     * @param Kx Number of bins for signal x.
     * @param Ky Number of bins for signal y.
     * @return Mutual information at bins Kx and Ky for signal x and y
     */
    double MI(const cv::Mat& x, const cv::Mat& y, unsigned int Kx, unsigned int Ky);

    /**
     * @brief Compute the time delay mutual information by computing the MI by chunks.
     * @param x Signal.
     * @param y Signal.
     * @param _bx Range of bins to consider for signal x.
     * @param _by Range of bins to consider for signal y.
     * @param tdmi output time delay mututal information value.
     * @param bestLag Lag that maximizes the mutual information of both signals.
     * @param doMic Flag to indicate if the mutual information coefficient must be 
     * calculated.
     */
    void computeChunkTDMI( const cv::Mat& x, const cv::Mat& y 
                          ,const std::vector< unsigned int >& _bx
                          ,const std::vector< unsigned int >& _by
                          , unsigned int n_chunks
                          ,double& tdmi, int& bestLag, bool doMic );    
    
    /**
     * @brief Compute time delay mutual information and compute the mutual information
     * coefficient, and select the best that maximizes MI within a range of bins for
     * the signals.
     * @param x Signal. 
     * @param y Signal.
     * @param _bx Range of bins for the signal x.
     * @param _by Range of bins for the signal y.
     * @param tdmi Maximum mutual information.
     * @param bestLag Lag that maximazes the mutual information,
     * @param T Absolute value of the window lag.
     */
    void computeMicTDMI( const cv::Mat& x, const cv::Mat& y, 
                         const std::vector< unsigned int >& _bx, 
                         const std::vector< unsigned int >& _by,
                         double& tdmi, int& bestLag, int T );    
    
    
    void elongatedKMeans
    ( const cv::Mat& X, 
      int K, std::vector<int>& labels, 
      cv::Mat& centers, 
      std::vector<int>& counter, 
      double epsilon, 
      double lambda, 
      bool lastZero );    

/********************************************************************************
 *                                                                              *
 *                                                                              *
 *                              CIRCULAR DATA METHODS                           *
 *                                                                              *
 *                                                                              *
 ********************************************************************************/
    
    /**
     * @brief Computes the circular median for circular data.
     * @param mat Matrix of circular data.
     * @return Circular median.
     */
    double circularMedian( const cv::Mat& mat );
    
    /**
     * @brief Computes the circular correlation coefficient for circular data.
     * @param X First circular data.
     * @param Y Second circular data.
     * @return Circular correlation coefficient.
     */
    double circularCorrelationCoefficient(const cv::Mat& X, const cv::Mat& Y);
    
    
    /**
     * @brief Computes the circular correlation coefficient for circular data.
     * @param X First circular data.
     * @param Y Second circular data.
     * @return Circular correlation coefficient.
     */
    double circularCorrelationCoefficient(const cv::Mat& X, const cv::Mat& Y, const cv::Mat& f1, const cv::Mat& f2, double tau=1.0);
    
   /**
    * @brief Computes K-Means for circular data.
    * @param X Circular dataset.
    * @param K Number of clusters to cluster into.
    * @param labels Best labels found.
    * @param centers Best centers found.
    */
    void sphericalKmeans( const cv::Mat& X, int K, 
                          std::vector<int>& labels,
                          cv::Mat& centers);
    
}


#endif