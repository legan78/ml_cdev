    #include "stats.h"
#include "common.h"
#include <stdio.h>
#include <set>
#include <fstream>
#include <vector>
#include <cstdlib>

#define REDUCE

namespace tinf
{
    
    /**
     * @brief Constructor
     * @param patterns Dataset of observations.
     * @param featSize Dimension of observations.
     * @param _bicLower Lower limit to use in BIC.
     * @param _bicUpper Upper limit to use in BIC.
     */
    mixtureModelSelector::mixtureModelSelector
    (const cv::Mat& patterns, int featSize, int _bicLower, int _bicUpper, int _covModel)
    {
        featureDimension = featSize;
        totalOfRegions   = patterns.cols/featureDimension;
        bicLower         = _bicLower;
        bicUpper         = _bicUpper;
        
        trainData = patterns.clone();
        
        selected_model = std::vector< cv::EM >( totalOfRegions );

        covModel  = _covModel;
    }
    
    /**
     * @brief Get the bic estimation in the procedure of selection.
     * @return A vector containing the BIC estimation in the range.
     */
    cv::Mat& mixtureModelSelector::getBIC()
    {
        return BIC;
    }

    /**
     * @brief Get the best label found by BIC.
     * @return A vector of best label correspondance.
     */
    cv::Mat& mixtureModelSelector::getLabels()
    {
        return labels;
    }

    /**
     * @brief Get the likelihood of all the estimated in BIC.
     * @return A vector containing the likelihood of the BIC process.
     */
    cv::Mat& mixtureModelSelector::getLogLikelihood()
    {
        return logLikelihood;
    }

    /**
     * @brief Get the set of mixture models.
     * @return Container of mixture models.
     */
    std::vector<cv::EM>& mixtureModelSelector::getRegionModels()
    {
        return selected_model;
    }

    /**
     * @brief Do model selecion by Bayesian information criteria.
     */
    void mixtureModelSelector::process()
    {
        std::vector<cv::Mat> bic( totalOfRegions );
        std::vector<cv::Mat> outLogLike( totalOfRegions );
        std::vector<cv::Mat> allLab( totalOfRegions );
        // For each region in the scene find modes of patterns using EM and _GMM
        // using a random sample of the whole patterns
        
        std::cout<<"[_GMM] Computing _GMM for "<<totalOfRegions<<" regions."<<std::endl;
        
        for(int i=0, j=0; i<=trainData.cols-featureDimension; i+=featureDimension)
        {
            std::cout<<"[_GMM] Computing EM-BIC for region "<<i/featureDimension<<std::endl;
            
            cv::Mat patterns = trainData( cv::Range::all(), cv::Range(i, i+featureDimension) );

            // Do training process for a _GMM using EM and perform selection of 
            // the best number of clusters using BIC
            cv::Mat likeTemp;
            cv::Mat BICTemp;
            cv::Mat labTemp;
            cv::EM  emTemp;
            
            tinf::computeBICSelection( patterns,           // Train data
                                       bicLower,           // Lowest number of modes
                                       bicUpper,           // Greatest number of modes
                                       bic[j],             // Resultant bic for the trainig process
                                       selected_model[j],  // Best model founded by BIC
                                       likeTemp,           // Final likelihood
                                       allLab[j],          // Labels for the training data
                                       covModel );	       // Covariance model for clustering
            
            outLogLike[j++] = likeTemp.clone();
        }
        
        cv::hconcat(bic, BIC);
        cv::hconcat(outLogLike, logLikelihood);        
        cv::hconcat(allLab, labels);
        
        labels.convertTo(labels, CV_64FC1);
    }
    
    /**
     * @brief Set a covariance matrices to the model.
     * @param otherCovs New covariances.
     */    
    void mixtureOfGaussians::setCovs(const std::vector<cv::Mat>& otherCovs)
    {
        this->covs = otherCovs;
    }

    /**
     * @brief Set a set of means
     * @param otherMeans New means
     */    
    void mixtureOfGaussians::setMeans(const cv::Mat& otherMeans)
    {
        this->means = otherMeans;
    }
    
    /**
     * @brief Set new responsibilities.
     * @param otherWeigths New responsibilities.
     */    
    void mixtureOfGaussians::setWeights(const cv::Mat& otherWeigths)
    {
        this->weights = otherWeigths;
    }
    
    /**
     * @brief Get the covariances from the current trained model.
     * @return A container of covariances.
     */    
    std::vector<cv::Mat>& mixtureOfGaussians::getCovs()
    {
        return covs;
    }
    
    /**
     * @brief Get the estimated means.
     * @return Matrix of current estimated means.
     */    
    cv::Mat& mixtureOfGaussians::getMeans()
    {
        return  means;
    }
    
    /**
     * @brief Get the estimated weights.
     * @return Matrix of current estimated weights.
     */    
    cv::Mat& mixtureOfGaussians::getWeights()
    {
        return weights;
    }
    
    /**
     * @brief Get number of current clusters.
     * @return Number of current clusters.
     */    
    int& mixtureOfGaussians::getNClusters()
    {
        return nclusters;
    }
                
    
    /**
     * 
     * @param X
     * @param n_states
     * @return 
     */
    cv::Mat discretize_data(const cv::Mat& X, const std::vector<unsigned int>& n_states)throw(){
        if(!X.cols || !X.rows || X.cols != n_states.size() || !n_states.size())
            throw DbgLogger::bad_input_exception("Stats", __FUNCTION__);

        cv::Mat statesMat = cv::Mat(X.rows, X.cols, CV_32SC1);
        double EPS = 1e-8, min, max;
        
        for(unsigned int i=0; i<X.cols; i++){
            cv::Mat binX = X.col(i);
            cv::minMaxLoc(binX, &min, &max, 0, 0);
            binX = binX - min;

            cv::minMaxLoc(binX, &min, &max, 0, 0);
            max = (max == 0.0)? 1 : max;

            binX = binX*(1.0-EPS)/max;
            
            for(unsigned int j = 0; j<X.rows; j++)
                statesMat.at<int>(j,i) = floor( binX.at<double>(j,0)*n_states[i] );
        }
        
        return statesMat;
    }
    
    
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
      double meanY, double stddevX, double stddevY )
    {
        double cc = 0.0;
        int N = X.rows;
        
        // Compute covariance of the two signals
        cv::Mat xhat = X - meanX;
        cv::Mat yhat = Y - meanY;
        
        cc = xhat.dot( yhat )/( double(N-1) );
        
                
        // Compute normalizer
        double normalizer = (stddevX*stddevY>0.0)?stddevX*stddevY:1e-5;
        
        cc = cc/(normalizer);

        return cc;
    }
    
    double computeCorrelation
    ( const cv::Mat& X, const cv::Mat& Y )
    {
        int N = X.rows;
        
        double sumXY = 0.0;
        double sumX  = 0.0;
        double sumY  = 0.0;
        double sumX2 = 0.0;
        double sumY2 = 0.0;
                
        for(int n=0; n<N; n++)
        {
            sumXY += X.at<double>(n,0)*Y.at<double>(n,0);
            sumX  += X.at<double>(n,0);
            sumY  += Y.at<double>(n,0);
            sumX2 += X.at<double>(n,0)*X.at<double>(n,0);
            sumY2 += Y.at<double>(n,0)*Y.at<double>(n,0);
        }
        
        double meanX = sumX/double(N);
        double meanY = sumY/double(N);
        double varX  = sumX2/double(N) - meanX*meanX;
        double varY  = sumY2/double(N) - meanY*meanY;
        
        double num = sumXY - meanY*sumX - meanX*sumY + double(N)*meanX*meanY;
        num /= double(N-1);
        
        double norm = sqrt(varX)*sqrt(varY);
        
        double r = num/( norm );

        if(isinf(r) || isnan(r))
        {
//            std::cout<<num<<" "<<norm<<std::endl;
            if( sumX == 0.0 && sumY == 0.0) r=0;
            else r=1;
        }
        
        return r;
    }
    
    /**
     * @brief Method to calculate a covariance matrix of a data colection.
     * @param X Data matrix. Each row is an observation and a column is a feature.
     * @return Covariance matrix
     */    
    cv::Mat covariance(const cv::Mat& X)
    {
        // Centering data
        cv::Mat xCent = X.clone();
        for(int i=0; i<X.cols; i++)
        {
            cv::Mat c = X.col(i) - cv::mean(X.col(i)).val[0];
            c.copyTo(xCent.col(i));
        }
        
        // Compute multiplication
        cv::Mat cov = xCent.t()*xCent;
        
        // Unbiased estimator
        cov = cov/double(X.rows-1.0);
        
        return cov;
    }
    
    /**
     * @brief Method to implement the canonical correlation between 2 set of observations.
     * @param X Data matrix. Each row is an observation.
     * @param Y Data matrix. Each row is an observation.
     * @return Correlation coeficients. The number of coeficients equals to the 
     * rank of data matrices.
     */    
    cv::Mat canonicalCorrelation(const cv::Mat& X, const cv::Mat& Y)
    {
        // Length of features
        int featLength = X.cols;
        
        // Concatenate matrices to compute as a one only covariance
        cv::Mat XY;
        cv::hconcat(X, Y, XY);

        // Compute covatiance
        cv::Mat S = tinf::covariance(XY);

        // Epsilon matrix
        cv::Mat eps = 1e-5*cv::Mat::eye(S.rows/2,S.rows/2,CV_64FC1);

        // Extract each sub matrix covariance for conviance
        cv::Mat sxx = S( cv::Range(0,featLength), cv::Range(0,featLength) ) + eps;
        cv::Mat sxy = S( cv::Range(0,featLength), cv::Range(featLength,featLength*2) );
        cv::Mat syx = S( cv::Range(featLength,featLength*2), cv::Range(0,featLength) );
        cv::Mat syy = S( cv::Range(featLength,featLength*2), cv::Range(featLength, featLength*2) ) + eps;

        // Find canonical matrix
        cv::Mat M = sxx.inv()*sxy*syy.inv()*syx + eps;

        cv::Mat eigenval, eigenvect;
                
        // Get correlation to the power 2
        cv::eigen(M, eigenval, eigenvect);       
        
        // Get correlations
        eigenval = cv::abs(eigenval);
        cv::sqrt(eigenval, eigenval);

        return eigenval;
    }
    
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
     ( const cv::Mat& X, const cv::Mat& Y, int Kx, int Ky, int T, cv::Mat* out_binX, cv::Mat* out_binY )
     {
        cv::Mat x, y;
        X.convertTo(x, CV_64FC1);
        Y.convertTo(y, CV_64FC1);
         
//        std::cout<<x<<std::endl;
//        std::cout<<y<<std::endl;
//        '
        std::vector<double> mi;
        double minX, maxX, minY, maxY;
        double eps = 1e-12;

        cv::minMaxLoc(x, &minX, &maxX);
        cv::minMaxLoc(y, &minY, &maxY);

        x = x - minX;
        y = y - minY;
        
        cv::minMaxLoc(x, &minX, &maxX);
        cv::minMaxLoc(y, &minY, &maxY);
        
        maxX=(maxX==0.0)?1:maxX;
        maxY=(maxY==0.0)?1:maxY;
        
        x = x*(1.0-eps)/maxX;
        y = y*(1.0-eps)/maxY; 

        int n = x.rows;

        cv::Mat Px, Py, PxLog, PyLog, PxyLog;

        cv::Mat binX = x.clone();
        cv::Mat binY = y.clone();  

        for(int i=0; i<x.rows; i++)
        {
            binX.at<double>(i,0) = floor( binX.at<double>(i,0)*Kx );
            binY.at<double>(i,0) = floor( binY.at<double>(i,0)*Ky );
        }       
        
        // Fin the labels for each observation
//        std::cout<<maxY<<std::endl;
//        std::cout<<binY<<std::endl;

        for(int t = -T; t<=T; t++)
        {        
            int absLag = abs(t);

            cv::Mat Pxy = cv::Mat::zeros(Kx, Ky, CV_64FC1);

            for(int jj = 1; jj<=(n-absLag); jj++)
            {
                int kk = jj + absLag;

                if(t<0)
                    std::swap<int>(kk, jj);            

                Pxy.at<double>( (int)binX.at<double>(kk-1,0), (int)binY.at<double>(jj-1,0) )++;            

                if(t<0)
                    std::swap<int>(kk, jj);     
            }

            Pxy = Pxy/(double(n-absLag));
            Pxy = Pxy + eps;

            cv::reduce( Pxy, Px, 1, CV_REDUCE_SUM );
            cv::reduce( Pxy, Py, 0, CV_REDUCE_SUM );

            Py = Py.t();

            PxLog = Px.clone();
            PyLog = Py.clone();
            PxyLog = Pxy.clone();

            for(int i=0; i<Px.rows; i++)
                PxLog.at<double>(i,0) = log2( PxLog.at<double>(i,0) );

            for(int j=0; j<Py.rows; j++)
                    PyLog.at<double>(j,0) = log2( PyLog.at<double>(j,0) );

            for(int i=0; i<Px.rows; i++)        
                for(int j=0; j<Py.rows; j++)
                    PxyLog.at<double>(i,j) = log2( PxyLog.at<double>(i,j) );
            
            cv::multiply(Px, PxLog, PxLog);
            cv::multiply(Py, PyLog, PyLog);
            cv::multiply(Pxy, PxyLog, PxyLog);

            double Hx = -cv::sum(PxLog).val[0];
            double Hy = -cv::sum(PyLog).val[0];
            double Hxy = -cv::sum(PxyLog).val[0];

            mi.push_back(  Hx + Hy - Hxy  );     
        }
        
        if(out_binX!=NULL) (*out_binX) = binX.clone();
        if(out_binY!=NULL) (*out_binY) = binY.clone();
       
        
        return mi;
    }
    
     
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
                      ,unsigned int nChildStates )throw(){
       if(data.empty() || !currentConf.size() || !parents.size())
           throw DbgLogger::bad_input_exception("Stats", __FUNCTION__);
       
       // Histogram construction
       hist = cv::Mat::zeros(1, nChildStates, CV_64FC1);
       counter = cv::Mat::zeros(1, nChildStates, CV_64FC1);

       // Loop for histogram construction
       unsigned int count=0;
       for(unsigned int i=0; i<data.rows; i++){        
           const cv::Mat& child_data = data.row(i);        
           unsigned int childConf = child_data.at<int>(0, child);        
           bool matchConf = true;

           // Search for a configuration in data that matches the current configuration
           for(unsigned int j = 0; j<parents.size(); j++)
               if( currentConf[j] != child_data.at<int>(0, parents[j])){ 
                   matchConf = false; 
                   break;
               }

           // If the configuration has matched, then increment counters
           if(matchConf){
               hist.at<double>(0, childConf)++;
               count++;
           }        
       }
       counter = hist.clone();
       if(count>0) hist = hist/double(count); 
   }

   
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
                          ,cv::Mat& condDist, cv::Mat& count
                          ,unsigned int childIndex)throw(){
         if(data.empty() || !states.size())
             throw DbgLogger::bad_input_exception("Stats", __FUNCTION__);

         if(!parentIndex.size())
             return histogram(data.col(childIndex), states[childIndex], condDist, count);
         
         // A complete cycle on the child variable
         unsigned int nConfigurations = 1, cycle = 0;

         for(unsigned int i=0; i<parentIndex.size(); i++)
             nConfigurations *= states[ parentIndex[i] ];

         std::vector<unsigned int> currentParentsConf(parentIndex.size(), 0);

         condDist = cv::Mat::zeros(nConfigurations, states[childIndex], CV_64FC1);
         count    = cv::Mat::zeros(nConfigurations, states[childIndex], CV_64FC1);

         for(unsigned int c = 0; c< nConfigurations; c++){
             cv::Mat hist, counter;
             cond_histogram(data, currentParentsConf
                               ,parentIndex, hist, counter
                                    ,childIndex, states[childIndex]);

             hist.copyTo(condDist.row(c));
             counter.copyTo(count.row(c));

             cycle++;
             /// Compute next parent configuration. See the repeating cycle
             for(unsigned int i=0; i<currentParentsConf.size(); i++){
                 if((cycle) % (i+1) == 0){
                     if((currentParentsConf[i]+1) % (states[i]) == 0)
                         currentParentsConf[i] = 0;
                     else
                         currentParentsConf[i]++;
                 }
             }
         }
     }

     /**
      * @brief Construct histogram given a set of data.
      * @param data Discrete variable data.
      * @param states Number of variable staes.
      * @param hist Resulting histogram.
      * @param count Frequency of child states.
      */
    void histogram(const cv::Mat& data, int states, cv::Mat& hist, cv::Mat& count){
        hist = cv::Mat::zeros(1, states, CV_64FC1);
        count = cv::Mat::zeros(1, states, CV_64FC1);

        for(int i=0; i<data.rows; i++)
            hist.at<double>(0, data.at<int>(i, 0) )++;

        count = hist.clone();
        hist = hist/double(data.rows);        
    }
     
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
                    ,unsigned int childIndex, cv::Mat* cond_dist, cv::Mat* cond_count){
        if(data.empty() || !states.size())
             throw DbgLogger::bad_input_exception("Stats", __FUNCTION__);

        double bic = 0.0;
        double likelihood = 0.0;
        double penalty = 0.0;

        bool flag = false;
        
        cv::Mat logLike, logSum, dist, count;
        
        if(parents.size())
            build_cond_dist(data, states, parents, dist, count, childIndex);
        else
            histogram(data.col(childIndex), states[childIndex], dist, count);
        
        if(cond_dist)*cond_dist = dist.clone();
        if(cond_count)*cond_count = count.clone();
        
        for(unsigned int i = 0; i<count.rows; i++){
            double sum = cv::sum(count.row(i))[0];
            if(sum == 0.0){
                flag = true;
                break;
            }
        }

        // automatically sets min of numeric limits to it
        // cv::log(dist, logLike);
        logLike = cv::Mat(dist.rows, dist.cols, CV_64FC1);
        for(unsigned int i=0; i<dist.rows; i++)
            for(unsigned int j=0; j<dist.cols; j++)
                logLike.at<double>(i,j) = log(dist.at<double>(i,j));

//        if(flag){
//            DbgLogger::LOGI("STATS", __FUNCTION__, "Showig distribution, log and counts");
//            std::cout << dist << std::endl;
//            std::cout << logLike << std::endl;
//            std::cout << count << std::endl;
//        }

        cv::multiply(logLike, count, logLike);        
        cv::reduce(logLike, logSum, 1, CV_REDUCE_SUM);
        likelihood = cv::sum(logSum).val[0];
        
//        if(flag){
//            DbgLogger::LOGI("STATS", __FUNCTION__, "Showig logmult and logsum");
//            std::cout << logLike << std::endl;
//            std::cout << logSum << std::endl;
//            std::cout << likelihood << std::endl;
//            getchar();
//        }

        unsigned int bi = dist.rows*(states[childIndex]-1);
        penalty = double(bi)/double(2.0);
        bic = likelihood - log(data.rows)*penalty;
        
        if(isnan(likelihood))
        // There is at least a probability equal to zero we dont want that
        // Give the lowest posible bic score
            bic = -std::numeric_limits<double>::max();

        return bic;
    }

   /**
    * @brief Computes BIC score for a given data and its parents.
    * @param data Data for child node.
    * @param states Number of states for each node in the graph.
    * @param setOfParents Set of parent index.
    * @return Bic score of child givent its parents
    */
    double bicScore(const cv::Mat& data
                    ,const std::vector<unsigned int>& states
                    ,std::vector<std::vector<unsigned int> >& setOfParents )
    {
       double bicScore=0.0;
       double likelihood=0.0;
       double penalty=0.0;

       cv::Mat logLike, logSum;

       for(unsigned int childIndex=0; childIndex<data.cols; childIndex++){
           std::vector<unsigned int>& parents = setOfParents[childIndex];
           cv::Mat dist, count;

         //  if( parents.size() )
           build_cond_dist(data, states, parents, dist, count, childIndex);
         //  else
         //      histogram(data.col(childIndex), states[childIndex], dist, count);

           cv::log(dist, logLike);
           cv::reduce(logLike, logSum, 0, CV_REDUCE_SUM);

           for(unsigned int t=0; t<data.rows; t++){
               unsigned int xit = data.at<int>(t, childIndex);            
               likelihood += logSum.at<double>(0, xit);
           }

           unsigned int bi = dist.rows*(states[childIndex]-1);
           penalty += double(bi)/double(2.0);
       }

       bicScore = likelihood - log(data.rows)*penalty;
       return bicScore;
    }
    
    
    /**
     * @brief Estimates the density of data using mixture of gaussians.
     * @param regionMovingPatterns Moving foreground patterns.
     * @param regionStaticPatterns Statuc foreground patterns.
     * @param bicUpperRange Low number of cluster to be considered.
     * @param bicLowRange Great number of clusters to be considered.
     * @param path2Save Path in where to save results.
     * @return Container of mixture of models.
     */      
    std::vector<_GMM> regionMOGDensityEstimation
    ( const cv::Mat& regionMovingPatterns, 
      const cv::Mat& regionStaticPatterns, 
      cv::Mat& confLabels,
      int bicUpperRange, int bicLowRange,
      const std::string& path2Save )
    {
        // Resulting labels after clustering
        std::ofstream file( (path2Save + "/gmmLabelsMatlabFormat.txt").c_str());
        
        // Mixture of gaussians for regions
        std::vector< _GMM > regionMOG;
        // Matrix of Bayesian Information Criteria and log-Likelihood
        cv::Mat allBIC = cv::Mat::zeros(bicUpperRange-bicLowRange+1, 1, CV_64FC1 );
        cv::Mat loglike = cv::Mat::zeros(bicUpperRange-bicLowRange+1, 1, CV_64FC1 );
        cv::Mat allLabels = cv::Mat::zeros(regionMovingPatterns.rows, 1, CV_32SC1);
        std::vector<int> states;
        
        states.push_back(0);
        
        // Compute mixture of gaussians for every region
        std::cout<<"INFO: Computing _GMM for regions: \n";
        for(int i=0; i<regionMovingPatterns.cols; i++)
        {            
            std::cout<<"\t\tRegion: "<<i<<std::endl;
            // Region i
            cv::Mat xi;
            cv::hconcat( regionMovingPatterns.col(i), regionStaticPatterns.col(i), xi );
                   
            // Perform model selection using bic computation
            cv::EM gmm;
            cv::Mat labels, BIC, ll;
            
            // Compute BayesianInformationCriteria for model selection
            computeBICSelection(xi, bicLowRange, bicUpperRange, BIC, gmm, ll, labels);
            
//            std::cout<<BIC<<std::endl;
            
            // Preparing data for saving
            cv::hconcat(allBIC, BIC, allBIC);
            cv::hconcat(loglike, ll, loglike);
            cv::hconcat(allLabels, labels, allLabels);
            
            // Output the number of selected clusters
            std::cout<<"\t\tNumber of bic clusters: "<<gmm.getMat("means").rows<<std::endl;
            states.push_back( gmm.getMat("means").rows );
            
            // Saving model
            regionMOG.push_back( std::pair<cv::EM, cv::Mat>( gmm, labels ) );            
        }
        
        std::cout<<"INFO: Done!"<<std::endl;
        // Save BIC and likelihood information to a file
        tinf::saveMatrix(allBIC, (path2Save + "/BIC.txt").c_str(), true);
        tinf::saveMatrix(loglike, (path2Save + "/logLikelihood.txt").c_str(), true);
        
        cv::vconcat(cv::Mat(states).t(), allLabels, allLabels);
        allLabels.convertTo(allLabels, CV_64FC1);
        
        allLabels( cv::Range::all(), cv::Range(1, allLabels.cols)).copyTo(confLabels);
        
        tinf::saveMatrix( confLabels, (path2Save + "/gmmLabels.txt").c_str(), true);
        
        file.close();
        
        confLabels.convertTo( confLabels, CV_32SC1 );
        
        return regionMOG;        
    }
    
    
    
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
    ( const cv::Mat& data, int bicLowRange, int bicUpperRange, cv::Mat& BIC,
       cv::EM& selected_model, cv::Mat& outLogLikelihood, cv::Mat& labels, int covModel )
    {                
        // Number of examples and dimension
        int n = data.rows;  // Number of examples
        int d = data.cols;  // Dimension of examples
        
        // _GMM container, bic and labels for each model
        std::vector<cv::EM> gmm_models;
        std::vector<cv::Mat> gmm_labels;
        cv::Mat bic = cv::Mat::zeros(bicUpperRange - bicLowRange+1, 1, CV_64FC1);
        
        // final likelihood for each model
        outLogLikelihood = cv::Mat::zeros(bicUpperRange - bicLowRange+1, 1, CV_64FC1);
        
        int i=0;
        for(int k=bicLowRange; k<=bicUpperRange; k++, i++)
        {            
            // 
            cv::Mat log_likelihood, labels;
            
            // COV_MAT_SPHERICAL corresponds to VII model: Unequal volume identical spherical shape and aligned
            // COV_MAT_DIAGONAL corresponds to  VVI model: Volume and shape diferent but aligned
            // COV_MAT_GENERIC corresponds to EEE model: The most generic used model
            cv::EM gmm = cv::EM(k, covModel);
            
            // Train the mixture model recover the log likelihood for each
            // training example and the labels
            gmm.train(data, log_likelihood, labels);
            
            // k-1 mixture components
            // k*d variances (VVI model) or k variaces (VII model) or d*(d+1)/2 (EEE model)
            // k*d means parameters
            int beta;
            if(covModel == cv::EM::COV_MAT_GENERIC) beta = k*d*(d+1)/2;
            if(covModel == cv::EM::COV_MAT_DIAGONAL) beta = k*d;
            if(covModel == cv::EM::COV_MAT_SPHERICAL) beta = k;
            
            int kk = k-1 + k*d + beta;
            double ll = cv::sum(log_likelihood).val[0];
            
            // Computing BIC
            double bicValue = 2.0*ll - double(kk)*log(n);
            
            std::cout<<"\t\tComponents: "<<k
                     <<" Free parameters: "<<kk
                     <<" logLikelihood: "<<ll
                     <<" bic: "<<bicValue
                     <<std::endl;
            
            // Save bic function
            bic.at<double>(i,0) = bicValue ;
            // Save log-likelihood
            outLogLikelihood.at<double>(i,0) =  ll;
            // Save models
            gmm_models.push_back( gmm );
            gmm_labels.push_back( labels.clone() );
        }
        
        BIC = bic.clone();
        cv::Point p;
        
        cv::minMaxLoc(BIC, 0, 0, 0, &p);
        
        std::cout<<"\t\tINFO:Maximum bic: "<<BIC.at<double>(p.y,0)<<std::endl;
        std::cout<<"\t\tINFO:Selected clusters: "<<p.y+bicLowRange<<std::endl;
        
        selected_model = gmm_models[p.y];
        labels = gmm_labels[ p.y ].clone();
    }
    
    
    /**
     * @brief Finds within cluster variance given a set of clusters and a dataset.
     * @param data Clustered data.
     * @param centers Set of centers of the clusters.
     * @param labels Label for each data point.
     * @return Within cluster variance.
     */
    double withinClusterVariance
    ( const cv::Mat& data, 
      const cv::Mat& centers, 
      const std::vector<int>& labels )
    {   
        // Covariance matrices for each cluster
        std::vector<cv::Mat> covMat;
        cv::Mat z = cv::Mat::zeros(centers.cols, centers.cols, CV_64FC1);

        // Initialize each covariance matrix
        for(int i=0; i< centers.rows; i++)
             covMat.push_back(z.clone());

        // Calculate covariance matrices for clusters
        for(int i=0; i<data.rows; i++)
        {
            cv::Mat dif = data.row(i) - centers.row(labels[i]);
            dif.convertTo(dif, CV_64FC1);
            covMat[labels[i]] =  covMat[labels[i]] + dif.t()*dif;       
        }

        cv::Mat W = cv::Mat::zeros(centers.cols, centers.cols, CV_64FC1);

        // Sum of covariances
        for(int i=0; i<covMat.size(); i++)
            W =  W + covMat[i];

        // Return within cluster variance
        return cv::determinant(W);
    }       
    
    
    /**
     * @brief Computes mean and standar deviation for a univariate data
     * @param data Data.
     * @param mean Resultant mean.
     * @param std Resultant standard deviation
     */    
    void computeMeanStd(const cv::Mat& data, double& mean, double& std )
    {
        double meanSqr = 0;
        int N = data.rows;
        
        mean = 0;
        std = 0;

        for(int i=0; i<data.rows; i++)
            {
                double val = data.at<double>(i,0);
                
                meanSqr += val*val;
                mean += val;
            }
                
        mean = mean/double(N);
        meanSqr = meanSqr/double(N);
        
        std = sqrt( meanSqr - mean*mean );
    }
    
    /**
     * @brief Computes normalized cross correlation for two images.
     * @param patch Image patch.
     * @param templ Template to match.
     * @return Normalized cross correlation.
     */    
    double normalizeCrossCorrelation(const cv::Mat& patch, const cv::Mat& templ)
    {
        double ncc;
        
        double meanPatch = cv::mean(patch).val[0];
        double meanTempl = cv::mean(templ).val[0];
        
        cv::Mat patchCentered = patch - meanPatch;
        cv::Mat templCentered = templ - meanTempl;
        
        cv::Mat sqrPatch, sqrTempl, patchTemplProd;
        
        cv::multiply( patchCentered, templCentered, patchTemplProd );
        cv::multiply( patchCentered, patchCentered, sqrPatch );
        cv::multiply( templCentered, templCentered, sqrTempl );
        
        double sumProd = cv::sum(patchTemplProd).val[0];
        double sumSqrPatch = cv::sum(sqrPatch).val[0];
        double sumSqrTempl = cv::sum(sqrTempl).val[0];
        
        ncc = sumProd / sqrt( sumSqrPatch * sumSqrTempl );
        
        
        return ncc;
    }
    
    /**
     * @brief computes normalized correlation of two images.
     * @param m1 First image.
     * @param m2 Second image.
     * @return Normalized correlation
     */    
    double computeNormalizedCorrelation(const cv::Mat& m1, const cv::Mat& m2)
    {
        int N = m1.cols*m1.rows;
        double ccr=0.0;
        
        double meanSqr1=0.0, meanSqr2=0.0, mean1=0.0, mean2=0.0, std1, std2;
        
        for(int i=0; i<m1.rows; i++)
            for(int j=0; j<m1.cols; j++)
            {
                double val1 = m1.at<double>(i,j);
                double val2 = m2.at<double>(i,j);
                
                meanSqr1 += val1*val1;
                meanSqr2 += val2*val2;
                
                mean1 += val1;
                mean2 += val2;
            }
        
        meanSqr1 /= double(N);
        meanSqr2 /= double(N);
        mean1 /= double(N);
        mean2 /= double(N);
        
        std1 = sqrt( meanSqr1 - mean1*mean1 );
        std2 = sqrt( meanSqr2 - mean2*mean2 );
        
        for(int i=0; i<m1.rows; i++)
            for(int j=0; j<m1.cols; j++)
                ccr += ( (m1.at<double>(i,j)-mean1)*(m2.at<double>(i,j)-mean2) )/(std1*std2);
            
        
        return ccr/double(N);
    }
    
    /**
     * @brief Computes the Kullback-Leiber divergence for two distributions.
     * @param p First distribution.
     * @param q Second distribution.
     * @return Kullback divergence.
     */    
    double kullbackLeiblerDivergence
    ( const cv::Mat& p, const cv::Mat& q )
    {
        double kld = 0.0;
        
        for(int i=0; i<p.rows; i++)
        {
            double ratio = p.at<double>(i,0)/q.at<double>(i,0);
            
            kld += p.at<double>(i,0)*log(ratio);            
        }
        
        return kld;
    }
    
    /**
     * @brief Computes the histogram of an image.
     * @param img Image.
     * @return Histogram of image.
     */    
    cv::Mat imgHistogram(const cv::Mat& img)
    {
        cv::Mat hist = cv::Mat::zeros(256, 1, CV_64FC1);
        
        for(int i=0; i<img.rows; i++)
            for(int j=0; j<img.cols; j++)
            {
                int bin = (int)img.at<uchar>(i,j);
                hist.at<double>(bin,0)++;
            }
        
        hist = hist/double(img.rows*img.cols);
        
        return hist;
    }
    
    
    /**
     * @brief Generates a sample of size n from a Population of size N by index selection.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Vector containing the index of the selected samples.
     */    
    std::vector<int> samplingWithoutReplacement(int populationSize, int sampleSize)
    {
        // Generator of random numbers
        std::default_random_engine generator
        ( std::chrono::system_clock::now().time_since_epoch().count() );

        std::uniform_real_distribution<double> U(0.0,1.0);

        std::vector<int> index(sampleSize);

        int& N = populationSize;
        int& n = sampleSize;
        int t = 0;
        int m = 0;
        double u;

        for(;m<sampleSize;)
        {
            u = U( generator );

            if( (N-t)*u >= n-m ) t++;
            else{
                index[m] = t;
                t++;
                m++;
            }

        }

        return index;
    }

    /**
     * @brief Generates a sample of size n from a Population of size N by indicator
     * variables indicating if a data point is selected.
     * @param N Size of population.
     * @param n Size of sample.
     * @return Indicator vector with 0 at the selected entries.
     */
    std::vector<int> samplingNoReplace(int N, int n)
    {
        // Generator of random numbers
        std::default_random_engine generator
        ( std::chrono::system_clock::now().time_since_epoch().count() );
        // Distribution to generate probabilities
        std::uniform_real_distribution<double> unif(0,1);
        // Indicator vector
        std::vector<int> indicator(N, 1);
        
        // Loop to select points in the sample
        for(int i=0; i<n; i++)
        {
            // Generate probability and weight selection
            double tau = unif(generator);
            double p = 1.0/(double(N-i));

            // Cumulative weight
            double cummP = 0.0;
            // Move to select data from population
            for(int j=0; j<N; j++)
            {
                cummP += double(indicator[j])*p;
                // Select data if the cummulative is greater
                if(cummP > tau) {
                    indicator[j]=0;
                    break;
                }
            }
        }
        
        return indicator;
    } 

    
    /**
     * @brief Get a sample of data from a set of data.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Matrix of samples.
     */    
    cv::Mat getSample(const cv::Mat& population, int sampleSize)
    {
        std::vector<int> sampleIndex = samplingWithoutReplacement( population.rows, sampleSize );
                
        cv::Mat sample = cv::Mat::zeros(sampleSize, population.cols, CV_64FC1);
        
        for(int i=0; i<sampleSize; i++)
        {
            cv::Mat r = population.row( sampleIndex[i] ).clone();
            
            r.copyTo( sample.row(i) );
        }
        
        return sample;
    }

    
    
    void elongatedKMeans(const cv::Mat& X, int K, std::vector<int>& labels, cv::Mat&centers, std::vector<int>& counter, double epsilon, double lambda, bool lastZero)
    {
        int N = X.rows;
        int dim = X.cols;
        double d, max, min, t;
        
        labels = std::vector<int>(N);        
        
        cv::Mat D = cv::Mat::zeros(N, K, CV_64FC1);
        cv::Mat Iq = cv::Mat::eye(dim, dim, CV_64FC1);
        cv::Mat newCenters = cv::Mat::zeros(K, dim, CV_64FC1);
        cv::Mat prevCenters;
        
        if(lastZero)
        {
            centers = cv::Mat::zeros( K, dim, CV_64FC1 );

            std::default_random_engine generator
            ( std::chrono::system_clock::now().time_since_epoch().count() );

            /*
             * Intialize centers
             */  
            std::cout<<"Initializing Centers"<<std::endl;
            for(int c=0; c<X.cols; c++)
            {
                cv::minMaxLoc(X.col(c), &min, &max, 0, 0);

                std::uniform_real_distribution<double> dist(min, max);

                for(int k = 0; k<K-1; k++)            
                    for(int dd = 0; dd<dim; dd++)
                        centers.at<double>(k, dd) = dist( generator );                                       
            }
        } else {
            
            cv::Mat aux = cv::Mat::zeros(1, dim, CV_64FC1);
            
            cv::vconcat(centers, aux, centers);            
        }
        
        /*
         * 
         */    
        int it = 0;
        std::cout<<"Do clusttering"<<std::endl;
        do{
            newCenters = cv::Scalar::all(0.0);
            prevCenters = centers.clone();
            
            counter = std::vector<int>(K, 0);
            
            
            
            for(int n = 0; n<N; n++)
            {                                
                cv::Mat x = X.row(n);
                min = 1E100;
                
                for(int k=0; k<K; k++)
                {
                    cv::Mat c = centers.row(k);
                    double cdot = c.dot(c);
                    
                    if( cdot > epsilon )
                    {
                        cv::Mat C = c.t()*c;
                        
                        cv::Mat M1 = Iq - C/cdot;
                        cv::Mat M2 = lambda*C/cdot;
                        
                        cv::Mat M = (1.0/lambda)*M1 + M2; 
                       
                        cv::Mat xc = x - c;
                        
                        cv::Mat edist = xc*M*xc.t();
                        d = edist.at<double>(0,0);                                                
                        
                    } else {
                        cv::Mat xc = x-c;
                        d = sqrt( xc.dot(xc) );
                    }
                    
                    if(d < min)
                    {
                        labels[n] = k;
                        min = d;
                    }                  
                    
                    cv::Mat otherc = newCenters.row(k);
                    otherc = otherc + x;
                    otherc.copyTo( newCenters.row(k) );
                }
                
                counter[labels[n]]++;
            }
            
            for(int k=0; k<K; k++)
            {
                cv::Mat otherc = newCenters.row(k);
                int denom = (counter[k]>0)?counter[k]:1;
                
                otherc = otherc/double( denom );                
                otherc.copyTo(centers.row(k));
            }
            
            prevCenters = centers - prevCenters;
            cv::multiply(prevCenters, prevCenters, prevCenters);
            cv::reduce(prevCenters, prevCenters, 1, CV_REDUCE_SUM);
            cv::sqrt(prevCenters, prevCenters);
            t = cv::sum( prevCenters ).val[0];    
            
            std::cout<<t<<std::endl;
            
            std::cout<<cv::Mat(counter)<<std::endl;
        }while(t > 1e-5 && ++it<100);
        
        
    }
    

    /**
     * @brief Compute the Mutual information of two random variables with fixed number
     * of bins.
     * @param x Signal.
     * @param y Signal.
     * @param Kx Number of bins for signal x.
     * @param Ky Number of bins for signal y.
     * @return Mutual information at bins Kx and Ky for signal x and y
     */
    double MI(const cv::Mat& x, const cv::Mat& y, unsigned int Kx, unsigned int Ky) {
        double N = double(x.rows);
        double mi, xMax, xMin, yMax, yMin;
        
        double EPS = 1e-8;

        cv::Mat binX = x.clone();
        cv::Mat binY = y.clone();

        cv::minMaxLoc(binX, &xMin, &xMax, 0, 0);
        cv::minMaxLoc(binY, &yMin, &yMax, 0, 0);

        binX = binX - xMin;
        binY = binY - yMin;

        cv::minMaxLoc(binX, &xMin, &xMax, 0, 0);
        cv::minMaxLoc(binY, &yMin, &yMax, 0, 0);

        xMax = (xMax == 0.0)? 1 : xMax;
        yMax = (yMax == 0.0)? 1 : yMax;

        binX = binX*(1.0-EPS)/xMax;
        binY = binY*(1.0-EPS)/yMax;

         /// Get discrete signal
         for(unsigned int i=0; i<x.rows; i++) {
             binX.at< double >(i, 0) = floor( binX.at< double >(i, 0)*Kx );
             binY.at< double >(i, 0) = floor( binY.at< double >(i, 0)*Ky );
         }  

        cv::Mat Pxy = cv::Mat::zeros( Kx, Ky, CV_64FC1 ); 
        cv::Mat Px = cv::Mat::zeros( Kx, 1, CV_64FC1 );
        cv::Mat Py = cv::Mat::zeros( Ky, 1, CV_64FC1 );

        cv::Mat logPxy = Pxy.clone(), logPx = Px.clone(), logPy = Py.clone();

        /// Build distributions
        for(unsigned int i=0; i< x.rows; i++){	
            Pxy.at< double >( int(binX.at< double >(i, 0) ), int(binY.at< double >(i, 0) ) )++;
            Px.at< double >( int(binX.at< double >(i,0)), 0 )++;
            Py.at< double >( int(binY.at< double >(i,0)), 0 )++;
        }

        for(unsigned int i=0; i< Kx; i++)
           for( unsigned int j=0; j< Ky; j++) {
              Pxy.at< double >(i,j) = Pxy.at< double >(i,j) / N + EPS;
              logPxy.at< double >(i,j) = log2( Pxy.at< double >(i,j) );
            }


        for(unsigned int i=0; i< Kx; i++) {
           Px.at< double >(i,0) = Px.at< double >(i,0) / N + EPS;
           logPx.at< double >(i,0) = log2( Px.at< double >(i,0) );
        }

        for( unsigned int j=0; j< Ky; j++){
           Py.at< double >(j,0) = Py.at< double >(j,0) / N + EPS;
           logPy.at< double >(j,0) = log2( Py.at< double >(j,0) );
        }


        double Hx = 0.0;
        double Hy = 0.0;
        double Hxy = 0.0;

        for(unsigned int i=0; i< Kx; i++)
            Hx += Px.at< double >(i,0)*logPx.at< double >(i,0);

        for(unsigned int i=0; i< Ky; i++)
            Hy += Py.at< double >(i,0)*logPy.at< double >(i,0);

        for(unsigned int i=0; i< Kx; i++)
           for( unsigned int j=0; j< Ky; j++) 
              Hxy += Pxy.at< double >(i,j)*logPxy.at< double >(i,j);

        Hx = -Hx;
        Hy = -Hy;
        Hxy = -Hxy;

        mi = Hx + Hy - Hxy;

        return mi;
    }    
    
    
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
                           ,unsigned int n_chunks
                           ,double& tdmi, int& bestLag, bool doMic ){

        /// For the moment we are just considering a division of 4 chunks per signal
        unsigned int numberOfChunks = n_chunks;//8
        unsigned int sampleSize = x.rows;
        unsigned int chunkSize = sampleSize / numberOfChunks;	
        int initStep = numberOfChunks - 1; // 3 for 4 chunks
        int max_lag, opt_bx, opt_by;
        double max_mi = std::numeric_limits< double >::min();
        double mi, max_mic;

        int bx = _bx[0];
        int by = _by[0];

        int bx_max = _bx[1];
        int bx_min = _bx[0];
        int by_max = _by[1];
        int by_min = _by[0];  
        
        std::vector< double > signal;
        
        bool flag = true;
        int currentLag;
        
        for( int step = initStep, chunk=1; step >= 0; step--, chunk++) {
            /// Get the chunk that will be shifted
            cv::Mat chunkY =  y.rowRange( chunkSize*step, chunkSize*(step+1) ); 

            currentLag = -step*chunkSize;
                        
            //std::cout << "First lag: " << currentLag << std::endl;

            for( unsigned int i=0; i < sampleSize - chunkSize; i++ ) {
                cv::Mat chunkX = x.rowRange(i, i + chunkSize );

                /// If doMic=true, the mutual information coefficient must be computed
                /// over the range of bins that have been provided.
                if( doMic ) {
                    max_mic = -std::numeric_limits< double >::max();
                    /// Compute the mutual information coefficient.
                    for( bx = bx_min; bx <= bx_max; bx++) {
                        for(by = by_min; by <= by_max; by++){
                            if( double(bx*by) < pow(sampleSize , 0.6) ) {
                                /// Compute mutual information coefficient and find the maximum
                                /// In the interval of bins to check out.
                                 mi = tinf::MI(chunkX, chunkY, bx, by);	    
                                 mi = mi / log2( std::min< double >( bx, by ) );

                                 if(max_mic < mi) {
                                    max_mic = mi;
                                    opt_bx = bx;
                                    opt_by = by;
                                 }	          	     
                            }			
                        }
                    }
                    mi = max_mic;
                } else 
                    /// Compute only the mutual information of the two signals
                    mi =  tinf::MI(chunkX, chunkY, bx, by);

                signal.push_back( mi );
                
                if( mi > max_mi){
                    max_mi = mi;
                    max_lag = currentLag;
                }

                currentLag++;  			
            }

            cv::Mat sing = cv::Mat(signal);
            sing.convertTo(sing, CV_64FC1);
//            std::stringstream ss;
//            ss << chunk;
////            std::string str = "/home/angel/Documentos/8chunks_" + ss.str() + ".txt";
////            saveMatrix(sing , str.c_str() );
//            //std::cout << "Last lag: " << currentLag << std::endl;
            signal.clear();

        }

        /// output maximum mi and the lag that maximizes it.
        tdmi = max_mi;
        bestLag = max_lag;
    }    
    

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
                         double& tdmi, int& bestLag, int T ){

        int bx, by;                     
        int bx_max = _bx[1];
        int bx_min = bx = _bx[0];
        int by_max = _by[1];
        int by_min = by = _by[0];                     

        int lag = T;
        int shift, shift2, optLag, opt_bx, opt_by;

        unsigned int sampleSize = x.rows;

        double maxMI = std::numeric_limits< double >::min(), max_mic ,mi;  
        double doMICNorm = false;
        
        std::vector< double > vecMi;

        /// Compute MIC for the time window of lags
        for(int i = -lag; i<=lag; i++){
            cv::Mat laggedSignal = cv::Mat::zeros(sampleSize, 1, CV_64FC1);
            int absLag = abs(i);

            shift = (i<0)? 0: absLag;
            shift2 = (i<0)? absLag: 0;

            for(int j = 0; j< int(sampleSize)-absLag ; j++)
                laggedSignal.at< double >( shift + j, 0 )= y.at< double >(shift2 + j, 0);					

            /// Compute mutual information coefficient
            if( doMICNorm ) {
                max_mic = std::numeric_limits< double >::min();
                
                /// compute the mutual information coefficient and get the maximum
                /// within a range of bins for both variables.
                for( bx = bx_min; bx <= bx_max; bx++) {
                    for(by = by_min; by <= by_max; by++){
                        if( double(bx*by) < pow(sampleSize , 0.6) ) {
                             mi = MI(x, laggedSignal, bx, by);	    
                             mi = mi / log2( std::min< double >( bx, by ) );

                             if(max_mic < mi) {
                                max_mic = mi;
                                opt_bx = bx;
                                opt_by = by;
                             }	          	     
                        }			
                    }
                }
            } else
                max_mic = MI(x, laggedSignal, bx, by);

            mi = max_mic;
            
            vecMi.push_back(mi);

            if( maxMI < mi ) { 
                maxMI = mi;
                optLag = i;
            }

        }
        
        tinf::saveMatrix(cv::Mat(vecMi), "/home/angel/Documentos/normalMI.txt");

        tdmi = maxMI;
        bestLag = optLag;
    }

    
/********************************************************************************
 *                                                                              *
 *                                                                              *
 *                              CIRCULAR DATA METHODS                           *
 *                                                                              *
 *                                                                              *
 ******************************************************************************/


    /**
     * @brief Computes the circular correlation coefficient for circular data.
     * @param X First circular data.
     * @param Y Second circular data.
     * @return Circular correlation coefficient.
     */    
    double circularCorrelationCoefficient(const cv::Mat& A, const cv::Mat& B)
    {
        double r;        
        int N = A.rows;
        
        double sumABProd = 0.0;
        double sumSinASqr = 0.0;
        double sumSinBSqr = 0.0;
        
        vonMises vmA(A);
        vonMises vmB(B);
        
        
        for(int n=0; n< N; n++)
        {
            double thetaA = atan2(vmA.getCenter().at< double >(0,1), vmA.getCenter().at< double >(0,0));
            double thetaB = atan2(vmB.getCenter().at< double >(0,1), vmB.getCenter().at< double >(0,0));
            
            double sinA = sin(A.at<double>(n, 0) - thetaA);
            double sinB = sin(B.at<double>(n, 0) - thetaB);
            
            sumSinASqr += sinA*sinA;
            sumSinBSqr += sinB*sinB;
            sumABProd  += sinA*sinB;
        }
        
        r = sumABProd / sqrt(sumSinASqr*sumSinBSqr);
        
        return r;
    }
    
        /**
     * @brief Computes the circular correlation coefficient for circular data.
     * @param X First circular data.
     * @param Y Second circular data.
     * @return Circular correlation coefficient.
     */
    double circularCorrelationCoefficient(const cv::Mat& A, const cv::Mat& B, const cv::Mat& f1, const cv::Mat& f2, double tau)
    {
        
        double r;
        
        int N = A.rows;
        
        double sumABProd = 0.0;
        double sumSinASqr = 0.0;
        double sumSinBSqr = 0.0;

        cv::Mat _A = cv::Mat::zeros(A.rows, 2, CV_64FC1);
        cv::Mat _B = cv::Mat::zeros(B.rows, 2, CV_64FC1);

        for(int j=0; j< _A.rows; j++){
            _A.at<double>(j,0) = cos(A.at<double>(j,0));
            _A.at<double>(j,1) = sin(A.at<double>(j,0));
            
            _B.at<double>(j,0) = cos(B.at<double>(j,0));
            _B.at<double>(j,1) = sin(B.at<double>(j,0));
        }

        vonMises vmA(_A, f1, tau);
        vonMises vmB(_B, f2, tau);
        
        for(int n=0; n< N; n++)
        {
            if(f1.at<double>(n,0)<tau || f2.at<double>(n,0)<tau) continue;
            
            double thetaA = atan2(vmA.getCenter().at< double >(0,1), vmA.getCenter().at< double >(0,0));
            double thetaB = atan2(vmB.getCenter().at< double >(0,1), vmB.getCenter().at< double >(0,0));            
            
            double sinA = sin(A.at<double>(n, 0) - thetaA);
            double sinB = sin(B.at<double>(n, 0) - thetaB);
            
            sumSinASqr += sinA*sinA;
            sumSinBSqr += sinB*sinB;
            sumABProd  += sinA*sinB;
        }
        
        r = sumABProd/sqrt(sumSinASqr*sumSinBSqr);
        
        return r;        
    }
    
    
    
    /**
     * @brief Computes the circular median for circular data.
     * @param mat Matrix of circular data.
     * @return Circular median.
     */    
    double circularMedian(const cv::Mat& mat)
    {
        double median = 1e100;
        int m;
        
        for(int i=0; i<mat.rows; i++)
        {
            if(mat.at<double>(i,0)<0) continue;
            
            cv::Mat fm = cv::abs( mat - mat.at<double>(i,0) );
            double f = cv::sum(fm).val[0];
            
            if(median > f){
                median = f; 
                m=i;
            }
        }
        
        return mat.at<double>(m,0);
    }
        
   /**
    * @brief Computes K-Means for circular data.
    * @param X Circular dataset.
    * @param K Number of clusters to cluster into.
    * @param labels Best labels found.
    * @param centers Best centers found.
    */
    void sphericalKmeans( const cv::Mat& X, int K, 
                          std::vector<int>& labels,
                          cv::Mat& centers )
    {
        // Init variables
        int maxSteps = 100;
        int N        = X.rows;   
        
        double  _pi  = 3.1416;

        labels  = std::vector<int>( N );
        centers = cv::Mat::zeros(K, 1, CV_64FC1);
        
        
        // Init centers by randomly take points        
        std::vector<int> sample;        
        sample = tinf::samplingWithoutReplacement(N, K);
        
    
        for(int k = 0; k < K; k++)
            centers.at<double>( k, 0 ) = X.at<double>( sample[k], 0 );
        
        // Main Loop for clustering
        for(int t = 0; t < maxSteps; t++)
        {
            std::vector<int> counter(K, 0);
            
            // Expectation step
            for(int n = 0; n < N; n++)
            {
                cv::Mat kx = cv::Mat(1, K, CV_64FC1);
                cv::Point maxPosition;
                
                double theta = X.at<double>(n, 0);
                
                // Compute cosine similarity between point and center
                for(int k = 0; k < K; k++)
                {
                    double thetaHat = centers.at<double>(k, 0); 
                    kx.at<double>(0, k) = cos( theta - thetaHat );
                }
                
                cv::minMaxLoc(kx, 0, 0, 0, &maxPosition);
                
                labels[n] = maxPosition.x;
                counter[ maxPosition.x ]++;
                
//                std::cout<< kx << "  " << n << std::endl;
                
            }
            
            // Maximization step
            
            std::vector<double> sumSin(K, 0.0);
            std::vector<double> sumCos(K, 0.0);
            
            for(int n = 0; n < N; n++)
            {
                sumSin[ labels[n] ] += sin( X.at<double>(n, 0) );
                sumCos[ labels[n] ] += cos( X.at<double>(n, 0) );
            }
            
            for(int k = 0; k < K; k++)
            {
                centers.at<double>( k, 0 ) = atan2( sumSin[k], sumCos[k] );
                
                if(centers.at<double>( k, 0 ) < 0)
                    centers.at<double>( k, 0 ) += 2.0*_pi;
            }

        }               

    }       
    
}

