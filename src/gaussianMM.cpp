#include <sstream>

#include "mxTypeModel.h"

#include <opencv2/core/core.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::write_xml;

#ifdef __linux__
using boost::property_tree::xml_writer_settings;
#else
using boost::property_tree::xml_writer_make_settings;
#endif

namespace tinf
{
/********************************************************************************\
 *                          GAUSSIAN MIXTURE MODEL                              *
\********************************************************************************/
    
	/// Maximum number of iterations
	unsigned int GMM::maxIt = 100;
	/// Epsilon tolerance
	double GMM::eps = 1e-6;

	double GMM::LOG2PI = log(2*3.1416);
    
    /// Gaussian mixture model tag
    const std::string GMM::TAG = "GaussianMixtureModel";

   /**
    * @brief Default constructor
    */
   GMM::GMM(){

   }

   /**
    * @brief Copy constructor.
    * @param gmm Gaussian mixture to copy.
    */
   GMM::GMM(const GMM& gmm) {
       copy(gmm);
   }

   /**
    * @brief Constructor by computing the EM for mixture of Gaussian with the given training data.
    * @param X Sample of data sequences for training.
    * @param K Number of components in the mixture.
    * @param _covModel The covariance model of the clusters
    */
   GMM::GMM
   ( const cv::Mat& _X, int _K, int _covModel, bool _verb )
    {
         /// Cast away the constness
         cv::Mat * X = const_cast<cv::Mat*>(&_X);

	     K = _K;
	     verbose = _verb;
		 N = _X.rows;
		 D = _X.cols;

		 /// Construction of mixture and parameter stuff
		 mixture = std::vector< Gaussian >( K );

		 latentZ   = cv::Mat::zeros(N, K, CV_64FC1);
		 sampleLLH = cv::Mat::zeros(1, N, CV_64FC1);
		 pWeights   = cv::Mat::zeros(1, K, CV_64FC1);
		 Nk        = std::vector< int >(K, 0);

		 covsDet = cv::Mat::zeros(1, K, CV_64FC1);
		 covsInvEig = std::vector< cv::Mat >( K );
		 covsEig = std::vector< cv::Mat >( K );
         logWeights = cv::Mat::zeros(1, K, CV_64FC1);       

		 covModel = _covModel;
		 maxEigenValue = 0.0;
         
         MLTypeModel<Gaussian>::maxIt = GMM::maxIt;
         
         switch(covModel){
             case COV_MODEL_EII:
                 covModelStr = "EII-Model";
                 break;
             case COV_MODEL_VII:
                 covModelStr = "VII-Model";
                 break;
             default:///vvv model
                 covModelStr = "VVV-Model";
         }
         
         MLTAG = GMM::TAG;         
         currentState = "Training|" + covModelStr;     
                  
                  
        /// Do clustering data
        process(X);
    }

   
   GMM::GMM(const std::string& file){
       read(file);
   }
   
    /**
     * @brief Destructor
     */
    GMM::~GMM() {
//        std::cout << "-> Destructor of GMM" << std::endl;
    }


    /**
    * @brief Copy a new instance in this instance.
    * @param gmm Instance to copy
    */
   void GMM::copy(const GMM& gmm)
   {
       this->currentState = gmm.currentState;
       this->mixture      = gmm.mixture;
       this->Nk           = gmm.Nk;
       this->labels       = gmm.labels;
       this->pWeights     = gmm.pWeights.clone();
       this->sampleLLH    = gmm.sampleLLH.clone();
       this->latentZ      = gmm.latentZ.clone();
       
       this->QEnd  = gmm.QEnd;
       this->eps   = gmm.eps; 
       this->maxIt = gmm.maxIt;
        
       this->K     = gmm.K;
       this->D     = gmm.D;
       this->N     = gmm.N;
       
       this->verbose = gmm.verbose;
       
       covModelStr   = gmm.covModelStr;
       
       covsInvEig = std::vector< cv::Mat >(K);
       covsEig    = std::vector< cv::Mat >(K);
       
       for(int i=0; i<K; i++){
           this->covsInvEig[i] = gmm.covsInvEig[i].clone();
           this->covsEig[i]    = gmm.covsEig[i].clone();
       }
       
       this->covsDet    = gmm.covsDet.clone();
       this->logWeights = gmm.logWeights.clone();
       
       this->maxEigenValue = gmm.maxEigenValue;
       this->covModel      = gmm.covModel;
   }

    /**
     * @brief Assination operator of the mixture of Gaussians
     * @param vmem Instance of the Gaussian mixture to copy
     * @return A reference to this instance.
     */
    GMM& GMM::operator=(const GMM& gmm){
        copy(gmm);

        return *this;
    }


    /**
     * @brief Do Maximization step following maximum likelihood framework
     * @param em
     */
    void GMM::maximizationStep(const cv::Mat* _X) {
        cv::Mat totalLatentZ;
        cv::Mat m = cv::Mat::zeros(1, D, CV_64FC1);
        cv::Mat c = cv::Mat::zeros(D, D, CV_64FC1);

        /// Compute total probability of latent variables
        cv::reduce(latentZ, totalLatentZ, 0, CV_REDUCE_SUM);
        std::stringstream ss;
        //ss << totalLatentZ/N;
        //DbgLogger::iVerbose(TAG, currentState, "Total latent: " + ss.str());

        /// Update parameters of the mixture
        for(int k = 0; k < K; k++) {
            c = cv::Scalar::all(0.f);
            m = cv::Scalar::all(0.f);

            /// Compute new weights of clusters
            pWeights.at<double>(0, k) = totalLatentZ.at< double >(0, k) / double( N );

            /// Get the weighted vector mean and weighted covariance matrix
            for(int n = 0; n < N; n++){
                m = m + latentZ.at< double >(n, k)*(*_X).row(n);
                c = c + latentZ.at< double >(n, k)*(*_X).row(n).t()*(*_X).row(n);
            }

            cv::max(totalLatentZ, 1.0, totalLatentZ);
            
            /// Use properties of Cov(X) = (X-m)*(X-m)^t to solve all in one loop
            m = m / totalLatentZ.at<double>(0, k);
            c = c / totalLatentZ.at<double>(0, k) - m.t()*m;

            for(int i = 0; i< c.rows; i++)
                c.at< double >(i,i) = std::max< double >(TINF_MINIMUM_STD_DEV, c.at< double >(i,i));

            mixture[k].getMean() = m.clone();
            mixture[k].getCovariance() = c.clone();
            mixture[k].getInvCovariance() = c.inv();
        }

        /// Get the variances along the axis
        computeEigenCovs();
        /// Update determinants
        computeDets();

        /// Update logweights and avoid nan
        cv::log(pWeights + 1e-10, logWeights);

        cv::Mat yzn;
        /// Compute data likelihood
        for(int n= 0; n < N; n++)
            sampleLLH.at< double >(0, n) = computeLikelihood(_X->row(n), yzn)[0];        

    }

    /**
     * @brief Compute the likelihood of the given datapoint.
     * This procedure is done using the log-sum-exp
     */
    cv::Vec2d GMM::computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const{
    	//// init data
    	yzn.create( cv::Size( K, 1), CV_64FC1 );

    	int    label = 0;
    	double expFact = 0.0;

    	for(int k = 0; k < K; k++){
    		/// Compute likelihood
            cv::Mat xc = xn - mixture[k].getMean();
    		expFact = computeExpFact(&xc, k);
            /// 
    		yzn.at< double >(0, k) = logWeights.at< double >(0, k)
    								 - 0.5 * double(D) * LOG2PI
    		                         - 0.5 * covsDet.at< double >(0, k)
    		                         - 0.5 * expFact;
    		//yzn.at< double >(0, k) = pWeights.at< double >(0,k) * mixture[k].getLikelihood(xn);

    		/// Get the cluster with maximum likelihood
    		if(yzn.at< double >(0, label) <  yzn.at< double >(0, k) ) label = k;
    	}

    	/// Compute log-sum-exp
    	double zmax = yzn.at< double >(0, label);
    	yzn -= zmax;											// subtract max
    	cv::exp(yzn, yzn);										// evaluate exponential
    	double logSumExp = zmax + std::log( cv::sum(yzn)[0] ); 	// compute log of sum

    	/// Save data
    	cv::Vec2d p;
    	p[0] = logSumExp;
    	p[1] = label;

    	return p;
    }


    /**
     * @brief Computes the exponential factor for the EII covariance matrix model
     * @param xc Centered data point for the cluster k
     * @param k Cluster index
     */
    double GMM::computeExpFact(const cv::Mat* xc, int k)const{
    	cv::Mat outXc;
        /// Set the function to estimate the exponent factor given the covariance model
		switch( covModel ){
			 case COV_MODEL_EII: 
                 return xc->dot(*xc)*covsInvEig[k].at< double >(0, 0);
			 case COV_MODEL_VII: 
			     cv::multiply(*xc, covsInvEig[k], outXc);
			     return xc->dot(outXc);			 
			 case COV_MODEL_VVV: 
			     outXc = (*xc) * mixture[k].getCovariance().inv();
			     return outXc.dot(*xc);		    
		}

		 return 0.0;
    }

    /**
    * @brief Initialize the expectation maximization by using K-means.
    * @param X Pointer to the data to do the initialization
    */
    void GMM::init(const cv::Mat* X) {
        /// Parameters for the K-means
        cv::Mat Xp, yzn;
        cv::Mat centers;
        int attempts = 100;

        (*X).convertTo(Xp, CV_32FC1);

        /// Compute K-means to initialize
        cv::kmeans( Xp , K, labels,
                    cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
                    attempts, cv::KMEANS_RANDOM_CENTERS, centers );

        centers.convertTo(centers, CV_64FC1);

        /// Get the initial parameters
        estimateParametersFromLabels(X, labels, centers, mixture, pWeights);

        /// Get the variances along the axis
        computeEigenCovs();
        /// Update determinants
        computeDets();
        /// Update logweights
        cv::log(pWeights, logWeights);

        /// Initial likelihood
        for( int i = 0; i < N; i++)
        	sampleLLH.at< double >(0, i) = computeLikelihood(X->row(i), yzn)[0];        
        
        /*std::stringstream ss;
        ss << covsDet;
        DbgLogger::iVerbose(TAG, currentState, "Determinats: " + ss.str());            */
            
//        std::stringstream ss;
//        ss << centers;
//        DbgLogger::iVerbose(TAG, currentState, "GMM Initial-Centers" + ss.str());
//        ss.str("");
//        ss << pWeights;
//        DbgLogger::iVerbose(TAG, currentState, "GMM Initial-Weights" + ss.str());
    }

    /**
     *
     * @param X
     * @param labels
     * @param mixture
     */
    void GMM::estimateParametersFromLabels
    ( const cv::Mat* X, const std::vector<int>& labels, const cv::Mat& centers, std::vector<Gaussian>& mixture, cv::Mat& pi)
    {
        int N = X->rows;
        int D = X->cols;
        int K = mixture.size();

        std::vector< int > labCounter(K, 0);
        std::vector< cv::Mat> cov(K, cv::Mat::zeros( D, D, CV_64FC1 ));

        /// Count initial assignations
        for(int i=0; i<N; i++) {
            int k = labels[i];
            labCounter[ k ]++;

            cv::Mat xc = (*X).row(i) - centers.row(k);
            cov[k] = cov[k] + xc.t()*xc;
        }

        /// Normalize intial parameters
        for(int k=0; k<K; k++)
        {
            mixture[k].getMean() = centers.row(k).clone();

            /// Avoid zero division
            int div = (labCounter[k] == 0)?1:labCounter[k];
            mixture[k].getCovariance() = cov[k] / div;
            mixture[k].getInvCovariance() = mixture[k].getCovariance().inv();
            pi.at<double>(0,k) = double(labCounter[k])/double(N);

           // std::cout << mixture[k].getCovariance() << std::endl;

        }
    }


    /**
     * @brief Compute the eigen values of the covariance matrices in the mixture
     */
    void GMM::computeEigenCovs(){

    	cv::Mat eigenValues, eigenVectors;

         std::stringstream ss;
    	/// Compute eigen vectors and eiven values and avoid small variance
    	for(int k = 0; k < K; k++){

    		cv::eigen( mixture[k].getCovariance(), eigenValues, eigenVectors );
    		cv::max(eigenValues, Gaussian::GAUSS_MIN_VAR, covsEig[k]);
            
            //covsEig[k] = covsEig[k].t();
            
    		if(covsEig[k].at< double >(0,0) > maxEigenValue)
    			maxEigenValue = covsEig[k].at< double >(0,0);

    		covsInvEig[k] = 1.0/covsEig[k];
    	}      
    }

    /**
     * @brief Set the covariance matrices the chosen covariance model
     */
    void GMM::updateCovs(){

    }


    /**
     * @brief Compute the determinants of the covariances
     */
    void GMM::computeDets(){
        // Compute the determinants of the covariance matrix given the eigen values
    	for(int k = 0; k<K; k++){
		  /// Compute determinant as the product of eigen values
   		  covsDet.at< double >(0,k) = 1.0;
//          std::stringstream ss;
//          ss << covsEig[k].rows;
//          DbgLogger::iVerbose(TAG, currentState, "rows " + ss.str());
		  for(int d = 0; d< covsEig[k].rows; d++)
			  covsDet.at< double >(0,k) *= covsEig[k].at< double >(0, d);
    	}
    }

   /**
     *
     * @return
     */
    int GMM::getParamsNumber()
    {
        // k-1 mixture components
        // k*d variances (VVI model) or k variaces (VII model) or d*(d+1)/2 (EEE model)
        // k*d means parameters
        int beta = 0;
        switch(covModel){
            case COV_MODEL_VVV:
                beta = K*D*(D+1)/2;
                break;             
            case COV_MODEL_VII:            
                beta = K*D;
                break;            
            default: // COV_MODEL_EII
                beta = K;
        }
        
        int kk = K-1 + K*D + beta;
        
        return kk;
    }
    

  void GMM::write(const std::string& fileName){
      
        /// boost tree to save as a xml
        ptree tree;
        /// Version of joint type model
        tree.add("GaussianMixtureModel.<xmlattr>.version", "0.0.1");        
        
        /**
         * Save von Mises distributions.
         */
        cv::Mat _centers = mixture[0].firstMoment().clone();
        cv::Mat _covs    = mixture[0].secondMoment().clone();
        
        std::stringstream dimStr;
        std::stringstream KStr;
        std::stringstream mllhStr;

        dimStr      << D;
        KStr        << K;
        mllhStr     << QEnd;        
        
        /*
         * Concat means and covs into one matrix only
         */
        for(int i = 1; i < K; i++) {
            cv::vconcat( _centers, mixture[i].firstMoment(),    _centers );
            cv::vconcat( _covs,    mixture[i].secondMoment(), _covs );
        }
        
        /// Convert all matrices to text
        std::string centersStr = mat2String( _centers );
        std::string covsStr    = mat2String( _covs );
        std::string mixcoefStr = mat2String( pWeights );
        
        /// Save von mises mixture
        ptree& mog_pt = tree.add("GMM", "");
        mog_pt.add("K", KStr.str());
        mog_pt.add("dim", dimStr.str());
        mog_pt.add("cov_model", covModelStr);
        mog_pt.add("model_likelihood", mllhStr.str());
        mog_pt.add("mix_coef", mixcoefStr);
        mog_pt.add("centers", centersStr);
        mog_pt.add("covs", covsStr);        
        

        /// Write the xml file
#ifdef __linux__
        write_xml(fileName, tree, std::locale(), xml_writer_settings<char>(' ', 4));
#else
        write_xml(fileName, tree, std::locale(), xml_writer_make_settings<std::string> (' ', 4)); 
#endif
   }
  
  void GMM::read(const std::string& fileName){
         const double minEigenValue = 4.e-3;
        
        // Create an empty property tree object
        using boost::property_tree::ptree;
        ptree pt;

        // Load the XML file into the property tree. If reading fails
        // (cannot open file, parse error), an exception is thrown.
        read_xml(fileName, pt); 

        /*
         * Read data for the mixture of von mises.
         */
        K = pt.get< int >("GMM.K");   
        D = pt.get< int >("GMM.dim");
        QEnd = pt.get< double >("GMM.model_likelihood");
        std::stringstream ss;
        ss.str( pt.get< std::string >("GMM.cov_model") );
        ss >> covModelStr;
        /// TODO: send exception for covariance model type
        
        /// Do for covariance type
        if( covModelStr == "VVV-Model" )
            covModel = COV_MODEL_VVV;
        else if( covModelStr == "EII-Model" )
            covModel = COV_MODEL_EII;
        else if( covModelStr == "VII-Model" )
            covModel = COV_MODEL_VII;
        else
            throw 1;// Trow an exception
	
        
        /// Get data as string
        std::string mixCoefMOG = pt.get< std::string >("GMM.mix_coef");
        std::string centersMOG = pt.get< std::string >("GMM.centers");
        std::string covsMOG    = pt.get< std::string >("GMM.covs");
        
        std::stringstream _mix_coef, _centers, _covs;
                
        _mix_coef.str( mixCoefMOG );
        _centers.str(centersMOG );
        _covs.str( covsMOG );
                
        cv::Mat center = cv::Mat::zeros(1, D, CV_64FC1);     
        cv::Mat S      = cv::Mat::zeros(D, D, CV_64FC1);
        mixture        = std::vector< Gaussian >( K );
        pWeights       = cv::Mat::zeros( 1, K, CV_64FC1 );
        covsEig        = std::vector< cv::Mat >(K);
        covsInvEig     = std::vector< cv::Mat >(K);
        covsDet        = cv::Mat::zeros(1, K, CV_64FC1);
                   
            
        /// Convert strings into doubles and get the mixtures of gaussians
        for( int k = 0; k < K; k++) {
            
            _mix_coef >> pWeights.at< double >(0,k);

            /// Load center and variance of cluster
            for( int ii = 0; ii < D; ii++) {
                _centers >> center.at< double >(0, ii);

                for(int jj = 0; jj < D; jj++)
                    _covs >> S.at< double >(ii, jj);                    
            }
                
            /// Get the respective gaussian
            mixture[k] = Gaussian(center, S);                
        }
        
        /// Get the variances along the axis
        computeEigenCovs();
        /// Update determinants
        computeDets();
        /// Update logweights
        cv::log(pWeights, logWeights);  
  }
  
    
}

