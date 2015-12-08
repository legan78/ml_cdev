#ifndef _MX_TYPE_MODEL_H_
#define _MX_TYPE_MODEL_H_

#include "mlTypeModel.h"

namespace tinf
{
    
        
/*********************************************************************************************\
 *                             MIXTURE OF VON MISES DISTRIBUTION                             *
\*********************************************************************************************/

    /**
     * @brief Class to implement the expectation maximization of a set of directional
     * variables with a mixture of von mises.
     */
    class VMEM :  public MLTypeModel< vonMises >
    {
        friend class JTTypeModel;
        friend class MatMixModel;
        public:

            /**
             * @brief Default constructor.
             */
            VMEM();

            /**
             * @brief Copy constructor.
             * @param vmem Von mixture to copy
             */
            VMEM(const VMEM& vmem);

            /**
             * @brief Constructor by computing the EM for mixture of von Mises with the given training data.
             * @param X Sample of circular data sequences for training. Data must be in the unit sphere.
             * @param K Number of components in the mixture.
             * @param maxIt Maximum number of iterations allowed
             * @param eps Epsilon to stop the algorithm for likelihood stability
             */
            VMEM(const cv::Mat& _X, int _K, bool _verbose = false, int _maxIt = 100, double _eps = 1e-8);

            /**
             * @brief Assination operator of the mixture of von mises
             * @param vmem Instance of the von mises mixture to copy
             * @return A reference to this instance.
             */
            VMEM& operator=(const VMEM& vmem);

            /**
             * @brief Destructor.
             */
            ~VMEM();

            /**
             * @brief Implemetation of getting the total number of parameters
             * @return Number of parameters for the model.
             */
            int getParamsNumber();
            
        protected:

            /**
             * @brief Copy a new instance in this instance.
             * @param vmem Instance to copy
             */
            void copy(const VMEM& vmem);

            /**
             * @brief Initialize the expectation maximization by using the spherical
             * K-means.
             * @param X Pointer to the data to do the initialization
             */
            void init(const cv::Mat* X);

            /**
             * @brief Do maximization step.
             * @param X Data to do maximization step.
             */
            void maximizationStep(const cv::Mat* X);


            /**
             * @brief Compute the likelihood of the given datapoint.
             * This procedure is done using the log sum exp
             */
            cv::Vec2d computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const;


    };

/********************************************************************************\
 *                          GAUSSIAN MIXTURE MODEL                              *
\********************************************************************************/
    
    /**
     * @brief Class to implement the Expectation Maximization algorithm for univariate GMM 
     * using the MAP and MLE framework to avoid singularities in the variances.
     */
    class GMM : public MLTypeModel< Gaussian >
    {
        public:           
            friend class JTTypeModel;
            friend class MatMixModel;
    		enum{COV_MODEL_EII = 0, COV_MODEL_VII, COV_MODEL_VVV, COV_MODEL_DEFAULT = COV_MODEL_VVV};

            /**
             * @brief Default constructor
             */
            GMM();

            /**
             * @brief Copy constructor.
             * @param gmm Gaussian mixture to copy.
             */
            GMM(const GMM& gmm);            
            
           /**
            * @brief Constructor by computing the EM for mixture of Gaussian with the given training data.
            * @param X Sample of data sequences for training.
            * @param K Number of components in the mixture.
            * @param _covModel The covariance model used in the mixture
            */        
           GMM( const cv::Mat& X, int _K, int _covModel = COV_MODEL_DEFAULT, bool verb = false);
            
           /**
            * @brief Constructor to load GMM from xml file
            * @param inputFile Name or full path to the xml file of the model.
            */
           GMM(const std::string& inputFile);
           
           
            /**
             * @brief Assination operator of the mixture of Gaussians
             * @param vmem Instance of the Gaussian mixture to copy
             * @return A reference to this instance.
             */        
            GMM& operator=(const GMM& gmm);
                               
            
            /**
             * @brief Destructor
             */
            ~GMM();

            /**
             * @brief Estimate the parameters of a GMM using the labels of data.
             * @param X Data used to generate the labels.
             * @param labels Data labels.
             * @param centers Intput centers.
             * @param mixture Output set of gaussians.
             * @param Output mixture coefficients.
             */
            static void estimateParametersFromLabels
            ( const cv::Mat* X, const std::vector<int>& labels, const cv::Mat& centers,
              std::vector<Gaussian>& mixture, cv::Mat& pi);

            void write(const std::string& file);
            
            /**
             * @brief Implemetation of getting the total number of parameters
             * @return Number of parameters for the model.
             */
            int getParamsNumber();
            
            /**
             * @brief Maximum number of iterations
             */
            static unsigned int maxIt;

   		 	 /**
              * @brief Epsilon tolerance
              */
   		 	 static double eps;
             
             /**
              * @brief String tag of the model
              */ 
             static const std::string TAG;

        protected:

            /**
             * @brief Constant log(2*pi)
             */
   		 	 static double LOG2PI;

            /**
             * @brief Copy a new instance in this instance.
             * @param gmm Instance to copy
             */
            void copy(const GMM& gmm);

            /**
             * @brief Initialize the expectation maximization by using K-means.
             * @param X Pointer to the data to do the initialization
             */
            void init(const cv::Mat* X);

            /**
             * @brief Do maximization step.
             * @param X Data to do maximization step.
             */
            void maximizationStep(const cv::Mat* X);


            /**
             * @brief Compute the likelihood of the given datapoint.
             * This procedure is done using the log sum exp
             */
            cv::Vec2d computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const;

            /**
             * @brief Compute the eigen values of the covariance matrices in the mixture
             */
            void computeEigenCovs();

            /**
             * @brief Set the covariance matrices the chosen covariance model
             */
            void updateCovs();

            /**
             * @brief Compute the determinants of the covariances
             */
            void computeDets();

            /**
             * @brief Function pointer to the computation of the exponential factor given the covarance model
             * @param xc Centered data point for the cluster k
             * @param k Cluster index
             */
            double computeExpFact(const cv::Mat*, int)const;
            
            /**
             * @brief Loads model from xml file.
             * @param fileName Name or full path to the xml file.
             */
            void read(const std::string& fileName);
            
            /**
             * @brief Covariace model as string
             */
            std::string covModelStr;

            /**
             * @brief The volume of each distribution, these are just the eigen values of each matrices
             */
            std::vector< cv::Mat > covsInvEig;
            std::vector< cv::Mat > covsEig;

            /**
             * @brief Determinant of the covariances.
             */
            cv::Mat covsDet;

            /**
             * @brief The logarithm of the cluster weights
             */
            cv::Mat logWeights;

            /**
             * @brief The maximum eigen value is useful for some covariance models.
             */
            double maxEigenValue;

            /**
             * @brief Covariance model for the mixture components.
             */
            int covModel;
    };    
    
    
    /********************************************************************************\
     *                              SPHERICAL K-MEANS                               *
    \********************************************************************************/

    /**
     * @brief Class to implement the spherical K-Means. 
     */
    class SPHKMeans
    {
        public:

            /**
             * @brief Default constructor.
             */
            SPHKMeans();

            /**
             * @brief Copy Constructor
             * @param sph Instance to copy.
             */
            SPHKMeans(const SPHKMeans& sph);

            /**
             * @brief Constructor by performing the Spherical K-means.
             * @param X Training sample of unit vectors, each row a unit vector.
             * @param K Number of clusters
             */
            SPHKMeans(const cv::Mat& X, int K);

            /**
             * @brief Assignation operator.
             * @param sph Instance to copy.
             * @return Reference to this instance.
             */
            SPHKMeans& operator=(const SPHKMeans& sph);

            /**
             * @brief Get the labels of the training data.
             * @return Constant reference to the labels of the training data.
             */
            const std::vector< int >& getTrainingLabels()const;

            /**
             * @brief Get the centers of the clusters of the training data
             * @return Constant reference to the centers found.
             */
            const cv::Mat& getCenters()const;

        protected:


            /**
             * @brief Labels for the training data.
             */
            std::vector< int > labels;

            /**
             * @brief Centers of the clusters. Each row is a center.
             */
            cv::Mat centers;
    };

/*******************************************************************************************************\
 *                              GAUSSIAN MIXTURE MODEL BY USING THE OPENCV CLASS                       *
\*******************************************************************************************************/
    
    /**
     * @brief GAUSSIAN MIXTURE MODEL BY INHERIT FROM THE OPENCV CLASS.
     */
    class _GMM_ : public MLTypeModel< Gaussian >, protected cv::EM
    {
        friend class JTTypeModel;
        friend class MatMixModel;

    	public:
            
            /**
             * @brief Default constructor.
             */
            _GMM_();
            
            /**
             * @brief Copy constructor.
             * @param gmm Instance to copy.
             */
            _GMM_(const _GMM_& gmm);
            
            /**
             * @brief Constructor by modelling the data using a gaussian mixture model.
             * @param X Data to model.
             * @param K Number of components to use.
             * @param covType Type of covariance to use.
             */
            _GMM_(const cv::Mat& X, int K, bool verb = false, int _maxIt = 100, double _eps = 1e-8, int covType = defaultCovModel);
            
            /**
             * @brief Operator of assignation.
             * @param gmm Instance to copy.
             * @return This instance.
             */
            _GMM_& operator=(const _GMM_&gmm);
            
            /**
             * @brief Likelihood of the data in the given component
             * @param x Data to get the likelihood.
             * @param component Component to which get the likelihood.
             * @return Likelihood.
             */
            double predict(const cv::Mat& x, const int& component);
            
            /**
             * @brief Get cluster and likelihood of the new observed point p.
             * @param p Point to evaluate.
             * @return Container of with[0] best label and [1] likelihood.
             */
            cv::Vec2f predict(const cv::Mat& x, cv::Mat* outYz=NULL);
            
            
            /**
             * @brief Get the number of free params if the model
             * @return Number of parameters in the model
             */
            int getParamsNumber();
            
            /**
             * @brief Get the type of the covariance matrices.
             * @return Type of covariance matrix.
             */
            int getCovMatModel();
            
            
            /**
             * @brief Default covariance type model.
             */
            static int defaultCovModel;
            
        protected:
            
            /**
             * @brief Initialize the expectation maximization by using the spherical
             * K-means.
             * @param X Pointer to the data to do the initialization
             */
            void init(const cv::Mat* X);

            /**
             * @brief Compute the likelihood of the given datapoint.
             */
            cv::Vec2d computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const {
            	return cv::Vec2d();
            }

            /**
             * @brief Do maximization step.
             * @param X Data to do maximization step.
             */
            void maximizationStep(const cv::Mat* X);
            
            /**
             * @brief Covariance model.
             */
            int covModel;
            
            
    };
    
}


#endif /*_MX_TYPE_MODEL_H_*/
