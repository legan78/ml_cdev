#ifndef _DISTRIBUTIONS_H_
#define _DISTRIBUTIONS_H_

#include <opencv2/opencv.hpp>
#include <chrono>
#include <random>

// #include <Eigen/Eigenvalues>

/**
 * @brief This namespace has the implementation of methods and classes to perform the 
 * inference of the topology of a network of cameras.
 */
namespace tinf
{
    
    /**
     * @brief Interface to implement the concept of probability distribution function.
     * Therefore, any new distribution must inherit from this class and implement
     * the interface methods.
     */
    class distribution
    {
        public:
            
            /**
             * @brief Compute the likelihood of the distribution function for a given point.
             * @param x Point to evaluate the likelihood in the distribution.
             * @return The likelihood of the distribution for the given point.
             */
            virtual double getLikelihood(const cv::Mat& x)const = 0;
            
            /**
             * @brief Access to the center of the distribution.
             * @return Constant reference to the center of the distribution.
             */
            virtual const cv::Mat& firstMoment()const = 0;

            /**
             * @brief Access to the center of the distribution.
             * @return Reference to the center of the distribution.
             */        
            virtual cv::Mat& firstMoment() = 0;

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            virtual const cv::Mat& secondMoment()const = 0;


            /**
             * @brief Access to the concentration parameter.
             * @return Reference to the concentration parameter of distr.
             */
            virtual cv::Mat& secondMoment() = 0;    
    };

    
/********************************************************************************\
 *                         VON MISES DISTRIBUTION                               *
\********************************************************************************/
    /**
     * @brief Class to implement the von Mises probability distribution.
     */
    class vonMises : public distribution
    {
        public: 

            /**
             * @brief Default constructor
             */
            vonMises();

            /**
             * @brief Constructor by center and concentration parameter.
             * @param _theta0 Center of the distribution.
             * @param _kappa Concentration parameter of the distribution.
             */
            vonMises(const cv::Mat& _theta0, double _kappa);

            /**
             * @brief Constructor by learning the parameters from data using the
             * maximum likelihood approach of the von Mises distribution.
             * @param X A sample of circular data.
             */
            vonMises(const cv::Mat& X);
            
            /**
             * @brief Constructor by learning the parameters from data using the
             * maximum likelihood approach of the von Mises distribution.
             * @param X A sample of circular data.
             */
            vonMises(const cv::Mat& X, const cv::Mat& mask, double tau);

            /**
             * @brief Copy constructor.
             */
            vonMises(const vonMises& vm);

            /**
             * @brief Operator of assignation.
             * @param vm Instance to copy.
             * @return Reference to this.
             */
            vonMises& operator=(const vonMises& vm);

            /**
             * @brief Destructor.  
             */
            ~vonMises();

            /**
             * @brief Get the likelihood of the given point in the von Mises distribution.
             * @param x Data to evaluate the likelihood.
             * @return Likelihood of the given point.
             */
            double getLikelihood(const cv::Mat& x)const;

            /**
             * @brief Access to the center of the distribution.
             * @return Constant reference to the center of the distribution.
             */
            const cv::Mat& getCenter()const;

            /**
             * @brief Access to the center of the distribution.
             * @return Reference to the center of the distribution.
             */        
            cv::Mat& getCenter();

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            const double& getConcentration()const;


            /**
             * @brief Access to the concentration parameter.
             * @return Reference to the concentration parameter of distr.
             */
            double& getConcentration();        
            
            /**
             * @brief Access to the center of the distribution.
             * @return Constant reference to the center of the distribution.
             */
            const cv::Mat& firstMoment()const;

            /**
             * @brief Access to the center of the distribution.
             * @return Reference to the center of the distribution.
             */        
            cv::Mat& firstMoment();

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            const cv::Mat& secondMoment()const;


            /**
             * @brief Access to the concentration parameter.
             * @return Reference to the concentration parameter of distr.
             */
            cv::Mat& secondMoment();  
            

            /**
             * @brief Method to calculate the concentration parameter by fisher proposal
             * @param x Parameter to evaluate the fisher proposal
             * @return Value of fisher proposal for concentration parameter
             */    
            static double fisherAInv(double x);            

            /**
             * @brief Implements the besel function of the first kind in 0 degree.
             * @param x Data to evaluate the bessel function.
             * @return Evaluation of the bessel function in the point x.
             */
            static double besselI0(double x);

            /**
             * @brief Maximum kappa allowed for the distribution to avoid singularities
             */
            static double MAX_KAPPA;

        protected:

            /**
             * @brief Center of the distribution.
             */
            cv::Mat theta0;

            /**
             * @brief Concentration parameter of the distribution.
             */
            double kappa;                
            
            /**
             * @brief Concentration parameter of the distribution as a matrix object.
             */
            cv::Mat kappaMat;
    };

    /********************************************************************************\
     *                          GAUSSIAN DISTRIBUTION                               *
    \********************************************************************************/

    /**
     * @brief Class to implement the multivariate Gaussian distribution.
     */
    class Gaussian : public distribution
    {
        public: 

    		/**
             * @brief Covariance model for the gaussian distribution.
             */
    		enum{ COV_MODEL_SPHERICAL = 0, 
                  COV_MODEL_DIAGONAL, 
                  COV_MODEL_GENERAL, 
                  COV_MODEL_DEFAULT_= COV_MODEL_GENERAL};

            /**
             * @brief Default constructor
             */
            Gaussian();

            /**
             * @brief Constructor by mean and covariance matrix
             * @param _mu Mean of the distribution.
             * @param _S Covariance matrix.
             */
            Gaussian(const cv::Mat& _mu, const cv::Mat& _S);

            /**
             * @brief Constructor by learning the parameters from data using the
             * maximum likelihood approach of the Gaussian distribution.
             * @param X A sample of training data.
             */
            Gaussian(const cv::Mat& X);

            /**
             * @brief Copy constructor.
             */
            Gaussian(const Gaussian& vm);

            /**
             * @brief Operator of assignation.
             * @param vm Instance to copy.
             * @return Reference to this.
             */
            Gaussian& operator=(const Gaussian& vm);

            /**
             * Destructor.  
             */
            ~Gaussian();

            /**
             * @brief Get the likelihood of the given point in the Gaussian distribution.
             * @param x Data to evaluate the likelihood.
             * @return Likelihood of the given point.
             */
            double getLikelihood(const cv::Mat& x)const;

            /**
             * @brief Access to the center of the distribution.
             * @return Constant reference to the center of the distribution.
             */
            const cv::Mat& getMean()const;

            /**
             * @brief Access to the center of the distribution.
             * @return Reference to the center of the distribution.
             */        
            cv::Mat& getMean();

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            const cv::Mat& getCovariance()const;

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            const cv::Mat& getInvCovariance()const;

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            cv::Mat& getInvCovariance();


            /**
             * @brief Access to the concentration parameter.
             * @return Reference to the concentration parameter of distr.
             */
            cv::Mat& getCovariance();                                
            
            /**
             * @brief Access to the center of the distribution.
             * @return Constant reference to the center of the distribution.
             */
            const cv::Mat& firstMoment()const;

            /**
             * @brief Access to the center of the distribution.
             * @return Reference to the center of the distribution.
             */        
            cv::Mat& firstMoment();

            /**
             * @brief Access to the concentration parameter.
             * @return Constant reference to the concentration parameter of distr.
             */
            const cv::Mat& secondMoment()const;

            /**
             * @brief Access to the concentration parameter.
             * @return Reference to the concentration parameter of distr.
             */
            cv::Mat& secondMoment();  
                        

            /**
             * @brief Maximum kappa allowed for the distribution to avoid singularities.
             */
            static double GAUSS_MIN_VAR;

        protected:

            /**
             * @brief Mean of the distribution.
             */
            cv::Mat mu;

            /**
             * @brief Covariance of the distribution.
             */
            cv::Mat S; 

            /**
             * @brief Covariance inverse.
             */
            cv::Mat Sinv;
    };

    /********************************************************************************\
     *                            HYBRID DISTRIBUTION                               *
    \********************************************************************************/

    /**
     * @brief Class to implement the hybrid distribution function of gaussian and 
     * von mixture joint distribution.
     */    
	 class HybridMode : public distribution{
	 public:
		 /**
          * @brief Default constructor
          */
		 HybridMode();

		 /**
          * @brief Copy constructor
          */
		 HybridMode(const HybridMode& hm);

		 /**
          * @brief Constructor by params
          */
		 HybridMode(const vonMises& vm, const Gaussian& g);

		 /**
          * @brief Asignation operator
          */
		 HybridMode& operator=(const HybridMode& hm);

		 /**
		  * @brief Get the likelihood of the given data point
		  * */
         double getLikelihood(const cv::Mat& x)const;

         /**
          * @brief Access to the center of the distribution.
          * @return Constant reference to the center of the distribution.
          */
         const cv::Mat& firstMoment()const;

         /**
          * @brief Access to the center of the distribution.
          * @return Reference to the center of the distribution.
          */
         cv::Mat& firstMoment();

         /**
          * @brief Access to the concentration parameter.
          * @return Constant reference to the concentration parameter of distr.
          */
         const cv::Mat& secondMoment()const;


         /**
          * @brief Access to the concentration parameter.
          * @return Reference to the concentration parameter of distr.
          */
         cv::Mat& secondMoment();

         /**
          * @brief Get the gaussian distribution in the hybrid mode.
          * @return Constant reference to the gaussian distribution in the hybrid mode.
          */
         const Gaussian& getGaussianDist()const;
         
         /**
          * @brief Get the von mises distribution in the hybrid mode.
          * @return Constant reference to the von mises distribution
          */
         const vonMises& getVonMisesDist()const;
         
	 private:
		 /**
          * @brief Von mises component.
          */
		 vonMises vm_mode;
         
         /**
          * @brief Gaussian component in the hybrid mix
          */
		 Gaussian  g_mode;
	 };

}


#endif
