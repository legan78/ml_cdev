#include <opencv2/core/core.hpp>

#include "distributions.h"


namespace tinf
{
    /********************************************************************************\
     *                         VON MISES DISTRIBUTION                               *
    \********************************************************************************/

    /// Maximum kappa allowed to avoid singularities
    double vonMises::MAX_KAPPA = 250.f;

    /*
     * Class to implement the von Mises distribution. Constructs the standard
     * circular gaussian
     */
    vonMises::vonMises()
    : theta0(cv::Mat::zeros(1,1,CV_64FC1)), kappa(1.f), 
      kappaMat(cv::Mat::zeros(1,1,CV_64FC1)){ }

    /**
     * @brief Constructor by center and concentration parameter.
     * @param _theta0 Center of the distribution.
     * @param _kappa Concentration parameter of the distribution.
     */
    vonMises::vonMises(const cv::Mat& _theta0, double _kappa)
    : theta0(_theta0.clone()), kappa(_kappa),
      kappaMat(cv::Mat::ones(1,1,CV_64FC1)*_kappa){ }

    /**
     * @brief Constructor by learning the parameters from data using the
     * maximum likelihood approach of the von Mises distribution.
     * @param X A sample of circular data.
     */
    vonMises::vonMises(const cv::Mat& X)
    {
        theta0   = cv::Mat::zeros(1, X.cols, CV_64FC1); 
        kappaMat = cv::Mat::zeros(1, 1, CV_64FC1);
        int N = X.rows;

        /// Estimate the mean
        for(int i=0; i<N; i++)
            theta0 = theta0 + X.row(i);

        double theta0Norm = sqrt( theta0.dot(theta0) );

        /// Estimate the concentration parameter
        double R = theta0Norm / double(N);

        /// Use the fisher function to estimate the concentration parameter.
        kappa = fisherAInv(R);    

        /// The mean is a unit vector
        theta0 =  theta0 / theta0Norm;
        
        kappaMat.at< double >(0,0) = kappa;
    }

    /**
     * @brief Constructor by learning the parameters from data using the
     * maximum likelihood approach of the von Mises distribution.
     * @param X A sample of circular data.
     */
    vonMises::vonMises(const cv::Mat& X, const cv::Mat& mask, double tau)
    {
        theta0 = cv::Mat::zeros(1, X.cols, CV_64FC1); 
        kappaMat = cv::Mat::zeros(1, 1, CV_64FC1);
        
        int N = X.rows;

        /// Estimate the mean with some mask
        for(int i=0; i<N; i++)
        {
            if(mask.at<double>(i,0) > tau)
            {
                theta0 = theta0 + X.row(i);
                N--;
            }
        }

        N = (N==0)?1:N;
        
        double theta0Norm = sqrt( theta0.dot(theta0) );

        /// Estimate the concentration parameter
        double R = theta0Norm / double(N);

        /// Use the fisher function to estimate the concentration parameter.
        kappa = fisherAInv(R);    

        /// The mean is a unit vector
        theta0 =  theta0 / theta0Norm;
        
        kappaMat.at<double>(0,0) = kappa;
    }

    /**
     * @brief Copy constructor.
     */
    vonMises::vonMises(const vonMises& vm)
    :theta0(vm.theta0.clone()), kappa(vm.kappa), kappaMat(vm.kappaMat.clone())
    { }

    /**
     * @brief Operator of assignation.
     * @param vm Instance to copy.
     * @return Reference to this.
     */
    vonMises& vonMises::operator=(const vonMises& vm)
    {
        theta0 = vm.theta0.clone();
        kappa  = vm.kappa;
        kappaMat = vm.kappaMat.clone();

        return *this;
    }

    /**
     * Destructor.
     */
    vonMises::~vonMises()
    {
        /// 
    }

    /**
     * @brief Get the likelihood of the given point in the von Mises distribution.
     * @param x Data to evaluate the likelihood.
     * @return Likelihood of the given point.
     */
    double vonMises::getLikelihood( const cv::Mat& x )const
    {
        double pi = 3.1416;
        double fact = 2.0*pi*besselI0(kappa);

        double cosSim = x.dot( theta0 );

    //    std::cout << "Fact  " << besselI0(kappa) <<std::endl;

        return exp( kappa*cosSim ) / fact;
    }


    /**
     * @brief Access to the center of the distribution.
     * @return Constant reference to the center of the distribution.
     */
    const cv::Mat& vonMises::getCenter()const
    {
        return theta0;
    }

    /**
     * @brief Access to the center of the distribution.
     * @return Reference to the center of the distribution.
     */        
    cv::Mat& vonMises::getCenter()
    {
        return theta0;
    }

    /**
     * @brief Access to the concentration parameter.
     * @return Constant reference to the concentration parameter of distr.
     */
    const double& vonMises::getConcentration()const
    {
        return kappa;
    }

    /**
     * @brief Access to the concentration parameter.
     * @return Reference to the concentration parameter of distr.
     */
    double& vonMises::getConcentration()
    {
        return kappa;
    }
    
    /**
     * @brief Access to the center of the distribution.
     * @return Constant reference to the center of the distribution.
     */
    const cv::Mat& vonMises::firstMoment()const
    {
        return theta0;
    }

    /**
     * @brief Access to the center of the distribution.
     * @return Reference to the center of the distribution.
     */        
    cv::Mat& vonMises::firstMoment()
    {
        return theta0;
    }

    /**
     * @brief Access to the concentration parameter.
     * @return Constant reference to the concentration parameter of distr.
     */
    const cv::Mat& vonMises::secondMoment()const
    {
        return kappaMat;
    }


    /**
     * @brief Access to the concentration parameter.
     * @return Reference to the concentration parameter of distr.
     */
    cv::Mat& vonMises::secondMoment()
    {
        return kappaMat;
    }
                
    

    /**
     * @brief Method to calculate the concentration parameter by dobson proposal
     * @param x Parameter to evaluate the dobson proposal
     * @return Value of dobson proposal for concentration parameter
     */    
    double vonMises::fisherAInv(double x)
    {
        if(x<0.53)
            return 2.0*x + x*x*x + 5.0*pow(x,5)/6.0;

        if(x >= 0.53 && x <=0.85 )
            return -0.4 + 1.39*x + (0.43)/(1.0-x);

        return 1.0/(x*x*x - 4.0*x*x + 3.0*x);
    }

    double vonMises::besselI0(double x)
    // Returns the modied Bessel function I0(x) for any real x.
    {
        double ax,ans;

        double y; //Accumulate polynomials in single precision.

        ax = fabs( x);

        if(ax < 3.75) 
        { //Polynomial t.
            y = x/3.75;
            y *= y;
            ans = 1.0 + y*(3.5156229 + y*(3.0899424   + y*( 1.2067492 + 
                        y*(0.2659732 + y*(0.360768e-1 + y*0.45813e-2 )))));
        } else {

            y = 3.75/ax;



            ans = ( exp(ax)/sqrt(ax) )*( 0.39894228 + y*(0.1328592e-1 + y*(0.225319e-2 + 
                                         y*(-0.157565e-2 + y*(0.916281e-2 + y*(-0.2057706e-1 + 
                                         y*(0.2635537e-1 + y*(-0.1647633e-1 + y*0.392377e-2))))))));
        }

        return ans;
    }


    /********************************************************************************\
     *                          GAUSSIAN DISTRIBUTION                               *
    \********************************************************************************/

    /// Maximum kappa allowed for the distribution to avoid singularities
    double Gaussian::GAUSS_MIN_VAR = 0.004;

    /**
     * @brief Default
     */
    Gaussian::Gaussian()
    { }

    /**
     * @brief Constructor by mean and covariance matrix
     * @param _mu Mean of the distribution.
     * @param _S Covariance matrix.
     */
    Gaussian::Gaussian(const cv::Mat& _mu, const cv::Mat& _S)
    : mu(_mu.clone()), S(_S.clone()), Sinv(S.inv())
    { }

    /**
     * @brief Constructor by learning the parameters from data using the
     * maximum likelihood approach of the Gaussian distribution.
     * @param X A sample of training data.
     */
    Gaussian::Gaussian(const cv::Mat& X)
    {   
        /// Number of observations
        int N = X.rows;

        if( N<=1 )
        {
            std::cerr << "Invalid to construct gaussian with <=1 training data" << std::endl;
            exit(-1);
        }

        /// Initialize distribution parameters
        S  = cv::Mat::zeros(X.cols, X.cols, CV_64FC1);

        /// Summing all data in X
        cv::reduce(X, mu, 0, CV_REDUCE_SUM);
        mu = mu / N;

        /// Estimation of the covariance matrix
        for(int i = 0; i< N; i++)
        {
            cv::Mat mx = X.row(i) - mu;
            mx = mx.t()*mx;

            S = S + mx;
        }

        /// Unbias estimator
        S = S/(N-1);

        /// Inverse
        Sinv = S.inv();
    }

    /**
     * @brief Copy constructor.
     */
    Gaussian::Gaussian(const Gaussian& g)
    : mu(g.mu.clone()), S(g.S.clone()), Sinv(g.Sinv.clone())
    {         }

    /**
     * @brief Operator of assignation.
     * @param g Instance to copy.
     * @return Reference to this.
     */
    Gaussian& Gaussian::operator=(const Gaussian& g)
    {
        mu = g.mu.clone();
        S  = g.S.clone();
        Sinv = g.Sinv.clone();

        return *this;
    }

    /**
     * Destructor.  
     */
    Gaussian::~Gaussian()
    { }


    /**
     * @brief Get the likelihood of the given point in the Gaussian distribution.
     * @param x Data to evaluate the likelihood.
     * @return Likelihood of the given point.
     */
    double Gaussian::getLikelihood(const cv::Mat& x)const
    {
        /// Data dimension
        int d = x.cols;

        /// Compute factors
        double pi = 3.1416;  
        double fact = pow(2.0*pi, d);
        
        cv::Mat eigenValues, eigenValuesInv, eigenVectors;
        cv::eigen(S, eigenValues, eigenVectors );
        cv::max(eigenValues, GAUSS_MIN_VAR, eigenValues);
        
        eigenValuesInv = 1.0/eigenValues;
        double det = 1.0;
        
        for(int i = 0; i<eigenValues.cols; i++)
            det *= eigenValues.at< double >(0, i);
                        
        cv::Mat mx = x - mu;
        mx = mx*Sinv*mx.t();
        
//        if(isnan(mx.at<double>(0, 0)) || isinf(mx.at<double>(0, 0)))std::cout<<"Determinant nan"<< mu <<std::endl;

        /// return likelihood
        return 1.0/sqrt(fact*det) * exp(-0.5 * mx.at<double>(0, 0));    
    }

    /**
     * @brief Access to the center of the distribution.
     * @return Constant reference to the center of the distribution.
     */
    const cv::Mat& Gaussian::getMean()const
    {
        return mu;
    }

    /**
     * @brief Access to the center of the distribution.
     * @return Reference to the center of the distribution.
     */        
    cv::Mat& Gaussian::getMean()
    {
        return mu;
    }

    /**
     * @brief Access to the concentration parameter.
     * @return Constant reference to the concentration parameter of distr.
     */
    const cv::Mat& Gaussian::getCovariance()const{
        return S;
    }

    /**
     * @brief Access to the concentration parameter.
     * @return Constant reference to the concentration parameter of distr.
     */
    const cv::Mat& Gaussian::getInvCovariance()const{
    	return Sinv;
    }


    /**
     * @brief Access to the concentration parameter.
     * @return Constant reference to the concentration parameter of distr.
     */
    cv::Mat& Gaussian::getInvCovariance() {
    	return Sinv;
    }

   /**
    * @brief Access to the center of the distribution.
    * @return Constant reference to the center of the distribution.
    */
    const cv::Mat& Gaussian::firstMoment()const
    {
        return mu;
    }

   /**
    * @brief Access to the center of the distribution.
    * @return Reference to the center of the distribution.
    */        
    cv::Mat& Gaussian::firstMoment()
    {
        return mu;
    }

   /**
    * @brief Access to the concentration parameter.
    * @return Constant reference to the concentration parameter of distr.
    */
    const cv::Mat& Gaussian::secondMoment()const
    {
        return S;
    }


   /**
    * @brief Access to the concentration parameter.
    * @return Reference to the concentration parameter of distr.
    */
    cv::Mat& Gaussian::secondMoment()
    {
        return S;
    }
            

    /**
     * @brief Access to the concentration parameter.
     * @return Reference to the concentration parameter of distr.
     */
    cv::Mat& Gaussian::getCovariance()
    {
        return S;
    }


    /********************************************************************************\
     *                            HYBRID DISTRIBUTION                               *
    \********************************************************************************/
    
	 // Default constructor
	HybridMode::HybridMode(){

	}

	 // Copy constructor
	HybridMode::HybridMode(const HybridMode& hm)
	: vm_mode(hm.vm_mode), g_mode(hm.g_mode) {

	}

	 // Constructor by params
	HybridMode::HybridMode(const vonMises& vm, const Gaussian& g){
		vm_mode = vm;
		g_mode  = g;
	}

	 /// Asignation operator
	HybridMode& HybridMode::operator=(const HybridMode& hm){
		 vm_mode = hm.vm_mode;
		 g_mode  = hm.g_mode;

		 return *this;
	 }

	 /// Do data prediction
	 double HybridMode::getLikelihood(const cv::Mat& xn)const{
		 /// Get individual likelihood
		 double ll_vm = vm_mode.getLikelihood( xn.colRange(0, 2) );
		 double ll_g  = g_mode.getLikelihood( xn.colRange(2,3) );
         
//         if(isnan(ll_vm*ll_g) || isinf(ll_vm*ll_g)){
//             std::cout << "is nan " << ll_vm << "  "  << ll_g << std::endl;
//         }

		 return ll_vm*ll_g;// Total likelihood corresponds to the product
	 }

     /**
      * @brief Access to the center of the distribution.
      * @return Constant reference to the center of the distribution.
      */
     const cv::Mat& HybridMode::firstMoment()const{
    	 return g_mode.firstMoment();
     }

     /**
      * @brief Access to the center of the distribution.
      * @return Reference to the center of the distribution.
      */
     cv::Mat& HybridMode::firstMoment(){
    	 return g_mode.firstMoment();
     }
     /**
      * @brief Access to the concentration parameter.
      * @return Constant reference to the concentration parameter of distr.
      */
     const cv::Mat& HybridMode::secondMoment()const{
    	 return g_mode.secondMoment();
     }


     /**
      * @brief Access to the concentration parameter.
      * @return Reference to the concentration parameter of distr.
      */
     cv::Mat& HybridMode::secondMoment(){
    	 return g_mode.secondMoment();
     }
     
     
     /**
      * 
      * @return 
      */
     const Gaussian& HybridMode::getGaussianDist()const {
         return this->g_mode;
     }

     /**
      * 
      * @return 
      */
     const vonMises& HybridMode::getVonMisesDist()const {
         return this->vm_mode;
     }     

}
