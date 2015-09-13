#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;

namespace ml {

    /**
     * @brief Class to implement the parameters of a gaussian model.
     */
    class NormalDist {
    public:

        /**
         * @brief Parameters for normal distribution.
         */
        class NormalParams {
        public:
            static NormalParams init_params(const MatrixXd& mean, const MatrixXd& S);
            /*
             * The mean of the distribution.
             */
            MatrixXd mean;

            /*
             * The covariance matrix of the distribution.
             */
            MatrixXd Sigma;

            /*
             * The inverse covariance matrix: information matrix.
             */
            MatrixXd invSigma;

            /*
             * The weight of the distribution. For mixture models is <1 for stand alone models is 1.
             */
            double weight;
        };

        /**
         * @brief Evaluates the likelihood of normal distribution of a point given some parameters.
         * @param x Point to evaluate the likelihood.
         * @param paramDist Parameters that rule the distribution.
         */
        static double nomal_likelihood(const MatrixXd& x, const NormalParams& paramDist);

    private:
        NormalDist();
        NormalDist(const NormalDist& nd);
        NormalDist& operator=(const NormalDist& d);       
    };

    /*
     * Typename for parameters of linearmodels.
     */
    typedef double HyperPlaneParams;

    /*
     * Parameters for learning algorithms
     */
    class LAlgoParams {
    public:
        /*
         * How to initialize parameters for the fitting algo.
         */
        enum {
            RANDOM_INIT=0,
            RANDOM_SELECT,
            ZEROS_INIT,
        };

        /**
         * @brief Initalization of algorithm parameters.
         * @param lr Learning rate.
         * @param eps Threshold of admitted error.
         * @param it Number of iterations to run the algorithm.
         * @param iState Method of how to initialize parameters.
         * @return Configured Algorithm parameters.
         */
        static LAlgoParams init_params(double lr, double eps, unsigned int it, unsigned int iState = RANDOM_INIT);
  
        /*
         * Learning rate for some algorithms.
         */ 
        double learningRate;

        /*
         * Error tolerance for finishing algorithm.
         */
        double errorTolerance;

        /*
         * Number of iterations allowd.
         */
        unsigned int iterations;

        /*
         * How to initialize the parameters for the algorithm.
         */
        unsigned int initialState;

    };
}



#endif
