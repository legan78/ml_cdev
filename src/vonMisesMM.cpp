
#include "mxTypeModel.h"



namespace tinf
{
/*******************************************************************************************\
 *                             MIXTURE OF VON MISES DISTRIBUTION                           *
\*******************************************************************************************/

    /**
     * @brief Default constructor.
     */
    VMEM::VMEM()
    { }

    /**
     * @brief Copy constructor.
     * @param vmem
     */
    VMEM::VMEM(const VMEM& vmem)
    {
        copy(vmem);
    }

    /**
     * @brief Constructor by computing the EM for mixture of
     * von Mises with the given training data.
     * @param X Sample of circular data for training.
     * Data must be in the unit sphere.
     * @param K Number of components in the mixture.
     * @param maxIt Maximum number of iterations allowed
     */
    VMEM::VMEM(const cv::Mat& _X, int _K, bool _verbose, int _maxIt, double _eps)
    {
        K = _K;
        eps = _eps;
        maxIt = _maxIt;
        verbose = _verbose;

        /// Cast away the constness
        cv::Mat * X = const_cast<cv::Mat*>(&_X);

        /// Do clustering using EM
        process( X );
    }

    /**
     * @brief Assination operator of the mixture of von mises
     * @param vmem Instance of the von mises mixture to copy
     * @return A reference to this instance.
     */
    VMEM& VMEM::operator=(const VMEM& vmem)
    {
        copy(vmem);

        return *this;
    }

    /**
     * @brief Destructor.
     */
    VMEM::~VMEM()
    {

    }


    /**
     * @brief Copy a new instance in this instance.
     * @param vmem Instance to copy
     */
    void VMEM::copy(const VMEM& vmem)
    {
        MLTAG = vmem.MLTAG;
        currentState = vmem.currentState;
        this->mixture    = vmem.mixture;
        this->Nk         = vmem.Nk;
        this->labels     = vmem.labels;
        this->pWeights    = vmem.pWeights.clone();
        this->sampleLLH  = vmem.sampleLLH.clone();
        this->latentZ    = vmem.latentZ.clone();                
        
        this->QEnd    = vmem.QEnd;
        this->maxIt   = vmem.maxIt;

        this->sampleLLH  = vmem.sampleLLH.clone();
        this->K          = vmem.K;
        this->D          = vmem.D;
        this->N          = vmem.N;
        
    }


    /**
     * @brief Do maximization step.
     * @param X Data to do maximization step.
     */
    void VMEM::maximizationStep(const cv::Mat* X)
    {
        cv::Mat totalLatentZ;
        cv::Mat r = cv::Mat::zeros(1, D, CV_64FC1);

        /// Compute total probability of latent variables
        cv::reduce(latentZ, totalLatentZ, 0, CV_REDUCE_SUM);

    //    std::cout << " New estimate" << std::endl;
        /// Update parameters of the mixture
        for(int k = 0; k<K; k++)
        {
            r = cv::Scalar::all(0.f);

            /// Compute new weights of clusters
            pWeights.at<double>(0,k) = totalLatentZ.at< double >(0, k) / double(N);

            /// Get the weighted vector
            for(int n = 0; n<N; n++)
                r = r + latentZ.at< double >(n, k)*(*X).row(n);

            double rnorm = sqrt( r.dot(r) );

            /// Update the component parameters
            double rnorm1 = (rnorm == 0.0)?1.0:rnorm;
            mixture[k].getCenter() = r / rnorm1; /// Avoid zero division

            double kappa = vonMises::fisherAInv( rnorm / totalLatentZ.at< double >(0, k) );

            /// To avoid singularities in the likelihood
            mixture[k].getConcentration() = std::min<double>(vonMises::MAX_KAPPA, kappa);

            /// To avoid singularities in the likelihood
            mixture[k].secondMoment().at< double >(0,0) = mixture[k].getConcentration();
        }

    }

    /**
     * @brief Compute the likelihood of the given datapoint.
     * This procedure is done using the log sum exp
     */
    cv::Vec2d VMEM::computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const{
        cv::Vec2d v;
        cv::Point pMax;

        /// Latent variable
        yzn = cv::Mat::zeros(1, K, CV_64FC1);

        /// Evaluate each component
        for(int k = 0; k < K; k++)
        {
            /// Be careful with singularities in the getLikelihood
            yzn.at< double >(0, k) = pWeights.at<double>(0,k)*mixture[k].getLikelihood(xn);
        }

        /// Get maximum likelihood component
        cv::minMaxLoc(yzn, 0, 0, 0, &pMax);

        /// Prediction
        v[0] = cv::sum(yzn)[0];
        v[1] = pMax.x;

        return v;
    }


    /**
     * @brief Initialize the expectation maximization by using the spherical
     * K-means.
     */
    void VMEM::init(const cv::Mat* X)
    {
        /// Parameters of the training data
        N = (*X).rows;
        D = (*X).cols;

        /// Spherical K-Means initialization
        SPHKMeans  sph(*X, K);

        /// Init parameters of the data
        latentZ = cv::Mat::zeros(N, K, CV_64FC1);
        mixture = std::vector<vonMises>( K );
        pWeights      = cv::Mat::zeros(1, K, CV_64FC1);
        labels  = sph.getTrainingLabels();
        Nk      = std::vector< int >(K, 0);
        sampleLLH = cv::Mat::zeros(1, N, CV_64FC1);

        /// Const reference to the centers of the spherical K-Means results
        const cv::Mat& centers = sph.getCenters();

        cv::Mat r = cv::Mat::zeros( K, D, CV_64FC1 );

        /// Count initial assignations
        for(int i=0; i<N; i++)
        {
            int k = labels[i];
            Nk[ k ]++;

            const cv::Mat& xn = (*X).row(i);
            cv::Mat c  = r.row(k);

            c = xn + c;

            c.copyTo( r.row(k) );
        }

        /// Estimate the concentration parameter and cluster weights
        ///std::cout << "First estimate: " << std::endl;

        for(int k = 0; k < K; k++)
        {
            const cv::Mat& c = r.row(k);
            double rNorm = sqrt( c.dot(c) );

            int _Nk = (Nk[k] == 0)?1:Nk[k];
            double R = rNorm / double(_Nk);

            /// Use the fisher function to estimate the concentration parameter.
            double kappa = vonMises::fisherAInv(R);
            mixture[k].getConcentration() = std::min<double>(vonMises::MAX_KAPPA, kappa);

            /// Estimate the cluster weights
            pWeights.at<double>(0,k) = double(Nk[k]) / double(N);

            /// Get the center of the clusters
            mixture[k].getCenter() = centers.row(k).clone();

        }

    }


    /**
     *
     * @return
     */
    int VMEM::getParamsNumber()
    {
        return K*(2 + D) - 1;
    }

}
