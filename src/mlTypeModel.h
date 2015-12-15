#ifndef _ML_TYPE_MODEL_H_
#define _ML_TYPE_MODEL_H_
#include "stats.h"
#include <stdlib.h>
#include "common.h"
#include "dbgLogger.h"

namespace tinf
{
    /**
     * @brief Interface of a statistical model for prediction. All statistical model
     * for clustering or classification must inherit from this class.
     */
    class StatsModel
    {
        public:
            
            /**
             * @brief Get cluster and likelihood of the new observed point p.
             * @param p Point to evaluate.
             * @return Container of with[0] best label and [1] likelihood.
             */
            virtual cv::Vec2d predict(const cv::Mat& x, cv::Mat* outYz = NULL)const = 0;
    };
    
    /**
     * @brief Interface and template class for maximum likelihood models. Specifically
     * this class implements the concept of mixture models of several distributions.
     * The distribution is the template argument, and the class implements also the 
     * interface of statistical model since a mixture model is used for predicting
     * and classification.
     */
    template<typename TDist>
    class MLTypeModel : public StatsModel
    {
        public:
            
            /**
             * @bried Default consntructor
             */
    		MLTypeModel()
    		: pWeights(cv::Mat()), sampleLLH(cv::Mat()), latentZ(cv::Mat()),
    		  QEnd(std::numeric_limits< double >::min()),
    		  N(0), K(0), D(0), verbose(false){

    		}

            /**
             * @brief Destructor.
             */
            virtual ~MLTypeModel(){

            };
            
            /**
             * @brief Get the likelihood of the training examples.
             * @return Likelihod of the training data.
             */
            double getModelLikelihood() {
                return QEnd;
            }
    
            /**
             * @brief Get cluster and likelihood of the new observed point p.
             * @param p Point to evaluate.
             * @return Container of with[0] best label and [1] likelihood.
             */
            virtual cv::Vec2d predict(const cv::Mat& x, cv::Mat* outYz = NULL)const {
            	cv::Mat yzn;

            	/// Compute likelihood and if needed copy out
            	cv::Vec2d p = computeLikelihood(x, yzn);
                if(outYz != NULL) (*outYz) = yzn.clone();

                return p;
            }


            /**
             * @brief Get the labels of the trining data.
             * @return constant reference to the training labels.
             */
            const std::vector< int >& getTrainingLabels()const {
                return labels;
            }

            /**
             * @brief Access to the learned mixture.
             * @return Constant reference to the mixture model.
             */
            const std::vector< TDist >& getMixture()const {
                return mixture;
            }
            
            /**
             * @return Constant reference to the mixture weights.
             */
            const cv::Mat& getMixtureCoefs()const {
                return pWeights;
            }
            
            /**
             * @return The number of components in the mixture
             */
            const int& getNumberOfComponents()const {
                return K;
            }                                   
            
            /**
             * @brief Get the number of parameters, also called the model dimension.
             * This method depends on the type of distribution used to extend the concept
             * of maximum likelihood type models. This is why each class that inherits
             * from this class must implement its own number of parameters.
             * @return Number of parameters in the model
             */
            virtual int getParamsNumber() = 0;

            /**
             * @brief Pretty print the model to a string for futher verbose and 
             * later debugging.
             * @return A string formated with the parameters of the model.
             */
            std::string model2String(){
                std::stringstream ss;

                ss << "Paramters\n";
                
                ss << "\t\t\t weight: " 
                   << pWeights
                   << std::endl;
                
                cv::Mat meanConcat = mixture[0].firstMoment();
                for(int i = 1; i<K; i++)
                    cv::vconcat(meanConcat, mixture[i].firstMoment(), meanConcat);                  
                
                ss << "\t\t\t centers: "
                   << std::endl
                   << meanConcat;
                       
                return ss.str();
            } 

            
       protected:
            
            /**
             * @brief Do expectation step.
             * @param Data to do expectation step.
             */
            void expectationStep(const cv::Mat* X) {
                cv::Mat yzn = cv::Mat::zeros(1, K, CV_64FC1);
                cv::Vec2d pMax;
                Nk = std::vector< int > (K, 0);

                /// Compute responsibilities
                for(int n = 0; n <N ; n++)
                {
                    yzn = cv::Scalar::all(0.0);
                    const cv::Mat& xn = (*X).row(n);

                    pMax = predict(xn, &yzn);

                    /// Normalizing
                    double sum  = cv::sum(yzn)[0]; 
                    
                    sum = (sum == 0.0)? 1.0: sum ;
                    yzn = yzn / sum;
                    
                    /// Copy to latent variables
                    yzn.copyTo( latentZ.row(n) );

                    /// Cluster assingation of point
                    labels[n] = pMax[1];
                    
                    Nk[labels[n]]++;
                }            
            }

            /**
             * @brief Compute the total training data likelihood.
             * @param X Data to evaluate the likelihood.
             * @return Likelihood of the data.
             */
            double computeDataLikelihood(const cv::Mat* x) {
            	QEnd = cv::sum(sampleLLH)[0];

                return QEnd;
            }                     
            
            /**
             * @brief Initialize the expectation maximization algorithm. This method
             * also depends on the type of distribution and must be implemented in the
             * children classes.
             * @param X Pointer to the data to do the initialization
             */
            virtual void init(const cv::Mat* X) = 0;

            /**
             * @brief Do maximization step. This process also depends on the distribution
             * used to build the model.
             * @param X Data to do maximization step.
             */
            virtual void maximizationStep(const cv::Mat* X) = 0;

            /**
             * @brief Compute the likelihood of the given datapoint. The likelihood
             * procedure is the same for all types of mixrure.
             * TODO: Currently must be implemented by each children class since now
             * there is different support of computation in the likelihood for the
             * different types of  distributions available.
             */
            virtual cv::Vec2d computeLikelihood(const cv::Mat& xn, cv::Mat& yzn)const = 0;

            /**
             * @brief Implementation of the expectation maximization algorithm for 
             * likelihood type models.
             * @param _X Data to fit the model with.
             */
            void process(cv::Mat* X) {
                std::stringstream ss;
                /// Initialize Expectation Maximization for hybrid mixture
                init( X );

                double Qold = std::numeric_limits< double >::min();
                double Qnew = std::numeric_limits< double >::min();

                /// Compute initial data likelihood
                Qold = computeDataLikelihood( X );                
                
                DbgLogger::LOGI(MLTAG, __FUNCTION__, "Processing clustering now taking place");

                /// Expectation maximization loop
                for(int it = 0; it < maxIt; it++) {
                    ss.str("");
                    
                    /// TODO: Currently we only stop the optimization procedure
                    /// But it will be better to trow an exeption
                    if(isnan(Qold) || isinf(Qold)){
                        break;
                    }
                    
                    /// Do expectation step
                    expectationStep( X );

                    /// Do maximization step
                    maximizationStep( X );

                    /// Compute complete data likelihood
                    Qnew = computeDataLikelihood( X );

                    /// Show results on the screen
                    if(verbose){
                        DbgLogger::LOGI( MLTAG, currentState, "Log likelihood: %lf", Qnew );
                    }
                    
                    ///  Stop if data likelihood does not change
                    if( fabs(Qnew - Qold) < eps) break;
                    Qold = Qnew;                    
                }

            
            }

            /**
             * @brief String Tag of the implementing model.
             */
            
            std::string MLTAG;
            
            /**
             * @brief Current state of the algorithm.
             */
            std::string currentState;
            
            /**
             * @brief Von mises components.
             */
            std::vector< TDist > mixture;

            /**
             * @brief Effective number of points assigned
             */
            std::vector< int > Nk;                   // Number of data asigned to components                      
           
            /**
             * @brief Labels of the training data.
             */
            std::vector< int > labels;

            /**
             * @brief Weight parameters of the components.
             */
            cv::Mat    pWeights;
            
            /** 
             * @brief Likelihood of the training process.
             */
            cv::Mat sampleLLH;

            /**
             * @brief Latent variables per cluster.
             */
            cv::Mat latentZ;
            
            /**
             * @brief Complete data likelihood.
             */
            double QEnd;
            
            /**
             * @brief Threshold to stop maximization.
             */
            double eps;
            
            /**
             * @brief Allowed maximum number of iterations.
             */
            int maxIt;

            /**
             * @brief Number of components.
             */
            int K;

            /**
             * @brief Number of observations.
             */
            int N;

            /**
             * @brief Number of data dimensions.
             */
            int D;
            
            /**
             * @brief Do verbose.
             */
            bool verbose;
    };
        

    /********************************************************************************\
     *                           MIXTURE MODEL SELECTOR                             *
    \********************************************************************************/
    
    /**
     * @brief This class implement the process of model selection for the maximum likelihood
     * type models. Currently this method only works for GMM and VMMM.
     * TODO: Add feature to suppoort also the hybrid mixture.
     */
    template<typename mltypemodel >
    class MLTypeModelSelector
    {
        public:
            
            /**
             * @brief Default constructor.
             */
            MLTypeModelSelector()
            { }
            
            /**
             * @brief Copy constructor.
             * @param selector Instance to copy.
             */
            MLTypeModelSelector(const MLTypeModelSelector< mltypemodel >& selector)
            {
                this->BIC = selector.BIC.clone();
                this->bicLower =  selector.bicLower;
                this->bicUpper = selector.bicUpper;
                this->models = selector.models;
            }
            
            /**
             * @brief Constructor and process of model selection using bayesian information criteria.
             * @param X Data to model.
             * @param lowLimit Lower limit of the range of paramter selection.
             * @param upperLimit Upper limit of the range of parameter selection.
             * @param maxIt Maximum number of iterations to process for each model.
             * @param eps Threshold to stop parameter estimation.
             * @param verb Verbose to see results on screen.
             */
            MLTypeModelSelector( const cv::Mat& X, int lowLimit = 1, int upperLimit = 3, 
                                 int _maxIt = 100, double _eps = 1e-8, bool verb = true)
            : N(X.rows), eps(_eps), maxIt(_maxIt), bicLower(lowLimit), bicUpper(upperLimit)
            {
                
                BIC = cv::Mat::zeros(bicUpper - bicLower + 1, 1, CV_64FC1);
                models = std::vector< mltypemodel >(bicUpper - bicLower + 1);
                
                
                int i=0;
                for(int k = bicLower; k <= bicUpper; k++, i++)
                {            
                    models[i] = mltypemodel(X, k, false, maxIt, _eps);

                    int kk = models[i].getParamsNumber();
                    double ll = models[i].getModelLikelihood();

//                     Computing BIC
                    double bicValue = 2.0*ll - double(kk)*log(N);

                    BIC.at< double > (i, 0) = bicValue;
                    
                    if(verb)
                    {
                        std::cout << "\t\tComponents: "
                                  << k
                                  << " Free parameters: "
                                  << kk
                                  << " logLikelihood: "
                                  << ll
                                  << " bic: "
                                  << bicValue
                                  << std::endl;
                                  //getchar();
                    }
                    
                }         
                
                cv::Point pMax;
                
                cv::minMaxLoc(BIC, NULL, NULL, NULL, &pMax);
                
                bestModelIndex = pMax.y; 
                
                std::cout << "Best model with " << models[bestModelIndex].getNumberOfComponents()<<std::endl;
            }
            
            /**
             * @brief Operator of assignation.
             * @param selector New instance to copy.
             * @return This instance.
             */
            MLTypeModelSelector& operator=(const MLTypeModelSelector& selector)
            {
                this->BIC = selector.BIC.clone();
                this->bicLower =  selector.bicLower;
                this->bicUpper = selector.bicUpper;
                this->models = selector.models;
                
                return *this;
            }
            
            /**
             * @brief Destructor.
             */
            ~MLTypeModelSelector()
            { }
            
            /**
             * @brief Get the best model that fits the data using BIC.
             * @return Best model
             */
            mltypemodel getBestModel()
            {
                /// Returns the best model
                return models[ bestModelIndex ];
            }
            
            /**
             * @brief Get the curve of BIC generated by the selection process.
             * @return A matrix of the BIC curve.
             */
            const cv::Mat& getBIC()const;
            
        protected:
            
            /**
             * @brief Generated models in the selection process.
             */
            std::vector< mltypemodel > models;
            
            /**
             * @brief Curve of bayessian information criteria.
             */
            cv::Mat BIC;
            
            /**
             * @brief Epsilon to stop the EM process.
             */
            double eps;
            
            /**
             * @brief Number of points in the dataset
             */
            int N;
            
            /**
             * @brief Maximum number of iteration for EM process.
             */
            int maxIt;
            
            /**
             * @brief Lower limit of the range of paramter selection.
             */
            int bicLower;
            
            /**
             * @brief Upper limit of the range of paramter selection.
             */
            int bicUpper;                  
            
            /**
             * @brief The index of the best model that fits the data
             */
            int bestModelIndex;
    };

}   


#endif
