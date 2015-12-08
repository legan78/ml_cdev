#ifndef _STATSMODEL_H_
#define _STATSMODEL_H_


#include "ModelParams.h"

namespace ml {
  
  /**
   * @brief Statistical model interface
   */
  template<typename ParamType, unsigned int d>
    class StatsModel {
    public:

       /*
        * Typedef for results of prediction. Depending on the statistical model
        * it could provide likelihood and class label
        */
        typedef std::vector<double>(d) p_result;

        typedef ModelParams<ParamType> m_params;

            
       /**
        * @brief predict the value of the model for the new unseen data.
        * @param x Data point to be evaluated in the model.
        * @return Value of prediction and classification.
        */ 
        virtual p_result predict(const MatrixXd& x) = 0;

       /**
        * @brief Access to mode parameters.
        * @return Constant reference to model parameters.
        */
        const m_params & get_params() const 
        { return params; }
                      
    protected:

        /*
         * Statistical model parameters
         */
        m_params params;

    };
}

#endif
