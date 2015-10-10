#ifndef _SUPERVISED_MODEL_H
#define _SUPERVISED_MODEL_H

#include "StatsModel.h"

namespace ml {

    template<typename ParamType, int d>
    class SupervisedModel : public StatsModel< ParamType, d> {
    public:

    protected:

        virtual p_result eval_model(const MatrixXd& x) = 0;

        unsigned int nClases;
        double classifyThreshold;
        double regParam;
    };


    template<int d>
    class Regresor : public SupervisedModel<HyperPlaneParams, d> {
    public:
    protected:

        p_restult eval_model(const MarixXd& x) {
            
        }

    };


}



#endif
