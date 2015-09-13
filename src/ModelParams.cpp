#include "ModelParams.h"


namespace ml {

    NormalDist::NormalParams NormalDist::NormalParams::init_params(const MatrixXd& mean, const MatrixXd& S) {
        NormalParams params;

        params.mean = mean;
        params.Sigma = S;
        
        params.invSigma = S.inverse();
        params.weight = 1.0;

        return params;
    }

    double NormalDist::nomal_likelihood(const MatrixXd& x, const NormalDist::NormalParams& paramDist) {
        const MatrixXd& m = paramDist.mean;
        const MatrixXd& S = paramDist.Sigma;
        const MatrixXd& iS = paramDist.invSigma;

        double val = 0;
        double pi = 3.1416;
        unsigned int k = m.cols();

        double fact = sqrt( pow( 2.0*pi, k)* S.determinant() );

        MatrixXd xm = x - m;
        MatrixXd expm = xm * iS * xm.transpose();

        val = (1.0/fact)*exp(-0.5 * expm(0,0));

        return val;
    }


    LAlgoParams LAlgoParams::init_params(double lr, double eps, unsigned int it, unsigned int iState) {

        LAlgoParams params;

        params.learningRate = lr;
        params.errorTolerance = eps;
        params.iterations = it;
        params.initialState = iState;

        return params;
    }
}
