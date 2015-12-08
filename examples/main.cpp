#include <iostream>
#include <cstdlib>


#include "ModelParams.h"

int main(int argc, char** argv) {

    std::cout << "Machine learning is a reallity" << std::endl;

    ml::NormalDist::NormalParams np = ml::NormalDist::NormalParams::init_params(MatrixXd::Zero(1,1), MatrixXd::Ones(1,1));

    std::cout << ml::NormalDist::nomal_likelihood(MatrixXd::Zero(1,1), np) << std::endl;

    return EXIT_SUCCESS;    
}
