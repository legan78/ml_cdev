#ifndef _STATS_H_
#define _STATS_H_

#include "distributions.h"


namespace tinf{
    /**
     * @brief Generates a sample of size n from a Population of size N by index selection.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Vector containing the index of the selected samples.
     */
    std::vector<int> samplingWithoutReplacement(int populationSize, int sampleSize);
    
}


#endif
