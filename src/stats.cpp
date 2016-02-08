#include "../include/stats.h"
#include "../include/common.h"
#include <stdio.h>
#include <set>
#include <fstream>
#include <vector>
#include <cstdlib>


namespace tinf
{
    
    /**
     * @brief Generates a sample of size n from a Population of size N by index selection.
     * @param populationSize Size of population.
     * @param sampleSize Size of sample.
     * @return Vector containing the index of the selected samples.
     */    
    std::vector<int> samplingWithoutReplacement(int populationSize, int sampleSize)
    {
        // Generator of random numbers
        std::default_random_engine generator
        ( std::chrono::system_clock::now().time_since_epoch().count() );

        std::uniform_real_distribution<double> U(0.0,1.0);

        std::vector<int> index(sampleSize);

        int& N = populationSize;
        int& n = sampleSize;
        int t = 0;
        int m = 0;
        double u;

        for(;m<sampleSize;)
        {
            u = U( generator );

            if( (N-t)*u >= n-m ) t++;
            else{
                index[m] = t;
                t++;
                m++;
            }

        }

        return index;
    }
}

