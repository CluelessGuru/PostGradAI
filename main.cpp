#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include <random>
#include <chrono>
#include <vector>
#include <map>
#include <stdexcept>
#include <any>
#include <map>
#include <set>
#include <iostream>
#include <iomanip>
#include <string_view>
#include <thread>

#include <mkl.h>
#include <omp.h>

#include <gsl_multimin.h>
#include <csv.hpp>
using namespace std;

//------------------------------------------------------------------------
//                             HEAD 
//------------------------------------------------------------------------
class Arguments;
double cross_validate(const Arguments* args, std::mt19937_64& engine, int npass = 0);
//------------------------------------------------------------------------
//                             SOURCE 
//------------------------------------------------------------------------
#include "util.cpp"
#include "prep.cpp"
#include "nn.cpp"
#include "read.cpp"
#include "arg.cpp"
#include "train.cpp"
#include "bsa.cpp"
#include "run.cpp"
//------------------------------------------------------------------------
//                             MAIN
//------------------------------------------------------------------------
int main(
    int nargs, char** argv)
  {
    mkl_set_num_threads(1);    
    Arguments* args = getArguments(nargs, argv);
    if (!args) return 0;

    int ncols = processArguments(args);
    args->print();

    Field_Labels labels = getFieldLabels(args->m_filename, args->m_label.size(), args->m_label.data());
    printFieldLabels(labels);

    if (args->m_natoms) {
      FeatureInfo feature_info = 
	cntFeatures(ncols, args->m_output.size(), &(args->m_output[0]),
                           args->m_exclude.size(), &(args->m_exclude[0]));
      run_feature_selection(args, feature_info); //feature slection
    }
    else{
      int seed = args->m_seed;
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 engine(seed);
      mkl_set_num_threads(std::max(1, args->m_nthreads));    
      args->m_msglvl = 4;
      double loss = cross_validate(args, engine);
      sprint("Average Error Measure ", loss, "\n");
    }
    delete args;

    do_user_stop(1);
    return 0;
  }

