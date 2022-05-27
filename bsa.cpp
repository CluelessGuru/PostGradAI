//------------------------------------------------------------------------
//                             GA_FITNESS
//------------------------------------------------------------------------
template<typename T>
class GA_Fitness{
  public:
    virtual ~GA_Fitness(){}
    virtual void evaluate(int natoms, T* pop, double* pop_fit) = 0;
    virtual void generate(int natoms, T* pop) = 0;
    virtual void interpret(T* params) = 0;

    int nparams(){return m_nparams;}
  protected:
    int m_nparams;
  };

template <typename T>
class NN_Fitness : public GA_Fitness<T>{
  public:
    NN_Fitness(Arguments* args, int nfeatures, int* feat2col, std::mt19937_64* engine): 
      m_args(args), m_feat2col(feat2col), m_engine(engine){this->m_nparams = nfeatures;}

    void evaluate(
        int natoms, T* pop, double* pop_fit)
    {
      int nexcl0 = m_args->m_exclude.size();
      const int* excl0_ptr = m_args->m_exclude.data();
    
      int seed = m_args->m_seed;
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      int nthreads = std::min(m_args->m_nthreads, natoms);
    
      std::seed_seq seq{seed};
      vector<std::uint32_t> seeds(natoms);
      seq.generate(seeds.begin(), seeds.end());

      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;
#if 0
      auto f = [&](int nthreads, int j){
        Arguments args(*m_args);

        std::mt19937_64 engine);  //separate engine but same seed for all
	int matoms = natoms/nthreads;

        for (int i = j*matoms; i < min((j+1)*matoms, natoms); ++i){
	  engine.seed(seeds[i]);
          T* pop_i = pop + i*this->m_nparams;
          args.m_exclude.clear();
      	  pick_exclude(this->m_nparams, m_feat2col, pop_i, args.m_exclude, nexcl0, excl0_ptr);
          pop_fit[i] = cross_validate(&args, engine);
          //sprint("ithread ", omp_get_thread_num(), " fit ", pop_fit[i], " ");
          //vprint(args.m_exclude.size(), args.m_exclude.data());
        }
      };
      vector<std::thread*> threads;
      for(int i = 1; i < nthreads; ++i)
	threads.push_back(new std::thread(f, nthreads, i));
      f(nthreads, 0);
      for(auto pthread : threads){
	pthread->join();
	delete pthread;
      }
#else
      omp_set_num_threads(nthreads);    
#pragma omp parallel
      {
        Arguments args(*m_args);
        std::mt19937_64 engine;  //separate engine but same seed for all
#pragma omp for
        for (int i = 0; i < natoms; ++i){
	  //engine.seed(seeds[i]);
	  engine.seed(seed);
          T* pop_i = pop + i*this->m_nparams;
          args.m_exclude.clear();
      	  pick_exclude(this->m_nparams, m_feat2col, pop_i, args.m_exclude, nexcl0, excl0_ptr);
          pop_fit[i] = cross_validate(&args, engine);
//#pragma omp critical
//	  {
//	    sprint("ithread ", omp_get_thread_num(), " fit ", pop_fit[i], " ");
//	    vprint(args.m_exclude.size(), args.m_exclude.data());
//	  }
        }
      }
#endif
      elapsed = (std::chrono::system_clock::now() - start);
      //cout << "Elapsed : " << elapsed.count() << " nthreads=" << m_args->m_nthreads << endl;
    }

    void generate(int natoms, T* pop)
    {
      std::uniform_real_distribution<double> ZtoO(0.0, 1.0);
      auto rand = [&]() { return ZtoO(*m_engine); };
      for (int i = 0; i < natoms; ++i){
        T* pop_i = pop + i*this->m_nparams;
        for (int j = 0; j < this->m_nparams; ++j) {
	  pop_i[j] = rand() > rand() ? T(1) : T(0);    
	}
      }
    }
    void interpret(T* pop_i)
    {
      int n = 0;
      cout << "-exclude=";
      for (int j = 0; j < this->m_nparams; ++j){
        if (is_excluded(pop_i[j])) {
	  if (n) cout << ",";
	  cout << m_feat2col[j]+1; 
	  ++n;
	}
      }
      cout << " Nexcluded=" << n << endl;;
    }
    template<typename Q>
    static void pick_exclude(
	int nparams, int* feat2col, Q* pop_i, vector<int>& excl, int nexcl0, const int* excl0)
    {
      for (int j = 0; j < nparams; ++j){
        if (is_excluded(pop_i[j])) excl.push_back(feat2col[j]+1); 
      }
      for (int j = 0; j < nexcl0; ++j) excl.push_back(excl0[j]); 
      sort(excl.begin(), excl.end());
    }
  protected:
    Arguments* m_args;
    int* m_feat2col;
    std::mt19937_64* m_engine;
    
    template<typename Q>
    static bool is_excluded(Q pop_ij){
      assert(pop_ij == Q(1) || pop_ij == Q(0));
      return pop_ij == Q(0);
    }
  };

//------------------------------------------------------------------------
//                             DBL_INT
//------------------------------------------------------------------------
class dbl_int {
  public:
  double val;
  int idx;
  void print(const char* msg) const {
    std::cout << msg << "=(" << val << "," << idx << ")" << std::endl;
  }
};

template <typename T>
class get_ith{
  public:
    get_ith(T* v, int ld, int* perm):m_v(v), m_ld(ld), m_perm(perm){
      m_ith_ptr = perm ? &get_ith<T>::pith : &get_ith<T>::ith;
    }
    virtual ~get_ith(){}
    virtual T* operator()(int i){
      return (this->*m_ith_ptr)(i);
    }
  protected:
    T* (get_ith<T>::*m_ith_ptr)(int i);
    T* m_v;
    int* m_perm;
    int m_ld;
  private:
    T* ith(int i){return m_v + i*m_ld;}
    T* pith(int i){return m_v + m_perm[i]*m_ld;}
  };

template <typename T>
class get_0th : public get_ith<T>{
  public:
    get_0th(T* v):get_ith<T>(v, 0, 0){}
    T* operator()(int i){return this->m_v;}
  private:
  protected:
  };

template<typename T>
void generate_set_based_atoms(
    int natoms, int nparams, int* ioper_t, int lioper, double* ktour_fit_t, int mx_ktour,  
    double* c, int seed, int iepoch, int nthreads, 
    get_ith<T>& get_currenti, get_ith<T>& get_pbesti, get_ith<T>& get_gbesti, get_ith<T>& get_newi,
    GA_Fitness<T>* fitness)                     
  {
    std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
    vector<std::uint32_t> seeds(natoms);
    omp_set_num_threads(nthreads);    
    std::seed_seq seq{seed, iepoch};
    seq.generate(seeds.begin(), seeds.end());
#pragma omp parallel
    {
      int ithd = omp_get_thread_num();
      double* ktour_fit = ktour_fit_t + ithd*mx_ktour;
      int* ioper = ioper_t + lioper*ithd;
      int* ivelo = ioper + 4*nparams;
      int* ioperi[4] = {ioper, ioper+nparams, ioper+2*nparams, ioper+3*nparams};
      std::mt19937_64 engine;
      auto rand = [&](){return zeroToOneUd(engine);};
#pragma omp for
      for (int i = 0; i < natoms; ++i){
        engine.seed(seeds[i]);
        T* currenti = get_currenti(i);
        T* newi = get_newi(i);
        T* pbesti = get_pbesti(i);
	T* gbest = get_gbesti(i);
        int noper[4] = {0, 0, 0 ,0};
        for (int j = 0; j < nparams; ++j){
          //(PBEST-CURRENT)
          if (currenti[j] && !pbesti[j]) ioperi[0][noper[0]++] = -j-1;
          if (!currenti[j] && pbesti[j]) ioperi[0][noper[0]++] = j;
          //(GBEST-CURRENT)
          if (currenti[j] && !gbest[j]) ioperi[1][noper[1]++] = -j-1;
          if (!currenti[j] && gbest[j]) ioperi[1][noper[1]++] = j;
      
          //(- Si)
          if (currenti[j] && gbest[j] && pbesti[j]) ioperi[2][noper[2]++] = -j-1;
          //(+ Ai)
          if (!currenti[j] && !gbest[j] && !pbesti[j]) ioperi[3][noper[3]++] = j; 
          ivelo[j] = 0;
        }
        for (int m = 0; m < 4; ++m){ 
          std::shuffle(ioperi[m], ioperi[m]+noper[m], engine);
          //k-tournament for last oper
          if (m == 3){
            int n_ktour = min(mx_ktour, noper[3]);
            //use ioperi[0], ioperi[2], noper[0] as workspace
            for (int k = 0; k < n_ktour; ++k){
              currenti[ioperi[3][k]] = 1;
              fitness->evaluate(1, currenti, ktour_fit+k);
              currenti[ioperi[3][k]] = 0;
              ioper[k] = k;
              ioperi[2][k] = ioperi[3][k]; 
            }
            sort(ioper, ioper + n_ktour, 
                 [&](int i, int j){return ktour_fit[i] < ktour_fit[j];});
            for (int k = 0; k < n_ktour; ++k) ioperi[3][k] = ioperi[2][ioper[k]];
          }
      
          double cnr = c[m]*rand()*noper[m];
          int n = (int)cnr;
          if (m > 2){ //modified count for last 2 operations
            double prop = cnr - n;
            if (rand() < prop) ++n;
          }
          int cnt = min(noper[m], n);
          for (int k = 0; k < cnt; ++k) {
            int g = ioperi[m][k];
            if (g < 0) 
              ivelo[-g-1] = -1; //remove
            else  
              ivelo[g] = 1; //add
          }
        }
        for (int j = 0; j < nparams; ++j){
          if (ivelo[j] > 0){
            assert(currenti[j] == 0);
            newi[j] = 1;
          }
          else if (ivelo[j] < 0){
            assert(currenti[j] == 1);
            newi[j] = 0;
          }
	  else{
	    newi[j] = currenti[j];
	  }
	}
      }
    }
  }

double BSAgetScale(
    double epsilon, std::mt19937_64& engine)
  {
    std::normal_distribution<double> zeroOneNd(0.0, 1.0);
    auto randn = [&](){return zeroOneNd(engine);};
    return epsilon*randn(); //scale factor
  }

template <typename T>
dbl_int BSAselect(
    int nparams, int natoms, T* pop, T* offsprings, double* pop_fit, double* offspring_fit)
  {
    dbl_int gb = {0, -1};
    for (int i = 0; i < natoms; ++i){
      T* pop_i = pop + i*nparams;
      T* offspring_i = offsprings + i*nparams;
      if (offspring_fit[i] < pop_fit[i]){
        memcpy(pop_i, offspring_i, sizeof(T)*nparams);
        pop_fit[i] = offspring_fit[i];
      }
      if (pop_fit[i] < gb.val || gb.idx == -1){
        gb.idx = i;
        gb.val = pop_fit[i];
      }
    }
    return gb;
  }

template<typename T>
double PSOupdateBest(
    double* pop_fit, double* pbest_fit, double& gbest_fit, 
    int natoms, int nparams, T* current, T* pbest, T* gbest)
  {
     int gbesti = -1;
     double epoch_best = std::numeric_limits<double>::infinity();
     for (int i = 0; i < natoms; ++i){
       if (pop_fit[i] < gbest_fit) {
         gbest_fit = pop_fit[i];
         gbesti = i;
       }
       if (pop_fit[i] < pbest_fit[i]){
         memcpy(pbest + i*nparams, current+i*nparams, sizeof(T)*nparams);
         pbest_fit[i] = pop_fit[i];
       }
       if (pop_fit[i] < epoch_best){
         epoch_best = pop_fit[i];
       }
     }
     if (gbesti > -1) memcpy(gbest, current+gbesti*nparams, sizeof(T)*nparams);
     return epoch_best;
  }

//------------------------------------------------------------------------
//                             BSAO
//------------------------------------------------------------------------
template<typename T>
class BSAO{
  public:
    BSAO(int natoms, int nepochs, double timeout, double mixrate, double epsilon, int seed):
      m_natoms(natoms), m_nepochs(nepochs), m_timeout(timeout), m_mixrate(mixrate), m_epsilon(epsilon), 
      m_seed(seed), m_pop(0){}

    void startingPopulation(T* pop){m_pop = pop;}

    double operator()(
        GA_Fitness<T>* fitness, T* best)
    {
      return run(fitness, best, m_pop, m_nepochs, m_timeout, m_natoms, m_mixrate, m_epsilon, m_seed);
    }

    static double run(
        GA_Fitness<T>* fitness, T* best, T* pop0,
	int nepochs, double timeout, int natoms, double mixrate, double epsilon, 
	int seed)
    {
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 engine(seed);
      int nparams = fitness->nparams();

      std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
      auto rand = [&](){return zeroToOneUd(engine);};

      //allocate permutation vector and wrk for crossover
      int* ipermv = (int*) malloc(sizeof(int)*(natoms + nparams));
      int* iwrk = ipermv + natoms;
      for (int i = 0; i < natoms; ++i) ipermv[i] = i;

      //generate populations
      const int npop = 3;
      T* pops = (T*) malloc(sizeof(T)*npop*nparams*natoms);
      T* pop = pops, *hist_pop = pop + nparams*natoms, *offsprings = hist_pop + nparams*natoms;
      if (pop0){ memcpy(pop, pop0, sizeof(T)*natoms*nparams); }
      else{ fitness->generate(natoms, pop); }

      //calculate fitness 
      double* pop_fit = (double*) malloc(sizeof(double)*2*natoms);
      double* offsprings_fit = pop_fit + natoms;
      fitness->evaluate(natoms, pop, pop_fit);

      Progress progress(1);
      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;

      //run BSA
      dbl_int gb_min;
      for (int iepoch = 0; iepoch < nepochs; ++iepoch) {
//always use the 1st old P. This is slightly different to the paper 
//where old P may not be used at all if not picked at 1st iteration
	if (iepoch == 0) {
	  fitness->generate(natoms, hist_pop); 
	}
	else{
          if (rand() < rand()) 
	    memcpy(hist_pop, pop, sizeof(T)*nparams*natoms);
	}
        std::shuffle(ipermv, ipermv+natoms, engine);
        double f = BSAgetScale(epsilon, engine);
        double mr = (rand() < rand() ? mixrate : 0);
        crossover(nparams, natoms, pop, hist_pop, ipermv, offsprings, f, mr, iwrk, engine);
        fitness->evaluate(natoms, offsprings, offsprings_fit);

        dbl_int epoch_min = BSAselect(nparams, natoms, pop, offsprings, pop_fit, offsprings_fit);
        if (!iepoch || epoch_min.val < gb_min.val) {
          gb_min.val = epoch_min.val; 
          gb_min.idx = epoch_min.idx;
	  fitness->interpret(pop + nparams * epoch_min.idx);
	  cout << iepoch+1 << " " << epoch_min.val << endl;
        }
        elapsed = (std::chrono::system_clock::now() - start);
        progress.show(epoch_min.val, iepoch+1, nepochs, elapsed.count(), timeout);
	if (elapsed.count() > timeout || do_user_stop()) break;
      }
      memcpy(best, pop + nparams * gb_min.idx, sizeof(T)*nparams);

      //free arrays
      free(ipermv);
      free(pops);
      free(pop_fit);
      return gb_min.val;
    }
  private:
    static void crossover(int nparams, int natoms, 
                   T* pop, T* hist_pop, int* hist_perm, T* offspring, 
                   double f, double mixrate, int* iwrk, std::mt19937_64& engine)
    {
      std::uniform_real_distribution<double> zeroToOne(0.0, 1.0);
      auto rand  = [&]() { return zeroToOne(engine); };
      auto randi = [&]() { return (engine)()%nparams; };

      for (int i = 0; i < natoms; ++i){
        T* pop_i = pop + i*nparams;
        T* hist_pop_i = hist_pop + hist_perm[i]*nparams;
        T* offspring_i = offspring + i*nparams;
        memcpy(offspring_i, pop_i, sizeof(T)*nparams);
        if (mixrate > 0.e0) {
          int nmix = (int)(mixrate*rand()*nparams);
          for (int j = 0; j < nparams; ++j) iwrk[j] = j;
          shuffle(iwrk, iwrk+nparams, engine);
          for (int imix = 0; imix < nmix; ++imix){
            int j = iwrk[imix];
	    offspring_i[j] = hist_pop_i[j] == pop_i[j] ? pop_i[j] : !pop_i[j];
          }
        }
        else{
          int j = randi(); 
	  offspring_i[j] = offspring_i[j] ? 0 : 1;
        }
      }
    }
  protected:
    T* m_pop;
    int m_natoms, m_nepochs, m_seed;
    double m_epsilon, m_mixrate, m_timeout;
  };

//------------------------------------------------------------------------
//                             SBMBSA
//------------------------------------------------------------------------
template<typename T>
class SBMBSA{
  public:
    SBMBSA(int natoms, int nepochs, double timeout, int nthreads, const double *c, int seed):
      m_natoms(natoms), m_nepochs(nepochs), m_timeout(timeout), m_nthreads(nthreads), 
      m_seed(seed), m_pop(0){
	for (int i = 0; i < 5; ++i) m_c[i] = c[i];
      }

    void startingPopulation(T* pop){m_pop = pop;}

    double operator()(
        GA_Fitness<T>* fitness, T* best)
    {
      return run(fitness, best, m_pop, m_nepochs, m_timeout, m_nthreads, m_natoms, m_c, m_seed);
    }

    static double run(
        GA_Fitness<T>* fitness, T* best, T* pop0,
	int nepochs, double timeout, int mxthreads, int natoms, double* c, int seed)
    {
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 engine(seed);

      int nparams = fitness->nparams();
      int mx_ktour = (int)(c[4]*nparams);
      int nthreads = std::min(mxthreads, natoms);

      std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
      std::normal_distribution<double> zeroToOneNd(0.0, 1.0);
      auto rand = [&](){return zeroToOneUd(engine);};
      auto randn = [&](){return zeroToOneNd(engine);};

      //allocate permutation vector and wrk for crossover
      int* ipermv = (int*) malloc(sizeof(int)*3*natoms); //ipermv(natoms), isort(natoms), itheta(natoms)
      int* isort = ipermv + natoms;
      int* itheta = isort + natoms;
      for (int i = 0; i < natoms; ++i) ipermv[i] = i;

      //generate populations
      int lioper = 5*nparams;
      int* ioper_t = (int*) malloc(sizeof(int)*lioper*nthreads);

      const int npop = 3;
      T* pops = (T*) malloc(sizeof(T)*npop*nparams*natoms);
      T* pop = pops, *hist_pop = pop + nparams*natoms, *offsprings = hist_pop + nparams*natoms;
      if (pop0){ memcpy(pop, pop0, sizeof(T)*natoms*nparams); } 
      else{ fitness->generate(natoms, pop); }

      //calculate fitness 
      double* pop_fit = (double*) malloc(sizeof(double)*(2*natoms+mx_ktour*nthreads));
      double* offsprings_fit = pop_fit + natoms, *ktour_fit_t = offsprings_fit + natoms;;
      fitness->evaluate(natoms, pop, pop_fit);

      Progress progress(1);
      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;

      //run BSA
      auto fitcmp = [&](int i, int j){return pop_fit[i] < pop_fit[j];};
      dbl_int gb_min;
      for (int iepoch = 0; iepoch < nepochs; ++iepoch) {
//always use the 1st old P. This is slightly different to the paper 
//where old P may not be used at all if not picked at 1st iteration
	if (iepoch == 0) {
	  fitness->generate(natoms, hist_pop); 
	}
	else{
          if (rand() < rand()) 
	    memcpy(hist_pop, pop, sizeof(T)*nparams*natoms);
	}
        std::shuffle(ipermv, ipermv+natoms, engine);

        for (int i = 0; i < natoms; ++i) isort[i] = i;
	std::sort(isort, isort+natoms, fitcmp);

        for (int i = 0; i < natoms; ++i){
          double prob = rand(), probj;
          int jmin = (1-prob)*natoms;
          int j = rand()*jmin;
          itheta[i] = isort[j];
        }

	get_ith popi(pop, nparams, 0);
	get_ith histpopi(hist_pop, nparams, ipermv);
	get_ith thetapopi(pop, nparams, itheta);
	get_ith offspringi(offsprings, nparams, 0);
        generate_set_based_atoms(natoms, nparams, ioper_t, lioper, ktour_fit_t, mx_ktour, c, seed, 
	                         iepoch, nthreads, popi, histpopi, thetapopi, offspringi, fitness); /**/
        fitness->evaluate(natoms, offsprings, offsprings_fit);

        mprint("pop_fit", natoms, 1, pop_fit);
        mprint("offspring_fit", natoms, 1, offsprings_fit);
        dbl_int epoch_min = BSAselect(nparams, natoms, pop, offsprings, pop_fit, offsprings_fit);
        mprint("new_fit", natoms, 1, pop_fit);

	cout << iepoch+1 << " " << epoch_min.val << endl;
        if (!iepoch || epoch_min.val < gb_min.val) {
          gb_min.val = epoch_min.val; 
          gb_min.idx = epoch_min.idx;
	  fitness->interpret(pop + nparams * epoch_min.idx);
	  cout << iepoch+1 << " " << epoch_min.val << endl;
        }
        elapsed = (std::chrono::system_clock::now() - start);
        progress.show(epoch_min.val, iepoch+1, nepochs, elapsed.count(), timeout);
	if (elapsed.count() > timeout || do_user_stop()) break;
      }
      memcpy(best, pop + nparams * gb_min.idx, sizeof(T)*nparams);

      //free arrays
      free(ipermv);
      free(pops);
      free(pop_fit);
      return gb_min.val;
    }
  protected:
    T* m_pop;
    int m_natoms, m_nepochs, m_seed, m_nthreads;
    double m_timeout;
    double m_c[5];
  };
//------------------------------------------------------------------------
//                             SBPSO
//------------------------------------------------------------------------
template<typename T>
class SBPSO{
  public:
    SBPSO(int natoms, int nepochs, double timeout, int nthreads, const double* c, int seed):
      m_natoms(natoms), m_nepochs(nepochs), m_timeout(timeout), m_nthreads(nthreads), 
      m_seed(seed), m_pop(0){
	for (int i = 0; i < 5; ++i) m_c[i] = c[i];
      }

    void startingPopulation(T* pop){m_pop = pop;}

    double operator()(
        GA_Fitness<T>* fitness, T* best)
    {
      return run(fitness, best, m_pop, m_nepochs, m_timeout, m_natoms, m_nthreads, m_c, m_seed);
    }

    static double run(
        GA_Fitness<T>* fitness, T* gbest, T* pop0,
	int nepochs, double timeout, int natoms, int mxthreads, double* c, int seed)
    {
      int nparams = fitness->nparams();
      int mx_ktour = (int)(c[4]*nparams);

      //generate populations
      int nthreads = std::min(mxthreads, natoms);
      int lioper = 5*nparams;
      int* ioper_t = (int*) malloc(sizeof(int)*lioper*nthreads);

      double* pop_fit = (double*) malloc(sizeof(double)*(2*natoms+mx_ktour*nthreads));
      double* cur_fit = pop_fit, *pbest_fit = cur_fit + natoms, *ktour_fit_t = pbest_fit + natoms;
      double gbest_fit = std::numeric_limits<double>::infinity();
      for (int i = 0; i < natoms; ++i) pbest_fit[i] = std::numeric_limits<double>::infinity();

      const int npop = 2;
      T* pops = (T*) malloc(sizeof(T)*npop*natoms*nparams);
      T* current = pops, *pbest = current + nparams*natoms;

      if (pop0){ memcpy(current, pop0, sizeof(T)*natoms*nparams); }
      else{ fitness->generate(natoms, current); }

      Progress progress(1);
      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;
      
      std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      vector<std::uint32_t> seeds(natoms);

      //run PSO
      for (int iepoch = 0; iepoch < nepochs; ++iepoch) {
        //calculate fitness 
        fitness->evaluate(natoms, current, pop_fit);
        double epoch_fit = PSOupdateBest(pop_fit, pbest_fit, gbest_fit, natoms, nparams, current, pbest, gbest);
	cout << iepoch+1 << " epoch_fit=" << epoch_fit << " gbest_fit=" << gbest_fit << endl;
        if (!iepoch || epoch_fit == gbest_fit) fitness->interpret(gbest);

        elapsed = (std::chrono::system_clock::now() - start);
        progress.show(epoch_fit, iepoch+1, nepochs, elapsed.count(), timeout);
	if (elapsed.count() > timeout || do_user_stop()) break;

	get_ith currenti(current, nparams, 0);
	get_ith pbesti(pbest, nparams, 0);
	get_0th gbesti(gbest);
        generate_set_based_atoms(natoms, nparams, ioper_t, lioper, ktour_fit_t, mx_ktour, c, seed, 
	                         iepoch, nthreads, currenti, pbesti, gbesti, currenti, fitness);                    
      }

      //free arrays
      free(ioper_t);
      free(pops);
      free(pop_fit);
      return gbest_fit;
    }
  private:
  protected:
    T* m_pop;
    int m_natoms, m_nepochs, m_nthreads, m_seed;
    double m_c[5], m_timeout;
  };

//------------------------------------------------------------------------
//                             BiPSO
//------------------------------------------------------------------------
template<typename T>
class BiPSO{
  public:
    BiPSO(int natoms, int nepochs, double timeout, int nthreads, const double* c, int seed):
      m_natoms(natoms), m_nepochs(nepochs), m_timeout(timeout), m_nthreads(nthreads), 
      m_seed(seed), m_pop(0){
	for (int i = 0; i < 3; ++i) m_c[i] = c[i];
      }

    void startingPopulation(T* pop){m_pop = pop;}

    double operator()(
        GA_Fitness<T>* fitness, T* best)
    {
      return run(fitness, best, m_pop, m_nepochs, m_timeout, m_natoms, m_nthreads, m_c, m_seed);
    }

    static double run(
        GA_Fitness<T>* fitness, T* gbest, T* pop0,
	int nepochs, double timeout, int natoms, int mxthreads, double* c, int seed)
    {
      int nparams = fitness->nparams();
      int nthreads = std::min(mxthreads, natoms);

      //generate populations
      double* pop_fit = (double*) malloc(sizeof(double)*(2*natoms + 2*natoms*nparams)); //current, best, velo0, velo1
      double* cur_fit = pop_fit, *pbest_fit = cur_fit + natoms;
      double* velo_0 = pbest_fit + natoms;
      double* velo_1 = velo_0 + natoms*nparams;
      double gbest_fit = std::numeric_limits<double>::infinity();
      for (int i = 0; i < natoms; ++i) pbest_fit[i] = std::numeric_limits<double>::infinity();

      const int npop = 2;
      T* pops = (T*) malloc(sizeof(T)*npop*natoms*nparams); //current, best
      T* current = pops, *pbest = current + nparams*natoms;

      if (pop0){ memcpy(current, pop0, sizeof(T)*natoms*nparams); }
      else{ fitness->generate(natoms, current); }
      for (int i = 0; i < nparams*natoms; ++i) velo_0[i] = velo_1[i] = 0.5; //initial probability of change is 50/50

      Progress progress(1);
      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;
      
      std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 engine(seed);
      auto rand = [&](){return zeroToOneUd(engine);};

      //run PSO
      for (int iepoch = 0; iepoch < nepochs; ++iepoch) {
        //calculate fitness 
        fitness->evaluate(natoms, current, pop_fit);
        double epoch_fit = PSOupdateBest(pop_fit, pbest_fit, gbest_fit, natoms, nparams, current, pbest, gbest);
	cout << iepoch+1 << " epoch_fit=" << epoch_fit << " gbest_fit=" << gbest_fit << endl;
        if (!iepoch || epoch_fit == gbest_fit) fitness->interpret(gbest);

        elapsed = (std::chrono::system_clock::now() - start);
        progress.show(epoch_fit, iepoch+1, nepochs, elapsed.count(), timeout);
	if (elapsed.count() > timeout || do_user_stop()) break;
        
	for (int iatom = 0; iatom < natoms; ++iatom){
	  T* pbesti = pbest + iatom * nparams;;
	  T* currenti = current + iatom*nparams;
	  double* velo_0i = velo_0 + iatom * nparams;
	  double* velo_1i = velo_1 + iatom * nparams;
	  for (int iparam = 0; iparam < nparams; ++iparam){
	    double dij11 = rand()*(pbesti[iparam] ? c[1] : -c[1]);
	    double dij12 = rand()*(gbest [iparam] ? c[2] : -c[2]);
            velo_0i[iparam] = c[0]*velo_0i[iparam] - dij11 - dij12;
            velo_1i[iparam] = c[0]*velo_1i[iparam] + dij11 - dij12;
	    velo_0i[iparam] = 1./(1.+exp(-velo_0i[iparam]));
	    velo_1i[iparam] = 1./(1.+exp(-velo_1i[iparam]));
	    double vij_c = currenti[iparam] ? velo_0i[iparam] : velo_1i[iparam];
	    if (rand() < vij_c) currenti[iparam] = !currenti[iparam];
	  }
	}
      }

      //free arrays
      free(pops);
      free(pop_fit);
      return gbest_fit;
    }
  private:
  protected:
    T* m_pop;
    int m_natoms, m_nepochs, m_nthreads, m_seed;
    double m_c[3], m_timeout;
  };
//------------------------------------------------------------------------
//                             BiMBSA
//------------------------------------------------------------------------
template<typename T>
class BiMBSA{
  public:
    BiMBSA(int natoms, int nepochs, double timeout, int nthreads, double mixrate, const double *c, int seed):
      m_natoms(natoms), m_nepochs(nepochs), m_timeout(timeout), m_nthreads(nthreads), 
      m_seed(seed), m_pop(0), m_mixrate(mixrate){
	for (int i = 0; i < 3; ++i) m_c[i] = c[i];
      }

    void startingPopulation(T* pop){m_pop = pop;}

    double operator()(
        GA_Fitness<T>* fitness, T* best)
    {
      return run(fitness, best, m_pop, m_nepochs, m_timeout, m_nthreads, m_natoms, m_mixrate, m_c, m_seed);
    }

    static double run(
        GA_Fitness<T>* fitness, T* best, T* pop0,
	int nepochs, double timeout, int mxthreads, int natoms, double mixrate, double* c, int seed)
    {
      if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::mt19937_64 engine(seed);

      int nparams = fitness->nparams();
      int nthreads = std::min(mxthreads, natoms);

      std::uniform_real_distribution<double> zeroToOneUd(0.0, 1.0);
      std::normal_distribution<double> zeroToOneNd(0.0, 1.0);
      auto rand = [&](){return zeroToOneUd(engine);};
      auto randn = [&](){return zeroToOneNd(engine);};

      //allocate permutation vector and wrk for crossover
      int* ipermv = (int*) malloc(sizeof(int)*(2*natoms+max(natoms, nparams))); //ipermv(natoms), itheta(natoms), iwrk(max(natoms, nparams))
      int* itheta = ipermv + natoms;
      int* iwrk = itheta + natoms;
      for (int i = 0; i < natoms; ++i) ipermv[i] = i;

      const int npop = 3;
      T* pops = (T*) malloc(sizeof(T)*npop*nparams*natoms); //current, historical, offspring
      T* pop = pops, *hist_pop = pop + nparams*natoms, *offsprings = hist_pop + nparams*natoms;
      if (pop0){ memcpy(pop, pop0, sizeof(T)*natoms*nparams); } 
      else{ fitness->generate(natoms, pop); }

      //calculate fitness 
      double* pop_fit = (double*) malloc(sizeof(double)*(2*natoms+4*natoms*nparams));
      double* offsprings_fit = pop_fit + natoms;
      double* velo_0 = offsprings_fit + natoms;
      double* velo_1 = velo_0 + natoms*nparams;
      double* offvelo_0 = velo_1 + natoms*nparams;
      double* offvelo_1 = velo_0 + natoms*nparams;
      fitness->evaluate(natoms, pop, pop_fit);
      for (int i = 0; i < nparams*natoms; ++i) velo_0[i] = velo_1[i] = 0.5; //initial probability of change is 50/50

      Progress progress(1);
      auto start = std::chrono::system_clock::now();
      std::chrono::duration<double> elapsed = start - start;

      //run BSA
      auto fitcmp = [&](int i, int j){return pop_fit[i] < pop_fit[j];};
      dbl_int gb_min;
      for (int iepoch = 0; iepoch < nepochs; ++iepoch) {
//always use the 1st old P. This is slightly different to the paper 
//where old P may not be used at all if not picked at 1st iteration
	if (iepoch == 0) {
	  fitness->generate(natoms, hist_pop); 
	}
	else{
          if (rand() < rand())
	    memcpy(hist_pop, pop, sizeof(T)*nparams*natoms);
	}
        std::shuffle(ipermv, ipermv+natoms, engine);

        for (int i = 0; i < natoms; ++i) iwrk[i] = i;
	std::sort(iwrk, iwrk+natoms, fitcmp);

        for (int i = 0; i < natoms; ++i){
          double prob = rand(), probj;
          int jmin = (1-prob)*natoms;
          int j = rand()*jmin;
          itheta[i] = iwrk[j];
        }

        double mr = (rand() < rand() ? mixrate : 0);
        mutate_crossover(nparams, natoms, pop, hist_pop, ipermv, offsprings, c, mr, iwrk, engine, itheta, 
	                 velo_0, velo_1, offvelo_0, offvelo_1);
        fitness->evaluate(natoms, offsprings, offsprings_fit);

        mprint("pop_fit", natoms, 1, pop_fit);
        mprint("offspring_fit", natoms, 1, offsprings_fit);
        dbl_int epoch_min = select(nparams, natoms, pop, offsprings, pop_fit, offsprings_fit,
	                              velo_0, velo_1, offvelo_0, offvelo_1);
        mprint("new_fit", natoms, 1, pop_fit);

	cout << iepoch+1 << " " << epoch_min.val << endl;
        if (!iepoch || epoch_min.val < gb_min.val) {
          gb_min.val = epoch_min.val; 
          gb_min.idx = epoch_min.idx;
	  fitness->interpret(pop + nparams * epoch_min.idx);
	  cout << iepoch+1 << " " << epoch_min.val << endl;
        }
        elapsed = (std::chrono::system_clock::now() - start);
        progress.show(epoch_min.val, iepoch+1, nepochs, elapsed.count(), timeout);
	if (elapsed.count() > timeout || do_user_stop()) break;
      }
      memcpy(best, pop + nparams * gb_min.idx, sizeof(T)*nparams);

      //free arrays
      free(ipermv);
      free(pops);
      free(pop_fit);
      return gb_min.val;
    }
  private:
  protected:
    T* m_pop;
    int m_natoms, m_nepochs, m_seed, m_nthreads;
    double m_timeout;
    double m_c[3], m_mixrate;

    static void mutate_crossover(int nparams, int natoms, 
                   T* pop, T* hist_pop, int* hist_perm, T* offspring, 
                   double* c, double mixrate, int* iwrk, std::mt19937_64& engine, int* itheta, 
		   double* velo_0, double* velo_1, double* offvelo_0, double* offvelo_1) 
    {
      std::uniform_real_distribution<double> zeroToOne(0.0, 1.0);
      auto rand  = [&]() { return zeroToOne(engine); };

      for (int i = 0; i < natoms; ++i){
        T* pop_i           = pop       + i           * nparams;
        T* hist_pop_i      = hist_pop  + hist_perm[i]* nparams;
        T* theta_pop_i     = pop       + itheta[i]   * nparams;
        T* offspring_i     = offspring + i           * nparams;
	double* velo_0i    = velo_0    + i           * nparams;
	double* velo_1i    = velo_1    + i           * nparams;
	double* offvelo_0i = offvelo_0 + i           * nparams;
	double* offvelo_1i = offvelo_1 + i           * nparams;

        for (int j = 0; j < nparams; ++j) iwrk[j] = j;
	int nmix = nparams;
	memcpy(offspring_i, pop_i, sizeof(T)*nparams);
        if (mixrate > 0.e0) {
          memcpy(offvelo_0i , velo_0i, sizeof(double)*nparams);
          memcpy(offvelo_1i , velo_1i, sizeof(double)*nparams);
          nmix = (int)(mixrate*rand()*nparams);
          shuffle(iwrk, iwrk+nparams, engine);
	}

        for (int imix = 0; imix < nmix; ++imix){
          int iparam = iwrk[imix];
	  double dij11 = rand()*(hist_pop_i[iparam]  ? c[1] : -c[1]);
	  double dij12 = rand()*(theta_pop_i[iparam] ? c[2] : -c[2]);
          offvelo_0i[iparam] = c[0]*velo_0i[iparam] - dij11 - dij12;
          offvelo_1i[iparam] = c[0]*velo_1i[iparam] + dij11 - dij12;
	  offvelo_0i[iparam] = 1./(1.+exp(-velo_0i[iparam]));
	  offvelo_1i[iparam] = 1./(1.+exp(-velo_1i[iparam]));
          assert(pop_i[iparam] == T(1) || pop_i[iparam] == T(0));
	  double vij_c = pop_i[iparam] ? offvelo_0i[iparam] : offvelo_1i[iparam];
	  if (rand() < vij_c) offspring_i[iparam] = !pop_i[iparam];
	  assert(offspring_i[iparam] == T(1) || offspring_i[iparam] == T(0));
        }
      }
    }

    static dbl_int select(
        int nparams, int natoms, T* pop, T* offsprings, double* pop_fit, double* offspring_fit,
  	double* velo_0, double* velo_1, double* offvelo_0, double* offvelo_1)
      {
        dbl_int gb = {0, -1};
        for (int i = 0; i < natoms; ++i){
          T* pop_i           = pop        + i*nparams;
          T* offspring_i     = offsprings + i*nparams;
	  double* velo_0i    = velo_0     + i*nparams;
	  double* velo_1i    = velo_1     + i*nparams;
	  double* offvelo_0i = offvelo_0  + i*nparams;
	  double* offvelo_1i = offvelo_1  + i*nparams;
          if (offspring_fit[i] < pop_fit[i]){
            memcpy(pop_i, offspring_i, sizeof(T)*nparams);
            pop_fit[i] = offspring_fit[i];
            memcpy(velo_0i, offvelo_0i, sizeof(double)*nparams);
            memcpy(velo_1i, offvelo_1i, sizeof(double)*nparams);
          }
          if (pop_fit[i] < gb.val || gb.idx == -1){
            gb.idx = i;
            gb.val = pop_fit[i];
          }
        }
        return gb;
      }
  };
