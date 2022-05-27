//------------------------------------------------------------------------
//                             TRAINCALLBACK
//------------------------------------------------------------------------
class Trainer;
class TrainCallback{
  public:
    virtual int operator()(Trainer* trainer){return 0;}
    virtual ~TrainCallback(){}
  private:
  protected:
  };
//------------------------------------------------------------------------
//                             TRAIN
//------------------------------------------------------------------------
class Trainer{
  public:
    Trainer(): 
      m_ncycles(0), m_thresh(0.e0), m_timeout(0), 
      m_network(0), m_loss(0), m_progress(0), m_log(0), m_callback(0){}
    Trainer(int ncycles, double thresh, double timeout):
      m_ncycles(ncycles), m_thresh(thresh), m_timeout(timeout),
      m_network(0), m_loss(0), m_progress(), m_log(0), m_callback(0){}
    virtual ~Trainer(){
      if (m_log) { fclose(m_log); }
    }
    virtual int lwrk(int nvec){return 0;}
    
    void log(FILE* ulog){ 
      if (m_log) { fclose(m_log); }
      m_log = ulog; 
    }
    void log(const char* filename){
      if (m_log) { fclose(m_log); }
      string logname(filename); logname += ".loss";

      m_log = fopen(logname.c_str(), "w"); assert(m_log);
      log_time_stamp(m_log);
    }
    int ncycles(){return m_ncycles;}
    double thresh(){return m_thresh;}
    double timeout(){return m_timeout;}
    double current_loss(){return m_cur_loss;}
    double current_cycle(){return m_cur_cycle;}
    Network* network(){return m_network;}
    Loss* loss(){return m_loss;}

    void callback(TrainCallback* callback){m_callback = callback;}

    void ncycles(int ncycles){m_ncycles = ncycles;}
    void thresh(double thresh){m_thresh = thresh;}
    void timeout(double timeout){m_timeout = timeout;}

    void network(Network* network){m_network = network;}
    void loss(Loss* loss){m_loss = loss;}
    void progress(Progress* progress){m_progress = progress;}

    virtual void InitTraining(int adapt){m_adapt = adapt;}
    virtual double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk) = 0;
    virtual void improved(int state){}
  private:
  protected:
    Network* m_network;
    Loss* m_loss;
    Progress* m_progress;
    FILE* m_log;
    TrainCallback* m_callback;

    int m_ncycles, m_min_j, m_adapt;
    double m_thresh, m_timeout, m_min_loss;

    void log_entry(int i, double loss1, double loss2) {
      assert(m_log);
      if (i == 1 || loss1 < m_min_loss){ //log only best
	m_min_loss = loss1;
	m_min_j = i;
      }
      fprintf(m_log, "%d %e %e %e %d\n", i, loss1, loss2, m_min_loss, m_min_j); fflush(m_log);
    }

    double m_cur_loss;
    double m_cur_cycle;
  };
//------------------------------------------------------------------------
//                        GRADIENT TRAINER
//------------------------------------------------------------------------
class GradientTrainer : public Trainer{
  public:
    GradientTrainer():m_rate(0){}
    GradientTrainer(int ncycles, double thresh, double timeout, double rate):
      Trainer(ncycles, thresh, timeout), m_rate(rate){}
    virtual ~GradientTrainer(){}

    double rate(){return m_rate;}
    virtual void rate(double rate){m_rate = rate;}

    virtual void CheckGradient(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk, double eps){};
  protected:
    double m_rate;
};
//------------------------------------------------------------------------
//                        GRADIENT DESCENT
//------------------------------------------------------------------------
class GD : public GradientTrainer{
  public:
    GD(){}
    virtual ~GD(){}
    GD(int ncycles, double thresh, double timeout, double rate):
      GradientTrainer(ncycles, thresh, timeout, rate),m_initrate(m_rate){}

    virtual int lwrk(int nvec) {
      return nvec * m_network->tOutput() + 2*m_network->mOutput() + m_network->nParams();
    } 
    double initial_rate(){return m_initrate;}
    void rate(double rate){m_initrate = m_rate = rate;}
    
    virtual void InitTraining(int adapt){m_adapt = adapt; m_better = 0;m_worse = 0;}
    virtual double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk);
    virtual void CheckGradient(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk, double eps);
    virtual void improved(int state){
      if (state > 0){
	++m_better;
	m_worse = 0;
	if (m_better == m_adjust){
	  m_rate = min(m_initrate*m_maxrate, m_rate*m_adjrate);
//	  sprint("+ rate ", m_rate, "\n");
	  m_better = 0;
	}
      }
      else if (state == 0){
	m_worse = 0;
      }
      else if (state < 0){
	m_better = 0;
	++m_worse;
	if (m_worse == m_adjust){
	  m_rate = max(m_initrate/m_minrate, m_rate/m_adjrate);
//	  sprint("- rate ", m_rate, "\n");
	  m_worse = 0;
	}
      }
    }
  protected:
    virtual void Update(int ncycles);

    double m_initrate;
    int m_better, m_worse;
    static constexpr double m_maxrate = 1;
    static constexpr double m_minrate = 100;
    static constexpr double m_adjrate = 1.2;
    static constexpr double m_adjust = 3;
  };

void GD::Update(
    int ncycles)
  {
    int nparams = m_network->nParams();
    double* params = m_network->getParams();
    double* gradient = m_network->getGradient();
    double scl = m_rate;

//    double dnrm_g = cblas_dnrm2(nparams, gradient, 1);
//    scl *= 1/dnrm_g;
    //sprint("dnrm_g=", dnrm_g, " scl=", scl, "\n");
    m_network->UpdateParameters(-scl);
  }

double GD::Train(
    int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk)
  { 
    int ldy = m_network->tOutput();
    double* g = wrk;
    double* y = g + m_network->nParams(); 
    double *dCdy = y + nvec * ldy;
    double *dCdx = dCdy + m_network->mOutput(); 

    double loss = 0;
    int ncycles = 0;
    auto start = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = start - start;
    
    m_loss->setTargets(ystar, ldystar);
    m_network->setGradient(g);
    m_network->setInputOutput(nvec, x, ldx, y, ldy);
    double* yn  = m_network->FeedForward();
    loss = m_loss->Evaluate(nvec, yn, ldy)/nvec;

    double best = std::numeric_limits<double>::max();
    double previous = std::numeric_limits<double>::max();

    while (loss > m_thresh && elapsed.count() < m_timeout 
	   && ncycles < m_ncycles
	   && !do_user_stop()){
      m_network->InitializeGradient();
      for (int i = 0; i < nvec; ++i){
        m_loss->Jacobian(yn+i*ldy, dCdy, i);
        m_network->PropagateBackward(i, dCdy, dCdx);
      }
      Update(ncycles);
      m_network->FeedForward();
      loss = m_loss->Evaluate(nvec, yn, ldy)/nvec;

      m_cur_loss = loss;
      m_cur_cycle = ncycles;
      
//    adaptive scheme       
      if (m_adapt){
        if (loss < best){ best = loss; improved(1); }
        else{
          if (loss < previous){ improved(0); }
          else{ improved(-1); }
        }
        previous = loss;
      }

      ++ncycles;
      elapsed = (std::chrono::system_clock::now() - start);
      if (m_progress) m_progress->show(loss, ncycles, m_ncycles, elapsed.count(), m_timeout);
      if (m_log) log_entry(ncycles, loss, loss/nvec);
      if (m_callback) (*m_callback)(this);
    }
    if (m_progress) m_progress->stop();
    return loss;
  }

void GD::CheckGradient(
    int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk, double eps)
  {
    Function::ClippingThreshold(0.);
    int ldy = m_network->tOutput();
    double* g = wrk;
    double* y = g + m_network->nParams(); 
    double *dCdy = y + nvec * ldy;
    double *dCdx = dCdy + m_network->mOutput(); 

    int nparams = m_network->nParams();
    double* params = m_network->getParams();

    m_loss->setTargets(ystar, ldystar);
    m_network->setGradient(g);
    m_network->setInputOutput(nvec, x, ldx, y, ldy);
    m_network->InitializeGradient();
 
    double* yn = m_network->FeedForward();
    double loss = m_loss->Evaluate(nvec, yn, ldy);
    for (int j = 0; j < nvec; ++j){
      m_loss->Jacobian(yn+j*ldy, dCdy, j);
      m_network->PropagateBackward(j, dCdy, dCdx);
    } 
    double dnrm_g0 = cblas_dnrm2(nparams, g, 1);
    mprint("g0", nparams, 1, g);

    double dw = eps;
    for (int j = 0; j < nparams; ++j){
      params[j] += dw;
      m_network->FeedForward();
      double loss_plus = m_loss->Evaluate(nvec, yn, ldy);

      /*params[j] -= 2*dw;
      m_network->FeedForward();
      double loss_minus = m_loss->Evaluate(nvec, yn, ldy);
      g[j] = (loss_plus-loss_minus)/(2*dw);
      params[j] += dw; **/


      g[j] = (loss_plus-loss)/dw;
      params[j] -= dw;
      // sprint("g[",j,"]=",g[j], " loss+=", loss_plus, " loss=", loss, "\n"); 
    }
    double dnrm_g = cblas_dnrm2(nparams, g, 1);
    mprint("g", nparams, 1, g);
    
    sprint("CheckGradient (eps=", eps, ")\n");
    sprint("||g0||=",dnrm_g0," ||g||=", dnrm_g, "\n");
    sprint("(||g||-||g0||)=",fabs(dnrm_g-dnrm_g0),"\n");
    sprint("(||g||-||g0||)/||g0||=",fabs((dnrm_g/dnrm_g0)-1),"\n");
    assert(fabs((dnrm_g/dnrm_g0)-1) < sqrt(eps));
  }
//------------------------------------------------------------------------
//                     MOMENTUM GRADIENT DESCENT
//------------------------------------------------------------------------
class MomentumGD : public GD{
  public:
    virtual ~MomentumGD(){}
    MomentumGD(int ncycles, double thresh, double timeout, double alpha, double beta):
      GD(ncycles, thresh, timeout, alpha),m_beta(beta){}
    virtual int lwrk(int nvec) { return GD::lwrk(nvec) + m_network->nParams(); } 
    virtual double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk){
      int nparams = m_network->nParams();
      m_moment = wrk;
      memset(m_moment, 0, sizeof(double)*nparams);
      return GD::Train(nvec, x, ldx, ystar, ldystar, wrk + nparams);
    }
    static void ModifyGradient(int nparams, double* gradient, double beta, double* moment){
      for (int i = 0; i < nparams; ++i){
	moment[i] = beta*moment[i] + (1.e0-beta)*gradient[i];
	gradient[i] = moment[i];
      }
    }
    virtual void ModifyGradient(){
      int nparams = m_network->nParams();
      double* gradient = m_network->getGradient();
      this->ModifyGradient(nparams, gradient, m_beta, m_moment);
    }
  private:
  protected:
    double m_beta;
    double* m_moment;
    
    virtual void Update(int ncycles){
      ModifyGradient();
      GD::Update(ncycles);
      //m_network->UpdateParameters(-m_rate);
    }
  };
//------------------------------------------------------------------------
//                     RMSP GRADIENT DESCENT
//------------------------------------------------------------------------
class RMSPGD : public MomentumGD{
  public:
    virtual ~RMSPGD(){}
    RMSPGD(int ncycles, double thresh, double timeout, double alpha, double beta):
      MomentumGD(ncycles, thresh, timeout, alpha, beta){}
    static void ModifyGradient(int nparams, double* gradient, double beta, double* moment){
      for (int i = 0; i < nparams; ++i){
	moment[i] = beta*moment[i] + (1.e0-beta)*(gradient[i]*gradient[i]);
	gradient[i] /= sqrt(moment[i]+m_eps);
      }
    }
    virtual void ModifyGradient(){
      int nparams = m_network->nParams();
      double* gradient = m_network->getGradient();
      this->ModifyGradient(nparams, gradient, m_beta, m_moment);
    }
  private:
  protected:
    static constexpr double m_eps = 1e-4;
  };
//------------------------------------------------------------------------
//                     ADAM GRADIENT DESCENT
//------------------------------------------------------------------------
class AdamGD : public GD{
  public:
    virtual ~AdamGD(){}
    AdamGD(int ncycles, double thresh, double timeout, double alpha, double beta1, double beta2):
      GD(ncycles, thresh, timeout, alpha),m_beta{beta1, beta2}{}
    virtual int lwrk(int nvec) { return GD::lwrk(nvec) + 2*m_network->nParams(); } 
    virtual double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk){
      int nparams = m_network->nParams();
      m_moment = wrk;
      memset(m_moment, 0, sizeof(double)*2*nparams);
      return GD::Train(nvec, x, ldx, ystar, ldystar, wrk + 2*nparams);
    }
    static void ModifyGradient(int nparams, double* gradient, 
	                       double beta1, double* moment1, double beta2, double* moment2, 
	                       int ncycles, double eps){
      double coeff1 = pow(beta1, ncycles+1.e0);
      double coeff2 = pow(beta2, ncycles+1.e0);
      for (int i = 0; i < nparams; ++i){
        moment1[i] = beta1*moment1[i] + (1.e0-beta1)*gradient[i];
	moment2[i] = beta2*moment2[i] + (1.e0-beta2)*gradient[i]*gradient[i];

	double m1hat = moment1[i]/(1.e0-coeff1);
	double m2hat = moment2[i]/(1.e0-coeff2);
	gradient[i] = m1hat/sqrt(m2hat+eps);
      }
    }
    virtual void ModifyGradient(int ncycles){
      int nparams = m_network->nParams();
      double* gradient = m_network->getGradient();
      ModifyGradient(nparams, gradient, m_beta[0], m_moment, m_beta[1], m_moment + nparams, ncycles, m_eps);
    }
  private:
  protected:
    double m_beta[2];
    double* m_moment;
    static constexpr double m_eps = 1e-4;
    
    virtual void Update(int ncycles){
      ModifyGradient(ncycles);
      GD::Update(ncycles);
      //m_network->UpdateParameters(-m_rate);
    }
  };
//------------------------------------------------------------------------
//                 STOCHASTIC GRADIENT DESCENT
//------------------------------------------------------------------------
class SGD : public GradientTrainer{
  public:
    SGD():m_batch(0), m_gd(0), m_engine(0){}
    SGD(int ncycles, double thresh, double timeout, double rate, int batch, 
	GradientTrainer *gd, mt19937_64* engine):
      GradientTrainer(ncycles, thresh, timeout, 0.0), m_gd(gd), m_batch(batch), m_engine(engine){
      m_network = m_gd->network(); 
      m_loss = m_gd->loss();
    }
    ~SGD(){}
    int lwrk(int nvec){
      int lwrk = m_batch*(m_network->nInput() + m_network->nOutput());
          lwrk += ((nvec+1)*sizeof(int))/sizeof(double);
          lwrk += m_gd->lwrk(m_batch);
      return lwrk;
    } 
    int batch(){return m_batch;}
    void batch(int batch){m_batch = batch;}
    void improved(int state){ m_gd->improved(state); }
    GradientTrainer* gd(){return m_gd;}

    void InitTraining(int adapt){ m_adapt = adapt; m_gd->InitTraining(0); }

    double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk);
    void CheckGradient(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk, double eps);
  private:
    int m_batch;
    GradientTrainer* m_gd;
    mt19937_64* m_engine;
  };

double SGD::Train(
    int nvec, double* x0, int ldx0, double* ystar0, int ldystar0, double* wrk0)
  {
    m_gd->ncycles(1); 
    m_gd->thresh(0); 
    m_gd->timeout(1); 

    int ldx = m_network->nInput();
    int ldystar = m_network->nOutput();

    double* x = wrk0;
    double* ystar = x + m_batch*ldx;
    int* permv = (int*)(ystar + m_batch*ldystar);
    double* wrk = (double*)permv + ((nvec+1)*sizeof(int))/sizeof(double);

    int ncycles = 0;
    auto start = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = start - start;

    for (int i = 0; i < nvec; ++i) permv[i] = i;
    
    double best = std::numeric_limits<double>::max();
    double previous = std::numeric_limits<double>::max();
    double loss = 0;

    int ivec = 0;
    shuffle(permv, permv+nvec, *m_engine);
    while (   elapsed.count() < m_timeout 
	   && ncycles < m_ncycles
	   && !do_user_stop()){
      int nbatch = std::min(m_batch, nvec - ivec);
      for (int i = 0; i < nbatch; ++i){
        int j = permv[ivec + i];
        memcpy(x + i*ldx, x0 + j*ldx0, sizeof(double)*ldx);
        memcpy(ystar + i*ldystar, ystar0 + j*ldystar0, sizeof(double)*ldystar);
      }
      loss += m_gd->Train(nbatch, x, ldx, ystar, ldystar, wrk);

      ivec += nbatch;
      elapsed = (std::chrono::system_clock::now() - start);
      if (ivec == nvec) {
  //    adaptive scheme 
        if (m_adapt){
          if (loss < best){ best = loss; m_gd->improved(1); }
          else{
  	    if (loss < previous){ m_gd->improved(0); }
  	    else{ m_gd->improved(-1); }
          } /**/
          previous = loss;
	}

	m_cur_loss = loss;
        m_cur_cycle = ncycles;

        ivec = 0;
        ++ncycles;
        shuffle(permv, permv+nvec, *m_engine);
        if (m_progress) m_progress->show(loss, ncycles, m_ncycles, elapsed.count(), m_timeout);
        if (m_log) log_entry(ncycles, loss, loss/nvec);
        if (m_callback) (*m_callback)(this);

	if (loss < m_thresh) break;
	loss = 0;
      }
    }
    if (m_progress) m_progress->stop();
    return loss;
  }

void SGD::CheckGradient(
    int nvec, double* x0, int ldx0, double* ystar0, int ldystar0, double* wrk0, double eps)
  {
    int ldx = m_network->nInput();
    int ldystar = m_network->nOutput();

    double* x = wrk0;
    double* ystar = x + m_batch*ldx;
    int* permv = (int*)(ystar + m_batch*ldystar);
    double* wrk = (double*)permv + ((nvec+1)*sizeof(int))/sizeof(double);

    for (int i = 0; i < nvec; ++i) permv[i] = i;
    shuffle(permv, permv+nvec, *m_engine);
    for (int ivec = 0; ivec < nvec; ){
       int nbatch = std::min(m_batch, nvec - ivec);
       for (int i = 0; i < nbatch; ++i){
	 int j = permv[ivec + i];
	 memcpy(x + i*ldx, x0 + j*ldx0, sizeof(double)*ldx);
	 memcpy(ystar + i*ldystar, ystar0 + j*ldystar0, sizeof(double)*ldystar);
       }
       m_gd->CheckGradient(nbatch, x, ldx, ystar, ldystar, wrk, eps);
       ivec += nbatch;
    }
  }
//------------------------------------------------------------------------
//                             TESTTRAINER
//------------------------------------------------------------------------
class TestTrainer : public Trainer{
  public:
    TestTrainer(int ncycles, double thresh, double timeout, Trainer* trainer):
      Trainer(ncycles, thresh, timeout), m_train(trainer){
      m_network = m_train->network(); 
      m_loss = m_train->loss();
    }
    void setTestData(int nvec, double* x, int ldx, double* ystar, int ldystar){
      m_nvec = nvec; m_x = x; m_ldx = ldx; m_ystar = ystar;m_ldystar = ldystar;
    }
    int lwrk(int nvec){
      int lwrk = m_nvec*m_network->tOutput();
      return max(m_train->lwrk(nvec), lwrk);
    }
    double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk);
    void InitTraining(int adapt){ m_adapt = adapt; m_train->InitTraining(0); }
  private:
    Trainer* m_train;
    int m_nvec;
    double* m_x, *m_ystar;
    int m_ldx, m_ldystar;
  };

double TestTrainer::Train(
    int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk)
  {
    m_train->ncycles(1); 
    m_train->thresh(0); 
    m_train->timeout(1); 

    double* y = wrk; int ldy = m_network->tOutput();
    int ncycles = 0;
    auto start = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = start-start;
    double loss = m_thresh + 1;
    double best = std::numeric_limits<double>::max();
    double previous = std::numeric_limits<double>::max();
    
    while (loss > m_thresh && elapsed.count() < m_timeout 
	   && ncycles < m_ncycles 
	   && !do_user_stop()){
      double train_loss = m_train->Train(nvec, x, ldx, ystar, ldystar, wrk);

      m_network->setInputOutput(m_nvec, m_x, m_ldx, y, ldy);
      m_loss->setTargets(m_ystar, m_ldystar);
      double* yn = m_network->FeedForward();
      loss = m_loss->Evaluate(m_nvec, yn, ldy)/m_nvec;

//    adaptive scheme       
      if (m_adapt){
        if (loss < best){ best = loss; m_train->improved(1); }
        else{
          if (loss < previous){ m_train->improved(0); }
          else{ m_train->improved(-1); }
        }
        previous = loss;
      }
      
      m_cur_loss = loss;
      m_cur_cycle = ncycles;

      ++ncycles;
      elapsed = (std::chrono::system_clock::now() - start);

      if (m_progress) m_progress->show(loss, ncycles, m_ncycles, elapsed.count(), m_timeout);
      if (m_log) log_entry(ncycles, loss, train_loss);
      if (m_callback) (*m_callback)(this);
    }
    if (m_progress) m_progress->stop();
    return loss;
  }

//------------------------------------------------------------------------
//                             SAVECALLBACK
//------------------------------------------------------------------------
class SaveBestCallback : public TrainCallback{
  public:
    SaveBestCallback(const char* prefix){
      m_fname = prefix; m_fname += ".parm";
      m_f = 0;
    }
    ~SaveBestCallback(){if (m_f) fclose(m_f);}
    int operator()(Trainer* trainer){
      if (trainer->current_cycle() < m_delay) return 0;

      if (trainer->current_cycle() == m_delay) {
        m_f = fopen(m_fname.c_str(), "w"); assert(m_f);
        m_best = std::numeric_limits<double>::max();
      }
      if (trainer->current_loss() < m_best){
	m_best = trainer->current_loss();
        Network* network = trainer->network();
	double* param = network->getParams();
	int nparam = network->nParams();
	fseek(m_f, 0, SEEK_SET);
	fwrite(param, sizeof(double),nparam, m_f);
      }
      return 0;
    }
  private:
  protected:
    string m_fname;
    FILE* m_f;
    double m_best;
    static constexpr int m_delay = 10;
  };












//------------------------------------------------------------------------
//                      GSL GRADIENT MINIMIZER
//------------------------------------------------------------------------
class GSL_Trainer : public GD{
  public:
    GSL_Trainer(){}
    virtual ~GSL_Trainer(){}
    GSL_Trainer(int ncycles, double thresh, double timeout, double rate):
      GD(ncycles, thresh, timeout, rate){}

    int lwrk(int nvec) {
      return nvec * m_network->tOutput() + 2*m_network->mOutput();
    } 
    double Train(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk);
    void CheckGradient(int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk){
      sprint("CheckGradient", " Not Supported\n");
    }
  protected:
    static double nn_f(const gsl_vector* v, void* params);
    static void nn_df(const gsl_vector* v, void* params, gsl_vector* df);
    static void nn_fdf(const gsl_vector* v, void* params, double* f, gsl_vector* df);

    double* m_wrk;
  };

double GSL_Trainer::nn_f(
    const gsl_vector* v, void* params)
  {
    GSL_Trainer* trainer = (GSL_Trainer*) params;
   
    assert(v->stride == 1);
    trainer->m_network->setParams(v->data);
    trainer->m_network->FeedForward();
    
    int nvec = trainer->m_network->nvec(), ldy;
    double* yn = trainer->m_network->getOutput(ldy);
    double loss = trainer->m_loss->Evaluate(nvec, yn, ldy)/nvec;
    //sprint("loss ", loss, "\n");
    return loss;
  }

void GSL_Trainer::nn_df(
    const gsl_vector* v, void* params, gsl_vector* df)
  {
    GSL_Trainer* trainer = (GSL_Trainer*) params;
    double* dCdy = trainer->m_wrk;
    double* dCdx = dCdy + trainer->m_network->mOutput(); 
    int nvec = trainer->m_network->nvec(), ldy;
    double* yn = trainer->m_network->getOutput(ldy);
    assert(df->stride == 1 && v->stride == 1);
    trainer->m_network->setGradient(df->data);
    trainer->m_network->setParams(v->data);
    trainer->m_network->InitializeGradient();
    for (int i = 0; i < nvec; ++i){
      trainer->m_loss->Jacobian(yn+i*ldy, dCdy, i);
      trainer->m_network->PropagateBackward(i, dCdy, dCdx);
    }
    //mprint("g", trainer->m_network->nParams(), 1, df->data);
  }

void GSL_Trainer::nn_fdf(
    const gsl_vector* v, void* params, double* f, gsl_vector* df)
  {
     *f = nn_f(v, params);
     nn_df(v, params, df);
  }

double GSL_Trainer::Train(
    int nvec, double* x, int ldx, double* ystar, int ldystar, double* wrk)
  { 
    int nparams = m_network->nParams();
    double* params = m_network->getParams();
    int ldy = m_network->tOutput();
    double* y = wrk; 
    m_wrk = y + nvec * ldy;
    m_loss->setTargets(ystar, ldystar);
    m_network->setInputOutput(nvec, x, ldx, y, ldy);

    const gsl_multimin_fdfminimizer_type *T = gsl_multimin_fdfminimizer_vector_bfgs2; 
    gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc (T, nparams);
    gsl_vector gsl_x{(size_t)nparams, 1, params, (gsl_block*)0 ,0};

    gsl_multimin_function_fdf nn;
    nn.n = nparams;
    nn.f = nn_f;
    nn.df = nn_df;
    nn.fdf = nn_fdf;
    nn.params = (void*)this;

    gsl_multimin_fdfminimizer_set(s, &nn, &gsl_x, m_rate, 0.1);
    
    double loss;
    int ncycles = 0, status;
    auto start = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = start - start;
    do{
      status = gsl_multimin_fdfminimizer_iterate (s);
      if (status) {
	cout << "GSL status " << status << endl;
	break;
      }
      gsl_vector* gsl_g = gsl_multimin_fdfminimizer_gradient(s);
      status = gsl_multimin_test_gradient(gsl_g, m_thresh);
      loss = gsl_multimin_fdfminimizer_minimum(s);
      
      ++ncycles;
      elapsed = (std::chrono::system_clock::now() - start);
      if (m_progress) m_progress->show(loss, ncycles, m_ncycles, elapsed.count(), m_timeout);
      if (m_log) log_entry(ncycles, loss, loss);
    }
    while (status == GSL_CONTINUE && elapsed.count() < m_timeout && ncycles < m_ncycles);
    if (m_progress) m_progress->stop();

    gsl_vector* gsl_xmin = gsl_multimin_fdfminimizer_x(s);
    cblas_dcopy(nparams, gsl_xmin->data, gsl_xmin->stride, params, 1);
    gsl_multimin_fdfminimizer_free (s);
    
    return loss;
  }
