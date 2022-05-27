//------------------------------------------------------------------------
//                             FUNCTION
//------------------------------------------------------------------------
class Function{
  public:
    Function(int id, int ninput, int noutput, bool ishead = false):
      m_ninput(ninput), m_noutput(noutput), 
      m_nparams(0), m_params(0), m_njac(0), m_jac(0), m_gradient(0), 
      m_toutput(noutput), m_moutput(noutput), m_ishead(ishead), m_id(id){}

    virtual ~Function(){}

    int nJacobian(){return m_njac;   } //size of jacobian
    int nParams()  {return m_nparams;} //number of parameters
    int nInput()   {return m_ninput; } //number of input
    int nOutput()  {return m_noutput;} //number of output
    int tOutput()  {return m_toutput;} //number of total output including intermediate functions
    int mOutput()  {return m_moutput;} //number of maximum output from all functions
    int tag()      {return m_id;}
    int nvec()     {return m_nvec;}

    double* getParams(){return m_params;}
    double* getJacobian(){return m_jac;}
    double* getGradient(){return m_gradient;}
    double* getOutput(int& ldy){return getOutput(&ldy);}
    double* getOutput(int* ldy = 0){if (ldy) *ldy = m_ldy; return m_y + (m_toutput - m_noutput);}
    double* getInput(int& ldx){ldx = m_ldx; return m_x;}

    virtual int setParams(double* params)    {m_params   = params;  return m_nparams;}
    virtual int setGradient(double* gradient){m_gradient = gradient;return m_nparams;}
    virtual int setJacobian(double* jac)     {m_jac      = jac;     return m_njac;   }
    virtual void setInputOutput(int nvec, double* x, int ldx, double* y, int ldy){
      m_nvec = nvec; m_x = x; m_ldx = ldx; m_y = y; m_ldy = ldy;
    }
    virtual void isHead(bool ishead){ m_ishead = ishead; }
    virtual bool isHead(){return m_ishead;}

    virtual double* FeedForward(){
      //mprint("FF:x0", m_ninput, m_nvec, m_x, m_ldx);
      this->feed_forward(m_nvec, m_x, m_ldx, m_y, m_ldy);
      //mprint("FF:y", m_noutput, m_nvec, m_y, m_ldy);
      return m_y + (m_toutput - m_noutput);
    }
    virtual double* PropagateBackward(int i, double* dCdy, double* dCdx, double scl = 0.e0){      
      double* x = m_x + i*m_ldx;   
      double* y = m_y + i*m_ldy;   
      //mprint("PB:x0", m_ninput, 1, x);
      this->gradient(x, y, dCdy);               //dC/dw = dC/dy * dy/dw
      //mprint("PB:x1", m_ninput, 1, x);
      if (!m_ishead) {
	this->cache_jacobian(x, y);
      //mprint("PB:x2", m_ninput, 1, x);
	this->propagate_backward(dCdy, dCdx, scl);  //dC/dx = dC/dy * J = dC/dy * dy/dx
      //mprint("PB:x3", m_ninput, 1, x);
      }
      //cout << "Gradient norm " << cblas_dnrm2(m_nparams, m_gradient, 1) << endl;
      if (m_clipping_thresh > 0){
        double gnrm = cblas_dnrm2(m_nparams, m_gradient, 1);
        if (gnrm > m_clipping_thresh) 
	  cblas_dscal(m_nparams, 1./gnrm, m_gradient, 1);
      }
      return m_gradient;
    }
    virtual void Initialize(){};
    virtual void InitializeGradient(){
      memset(m_gradient, 0, sizeof(double)*m_nparams);
    }
    virtual void UpdateParameters(double scl){
      cblas_daxpy(m_nparams, scl, m_gradient, 1, m_params, 1);
    }

    virtual void feed_forward(int nvec, double*x, int ldx, double*y, int ldy){};
    virtual void cache_jacobian (double* x, double* y){}             
    virtual void propagate_backward(double* dCdy, double* dCdx, double scl){};   
    virtual void gradient(double* x, double* y, double* dCdy){} 
    virtual void print(){cout << "Function(" << m_ninput << "," << m_noutput << ")";}
    static double ClippingThreshold(double clipping_threshold) {
      double old_thresh = m_clipping_thresh;
      m_clipping_thresh = clipping_threshold;
      return old_thresh;
    }
  protected:
    int m_ninput, m_noutput, m_toutput, m_moutput, m_id;
    int m_nparams, m_njac;
    double* m_params, *m_jac, *m_gradient;
    double* m_x, *m_y;
    int m_ldx, m_ldy, m_nvec;
    bool m_ishead;
    static double m_clipping_thresh;
  };

double Function::m_clipping_thresh = 1.e0;
//------------------------------------------------------------------------
//                             LINEAR
//------------------------------------------------------------------------
class Linear : public Function{
  public:
    Linear(int id, int ninput, int noutput, std::mt19937_64* engine, int nobias):
      Function(id, ninput, noutput), m_engine(engine),m_nobias(nobias) { }
  protected:
    std::mt19937_64* m_engine;
    int m_nobias;
  };
//------------------------------------------------------------------------
//                             FULLRANK
//------------------------------------------------------------------------
class Fullrank : public Linear{
  public:
    Fullrank(int id, int ninput, int noutput, std::mt19937_64* engine, int nobias = 0):
      Linear(id, ninput, noutput, engine, nobias) { 
	m_nparams = nParams(ninput, noutput); 
      }

    static int nParams(int ninput, int noutput){ return (ninput+1)*noutput;  }
    static void feed_forward(int nvec, double*x, int ldx, double*y, int ldy, int ninput, int noutput, double* params, int nobias);
    static void propagate_backward(double* dCdy, double* dCdx, int ninput, int noutput, double* params, double scl);
    static void gradient(double* x, double* y, double* dCdy, int ninput, int noutput, double* gradient, int nobias);

    void Initialize();
    void feed_forward(int nvec, double*x, int ldx, double*y, int ldy){
      feed_forward(nvec, x, ldx, y, ldy, m_ninput, m_noutput, m_params, m_nobias);
    }
    void propagate_backward(double* dCdy, double* dCdx, double scl){
      propagate_backward(dCdy, dCdx, m_ninput, m_noutput, m_params, scl);
    }
    void gradient(double* x, double* y, double* dCdy){
      gradient(x, y, dCdy, m_ninput, m_noutput, m_gradient, m_nobias);
    }
    void print(){
      cout << "Fullrank"; 
      if(m_nobias) cout << "[No Bias]"; 
      cout << "(" << m_ninput << "," << m_noutput << ")";
    }
  protected:
  };

void Fullrank::Initialize()
  {
    //double he = sqrt(6.e0/(m_ninput+m_noutput));
//    std::uniform_real_distribution<double> m121(-he, he);
    
    double xavier = sqrt(1.e0/m_ninput);
    std::normal_distribution<double> m121(0, xavier);
    auto randW = [&]() { return m121(*m_engine); };

    for (int i = 0; i < m_noutput; ++i){
      double* paramsi = m_params + i * (m_ninput+1);
      for (int j = 0; j < m_ninput; ++j) paramsi[j] = randW();
      if (!m_nobias) paramsi[m_ninput] = 0;
      else paramsi[m_ninput] = 0;//randW();
    }
    //mprint("In:Full:params", m_nparams, 1, m_params, 1);
  }

void Fullrank::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy, 
    int ninput, int noutput, double* params, int nobias)
  {
    double* weights = params;
    double* bias = params + ninput;
    
    if (nobias){
      for (int i = 0; i < noutput; ++i){ y[i] = 0; }
    }
    else{
      cblas_dcopy(noutput, bias, ninput+1, y, 1);   
    }
    for (int i = 1; i < nvec; ++i){
      double* yi = y + i*ldy;
      cblas_dcopy(noutput, y, 1, yi, 1);   
    }
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
	        noutput, nvec, ninput, //y = W^T * x + b
                1.e0, weights, ninput+1, x, ldx, 1.e0, y, ldy);
    //mprint("FF:Full:Xn", ninput, nvec, x, ldx);
    //mprint("FF:Full:Yn", noutput, nvec, y, ldy);
    //mprint("FF:Full:params", (ninput+1)*noutput, 1, params, 1);
  }

void Fullrank::propagate_backward(
    double* dCdy, double* dCdx, 
    int ninput, int noutput, double* params, double scl)
  {
    //if (scl != 0.e0) mprint("PB:Full:dCdx b4", ninput, 1, dCdx, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, ninput, noutput,                  //dC/dx = (dy/dx)^T dC/dy => y = W x
                1.e0, params, ninput+1, dCdy, 1, scl, dCdx, 1);
   // mprint("PB:Full:params", (ninput+1)*noutput, 1, params, 1);
   // mprint("PB:Full:dCdy", noutput, 1, dCdy, 1);
   // mprint("PB:Full:dCdx", ninput, 1, dCdx, 1);
  }

void Fullrank::gradient(
    double* x, double* y, double* dCdy, 
    int ninput, int noutput, double* gradient, int nobias)
  {
    //mprint("G:Full:x", ninput, 1, x, 1);
    //mprint("G:Full:dCdy", noutput, 1, dCdy, 1);
    //mprint("G:Full:G (b4)", noutput*(ninput+1), 1, gradient, 1);
    for (int i = 0; i < noutput; ++i){
      double* dCdw = gradient + i*(ninput+1);
      cblas_daxpy(ninput, dCdy[i], x, 1, dCdw, 1);                               //dC/dw = (dy/dw)^T * dC/dy
      if (!nobias) dCdw[ninput] += dCdy[i];   
    }
    //mprint("G:Full:G (af)", noutput*(ninput+1), 1, gradient, 1);
  }
//------------------------------------------------------------------------
//                            LOWRANK
//------------------------------------------------------------------------
class Lowrank : public Linear{
  public:
    Lowrank(int id, int ninput, int noutput, int urank, std::mt19937_64* engine, int nobias = 0):
      Linear(id, ninput, noutput, engine, nobias) { 
      m_rank = rank(ninput, noutput, urank);
      m_nparams = nParams(ninput, noutput, m_rank); 
      m_toutput = tOutput(noutput, m_rank);
      m_njac = m_rank;
    }
    static int rank(int ninput, int noutput, int rank){ return min(min(noutput, ninput), rank); }
    static int nParams(int ninput, int noutput, int rank){ return rank*(ninput+noutput) + noutput; }
    static int tOutput(int noutput, int rank){ return rank + noutput; }
    static void feed_forward(int nvec, double*x, int ldx, double*y, int ldy, int ninput, int noutput, int rank, double* params);
    static void propagate_backward(double* dCdy, double* dCdx, int ninput, int noutput, int rank, double* params, double* jac, double scl);
    static void gradient(double* x, double* y, double* dCdy, int ninput, int noutput, int rank, double* params, double* gradient, int nobias);

    void Initialize();
    void feed_forward(int nvec, double*x, int ldx, double*y, int ldy){
      feed_forward(nvec, x, ldx, y, ldy, m_ninput, m_noutput, m_rank, m_params);
    }
    void propagate_backward(double* dCdy, double* dCdx, double scl){
      propagate_backward(dCdy, dCdx, m_ninput, m_noutput, m_rank, m_params, m_jac, scl);
    }
    void gradient(double* x, double* y, double* dCdy){
      gradient(x, y, dCdy, m_ninput, m_noutput, m_rank, m_params, m_gradient, m_nobias);
    }
    void print(){
      cout << "Lowrank[" << m_rank << "]";
      if(m_nobias) cout << "[No Bias]"; 
      cout << "(" << m_ninput << "," << m_noutput << ")";
    }
  protected:
    int m_rank;
  };

void Lowrank::Initialize()
  {
//    double he = sqrt(6.e0/(m_ninput+m_noutput));
//    std::uniform_real_distribution<double> m121(-he, he);

    double xavier = sqrt(1.e0/m_ninput);
    std::normal_distribution<double> m121(0, xavier);

    auto randW = [&]() { return m121(*m_engine); };

    double* V = m_params; 
    double* U = V + m_rank*m_ninput;
    double* b = U + m_rank*m_noutput;
    for (int i = 0; i < m_rank; ++i){
      double* Vi = V + i*m_ninput;
      double* Ui = U + i*m_noutput;
      for (int j = 0; j < m_ninput; ++j) {
	double w = randW(); 
	Vi[j] = w > 0 ? sqrt(w) : -sqrt(-w);
      }
      for (int j = 0; j < m_noutput; ++j) {
	double w = randW(); 
       	Ui[j] = w > 0 ? sqrt(w) : -sqrt(-w);
      }
    }
    if (m_nobias)
      for (int j = 0; j < m_noutput; ++j) b[j] = 0;
    else
      for (int j = 0; j < m_noutput; ++j) b[j] = 0;
  }

void Lowrank::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy, 
    int ninput, int noutput, int rank, double* params)
  {
    double* weights = params;
    double* bias = params + ninput;
    
    double* V = params; 
    double* U = V + rank*ninput;
    double* b = U + rank*noutput;
    
    //mprint("V", ninput, rank, V);
    //mprint("U0", noutput, rank, U);
    //mprint("b0", noutput, 1, b);
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, rank, nvec, ninput, 
                1.e0, V, ninput, x, ldx, 0.e0, y, ldy);
    //mprint("y10", rank, nvec, y, ldy);

    for (int i = 0; i < nvec; ++i){
      double* yi = y + i*ldy + rank;
      cblas_dcopy(noutput, b, 1, yi, 1);   
    }
    //mprint("y11", rank, nvec, y, ldy);
    //mprint("U1", noutput, rank, U);
    //mprint("y2", noutput, nvec, y+rank, ldy);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, noutput, nvec, rank, 
                1.e0, U, noutput, y, ldy, 1.e0, y+rank, ldy);
    //mprint("y3", noutput, nvec, y+rank, ldy);
  }

void Lowrank::propagate_backward(
    double* dCdy, double* dCdx, 
    int ninput, int noutput, int rank, double* params, double* jac, double scl)
  {
    double* V = params;
    double* U = V + rank*ninput;
    
    cblas_dgemv(CblasColMajor, CblasTrans, noutput, rank,
                1.e0, U, noutput, dCdy, 1, 0.e0, jac, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, ninput, rank,       
                1.e0, V, ninput, jac, 1, scl, dCdx, 1);
  }

void Lowrank::gradient(
    double* x, double* y, double* dCdy, 
    int ninput, int noutput, int rank, double* params, double* gradient, int nobias)
  {
    double* V = params; 
    double* U = V + rank*ninput;
    double* b = U + rank*noutput;

    double* dCdv = gradient; 
    double* dCdu = dCdv + rank*ninput;
    double* dCdb = dCdu + rank*noutput;

    for (int i = 0; i < noutput; ++i){
      for (int j = 0; j < rank; ++j){
        double* dCdvi = dCdv + j*ninput;
	double uij = U[j*noutput + i];
        cblas_daxpy(ninput, dCdy[i]*uij, x, 1, dCdvi, 1);        
      }
    }

    for (int i = 0; i < rank; ++i){
      double* dCdui = dCdu + i*noutput;
      cblas_daxpy(noutput, y[i], dCdy, 1, dCdui, 1);        
    }
    if (!nobias) cblas_daxpy(noutput, 1.e0, dCdy, 1, dCdb, 1);        
  }
//------------------------------------------------------------------------
//                             RBF
//------------------------------------------------------------------------
class RBF : public Function{
  public:
    RBF(int id, int ninput, int noutput, std::mt19937_64* engine):
      Function(id, ninput, noutput), m_engine(engine) { }
  protected:
    std::mt19937_64* m_engine;
  };
//------------------------------------------------------------------------
//                             FULLRANK
//------------------------------------------------------------------------
class GaussRBF : public RBF{
  public:
    GaussRBF(int id, int ninput, int noutput, std::mt19937_64* engine):
      RBF(id, ninput, noutput, engine) { 
	m_nparams = nParams(ninput, noutput); 
        m_njac = ninput*noutput;
      }

    static int nParams(int ninput, int noutput){ return (ninput+1)*noutput;  }
    static void feed_forward(int nvec, double*x, int ldx, double*y, int ldy, int ninput, int noutput, double* params);
    static void propagate_backward(double* dCdy, double* dCdx, int ninput, int noutput, double* params, double scl);
    static void gradient(double* x, double* y, double* dCdy, int ninput, int noutput, double* gradient, double* params);
    static void cache_jacobian( double*x, double*y, double* jac, int ninput, int noutput, double* params);

    void Initialize();
    void feed_forward(int nvec, double*x, int ldx, double*y, int ldy){
      feed_forward(nvec, x, ldx, y, ldy, m_ninput, m_noutput, m_params);
    }
    void cache_jacobian (double* x, double* y){
      cache_jacobian(x, y, m_jac, m_ninput, m_noutput, m_params);
    }
    void propagate_backward(double* dCdy, double* dCdx, double scl){
      propagate_backward(dCdy, dCdx, m_ninput, m_noutput, m_jac, scl);
    }
    void gradient(double* x, double* y, double* dCdy){
      gradient(x, y, dCdy, m_ninput, m_noutput, m_gradient, m_params);
    }
    void print(){cout << "GaussRBF(" << m_ninput << "," << m_noutput << ")";}
  protected:
  };

void GaussRBF::Initialize()
  {
    std::uniform_real_distribution<double> zeroOneNd(0.0, 1.0);
    auto rand = [&]() { return zeroOneNd(*m_engine); };

    for (int i = 0; i < m_noutput; ++i){
      double* centeri = m_params + i * (m_ninput+1);
      for (int j = 0; j < m_ninput; ++j) centeri[j] = rand();
      centeri[m_ninput] = 0.25;
    }
    //mprint("In:GRBF:params", m_nparams, 1, m_params, 1);
  }

void GaussRBF::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy, 
    int ninput, int noutput, double* params)
  {
    for (int i = 0; i < nvec; ++i){
      double* yi = y + i*ldy;
      double* xi = x + i*ldx;
      for (int j = 0; j < noutput; ++j){
	double* centerj = params+j*(ninput+1);
	double d = 0;
	for(int k = 0; k < ninput; ++k) d += pow(xi[k] - centerj[k],2);
	yi[j] = exp(-d/fabs(centerj[ninput]));
      }
    }
    //mprint("GRBF:Full:Xn", ninput, nvec, x, ldx);
    //mprint("GRBF:Full:Yn", noutput, nvec, y, ldy);
    //mprint("GRBF:Full:params", (ninput+1)*noutput, 1, params, 1);
  }

void GaussRBF::cache_jacobian(
    double*x, double*y, double* jac, 
    int ninput, int noutput, double* params)
  {
    for (int i = 0; i < noutput; ++i){
      double* jaci = jac + i*ninput;
      double* centeri = params + i*(ninput+1);
      for (int j = 0; j < ninput; ++j) {
	jaci[j] = -2*y[j]*(x[j]-centeri[j])/fabs(centeri[ninput]);
      }
    }
  }

void GaussRBF::propagate_backward(
    double* dCdy, double* dCdx, 
    int ninput, int noutput, double* jac, double scl)
  {
    //if (scl != 0.e0) mprint("PB:GRBF:dCdx b4", ninput, 1, dCdx, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, ninput, noutput,                  //dC/dx = (dy/dx)^T dC/dy => y = W x
                1.e0, jac, ninput, dCdy, 1, scl, dCdx, 1);
    //mprint("PB:GRBF:params", (ninput+1)*noutput, 1, params, 1);
    //mprint("PB:GRBF:dCdy", noutput, 1, dCdy, 1);
    //mprint("PB:GRBF:dCdx", ninput, 1, dCdx, 1);
  }

void GaussRBF::gradient(
    double* x, double* y, double* dCdy, 
    int ninput, int noutput, double* gradient, double* params)
  {
    for (int i = 0; i < noutput; ++i){
      double* dCdw = gradient + i*(ninput+1);
      double* centeri = params+i*(ninput+1);
      double sigma = centeri[ninput];
      double d = 0;
      for(int k = 0; k < ninput; ++k) {
	double r = x[k] - centeri[k];
	dCdw[k] += dCdy[i]*2*y[i]*r/fabs(sigma);
	d += r*r;
      }
      dCdw[ninput] += dCdy[i]*y[i]*d*sigma/pow(fabs(sigma),3);   
    }
    //mprint("PB:GRBF:G", noutput*(ninput+1), 1, gradient, 1);
  }
//------------------------------------------------------------------------
//                             SIGMOID
//------------------------------------------------------------------------
class Sigmoid : public Function{
  public:
    Sigmoid(int id, int ninput, int adapt = 0): Function(id, ninput, ninput), m_adapt(adapt) {
      m_njac = ninput;
      if (m_adapt) m_nparams = ninput;
    }
    void Initialize() {for(int i = 0; i < m_nparams; ++i) m_params[i] = 1;}

    virtual void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    virtual void cache_jacobian (double* x, double* y);
    virtual void propagate_backward(double* dCdy, double* dCdx, double scl);   
    virtual void print(){if (m_adapt) cout << "Adaptive "; cout << "Sigmoid(" << m_noutput << ")";}
    virtual void gradient(double* x, double* y, double* dCdy);
  protected:
    int m_adapt;
  };

void Sigmoid::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy)
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      for (int j = 0; j < m_ninput; ++j){
	double xij = xi[j]*(m_adapt ? abs(m_params[j]) : 1);
	yi[j] = 1.e0/(1.e0+exp(-xij));
      }
    }
  }

void Sigmoid::cache_jacobian(
    double* x, double* y)
  {
    for (int i = 0; i < m_ninput; ++i){
      m_jac[i] = y[i]*(1-y[i])*(m_adapt ? abs(m_params[i]) : 1);
    }
  }

void Sigmoid::propagate_backward(
    double* dCdy, double* dCdx, double scl)
  {
    if(scl == 0){
      for (int i = 0; i < m_ninput; ++i){ dCdx[i] = m_jac[i]*dCdy[i]; }
    }
    else{
      if (scl == 1)
        for (int i = 0; i < m_ninput; ++i){ dCdx[i] += m_jac[i]*dCdy[i]; }
      else
        for (int i = 0; i < m_ninput; ++i){ dCdx[i] = scl*dCdx[i] + m_jac[i]*dCdy[i]; }
    }
  }

void Sigmoid::gradient(
    double* x, double* y, double* dCdy)
  {
    if (!m_adapt) return;

    for (int i = 0; i < m_ninput; ++i){
      double g = y[i]*(1-y[i])*x[i]*dCdy[i];
      m_gradient[i] += (m_params[i] > 0 ? g : -g);
    }
  }
//------------------------------------------------------------------------
//                             LOGSIGMOID
//------------------------------------------------------------------------
class LogSigmoid : public Sigmoid{
  public:
    LogSigmoid(int id, int ninput, int adapt): Sigmoid(id, ninput, adapt){}

    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian (double* x, double* y);
    void print(){if (m_adapt) cout << "Adaptive "; cout << "LogSigmoid(" << m_noutput << ")";}
    void gradient(double* x, double* y, double* dCdy);
  private:
  };

void LogSigmoid::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy)
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      for (int j = 0; j < m_ninput; ++j){
	double xij = xi[j]*(m_adapt ? abs(m_params[j]) : 1);
	yi[j] = -log(1.e0+exp(-xij));
      }
    }
  }

void LogSigmoid::cache_jacobian(
    double* x, double* y)
  {
    for (int i = 0; i < m_ninput; ++i){
      m_jac[i] = (1-exp(y[i]))*(m_adapt ? abs(m_params[i]) : 1);
    }
  }

void LogSigmoid::gradient(
    double* x, double* y, double* dCdy)
  {
    if (!m_adapt) return;

    for (int i = 0; i < m_ninput; ++i){
      double g = (1-exp(y[i]))*x[i]*dCdy[i];
      m_gradient[i] += (m_params[i] > 0 ? g : -g);
    }
  }
//------------------------------------------------------------------------
//                             LU
//------------------------------------------------------------------------
class LU : public Function{
  public:
    LU(int id, int ninput): Function(id, ninput, ninput){
      m_njac = ninput;
    }
    virtual ~LU(){} 
    void propagate_backward(double* dCdy, double* dCdx, double scl){
      propagate_backward(m_ninput, m_jac, dCdy, dCdx, scl);
    }
    static void propagate_backward(int input, double* jac, double* dCdy, double* dCdx, double scl = 0.e0);   
  protected:
    double m_leak;
  };

void LU::propagate_backward(
    int ninput, double* jac, double* dCdy, double* dCdx, double scl)
  {
    if(scl == 0){
      for (int i = 0; i < ninput; ++i){ dCdx[i] = jac[i]*dCdy[i]; }
    }
    else{
      if (scl == 1)
        for (int i = 0; i < ninput; ++i){ dCdx[i] += jac[i]*dCdy[i]; }
      else
        for (int i = 0; i < ninput; ++i){ dCdx[i] = scl*dCdx[i] + jac[i]*dCdy[i]; }
    }
  }

//------------------------------------------------------------------------
//                             ReLU
//------------------------------------------------------------------------
class ReLU : public LU{
  public:
    ReLU(int id, int ninput, double leak, int adapt = 0): 
      LU(id, ninput), m_leak(leak), m_adapt(adapt){
        if (m_adapt) m_nparams = ninput;
      }

    void Initialize() {for(int i = 0; i < m_nparams; ++i) m_params[i] = 1.e0;}
    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian (double* x, double* y);
    void print(){if (m_adapt) cout << "Adaptive "; cout << "ReLu[" << m_leak << "](" << m_noutput << ")";}
    void gradient(double* x, double* y, double* dCdy);
  protected:
    double m_leak;
    int m_adapt;
  };

void ReLU::feed_forward(
    int nvec, double*x, int ldx, double* y, int ldy) 
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      for (int j = 0; j < m_ninput; ++j){
	double alpha = (m_adapt ? abs(m_params[j]) : 1);
	yi[j] = (xi[j] < 0 ? m_leak*alpha*xi[j] : alpha*xi[j]);
      }
    }
  }

void ReLU::cache_jacobian(
    double* x, double* y)
  {
    for (int i = 0; i < m_ninput; ++i){
      double alpha = m_adapt ? abs(m_params[i]) : 1;
      m_jac[i] = (y[i] < 0 ? m_leak*alpha : alpha);
    }
  }

void ReLU::gradient(
    double* x, double* y, double* dCdy)
  {
    if (!m_adapt) return;

    for (int i = 0; i < m_ninput; ++i){
      double g = (x[i] < 0 ? m_leak*x[i] : x[i])*dCdy[i];
      m_gradient[i] += (m_params[i] > 0 ? g : -g);
    }
  }
//------------------------------------------------------------------------
//                             SeLU
//------------------------------------------------------------------------
class SeLU : public LU{
  public:
    SeLU(int id, int ninput): LU(id, ninput){ }

    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian (double* x, double* y);
    void print(){cout << "SeLu(" << m_noutput << ")";}
  protected:
    static constexpr double m_alpha = 1.6732632423543772848170429916717;
    static constexpr double m_lamda = 1.0507009873554804934193349852946;
  };

void SeLU::feed_forward(
    int nvec, double*x, int ldx, double* y, int ldy) 
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      for (int j = 0; j < m_ninput; ++j){
	yi[j] = m_lamda * (xi[j] < 0 ? m_alpha*(exp(xi[j])-1.e0) : xi[j]);
      }
    }
  }

void SeLU::cache_jacobian(
    double* x, double* y)
  {
    const double la = m_lamda*m_alpha;
    for (int i = 0; i < m_ninput; ++i){
      m_jac[i] = y[i] < 0 ?  y[i] + la : m_lamda;
    }
  }
//------------------------------------------------------------------------
//                             SOFTMAX
//------------------------------------------------------------------------
class Softmax : public Function{
  public:
    Softmax(int id, int ninput): Function(id, ninput, ninput){
      m_njac = (ninput*(ninput+1))/2;
    }

    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian(double* x, double* y);
    void propagate_backward(double* dCdy, double* dCdx, double scl);   
    void print(){cout << "Softmax(" << m_noutput << ")";}
  private:
  };

void Softmax::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy)
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      double sum = 0, mx = xi[0];
      for (int j = 1; j < m_ninput; ++j) mx = max(mx, xi[j]); 
      for (int j = 0; j < m_ninput; ++j) sum += exp(xi[j]-mx); 
      for (int j = 0; j < m_ninput; ++j) yi[j] = exp(xi[j]-mx)/sum;
    }
    //mprint("FF:Soft:Yn", m_noutput, nvec, y, ldy);
  }

void Softmax::cache_jacobian(
    double* x, double* y)
  {
    //mprint("CJ:Soft:y", m_ninput, 1, y, 1);
    double* jaci = m_jac;
    for (int i = 0; i < m_ninput; ++i){
      for (int j = 0; j < i; ++j) {
        jaci[j] = -y[i]*y[j];  
      }
      jaci[i] = y[i]*(1-y[i]); 
      jaci += i+1;
    }
  }

void Softmax::propagate_backward(
    double* dCdy, double* dCdx, double scl)
  {
    cblas_dspmv(CblasColMajor, CblasUpper, m_ninput, 
	        1.e0, m_jac, dCdy, 1, scl, dCdx, 1);
  }
//------------------------------------------------------------------------
//                             LOGSOFTMAX
//------------------------------------------------------------------------
class LogSoftmax : public Function{
  public:
    LogSoftmax(int id, int ninput): Function(id, ninput, ninput){ 
      m_njac = ninput*ninput;
    }

    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian(double* x, double* y);
    void propagate_backward(double* dCdy, double* dCdx, double scl);   
    void print(){cout << "LogSoftmax(" << m_noutput << ")";}
  private:
  };

void LogSoftmax::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy)
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      double sum = 0, mx = xi[0];
      for (int j = 1; j < m_ninput; ++j) mx = max(mx, xi[j]); 
      for (int j = 0; j < m_ninput; ++j) sum += exp(xi[j]-mx); 
      for (int j = 0; j < m_ninput; ++j) {
	yi[j] = xi[j] - mx - log(sum);
	if (isinf(yi[j])) {
	  cout << xi[j] << " " << mx << " " << sum << endl;
          mprint("xi", m_ninput, 1, xi, ldx);
	}
      }
    } /**/
    //mprint("FF:LogSoft:Yn2", m_noutput, nvec, y, ldy);
  }

void LogSoftmax::cache_jacobian(
    double* x, double* y)
  {
    for (int i = 0; i < m_ninput; ++i){
      double* jaci = m_jac + i* m_ninput;
      for (int j = 0; j < i; ++j) {
        jaci[j] = -exp(y[j]);   
      }
      jaci[i] = 1-exp(y[i]);  
      for (int j = i+1; j < m_ninput; ++j) {
        jaci[j] = -exp(y[j]);   
      }
    } /**/
    //mprint("CJ:LogSoft:jac", m_ninput*(m_ninput+1)/2, 1, m_jac, 1);
  }

void LogSoftmax::propagate_backward(
    double* dCdy, double* dCdx, double scl)
  {
    cblas_dgemv(CblasColMajor, CblasNoTrans, m_ninput, m_ninput, 
	        1.e0, m_jac, m_ninput, dCdy, 1, scl, dCdx, 1);
  }
//------------------------------------------------------------------------
//                             HERMITE
//------------------------------------------------------------------------
class Hermite : public Function{
  public:
    Hermite(int id, int ninput, int degree, std::mt19937_64* engine): 
      Function(id, ninput, ninput), m_degree(degree), m_engine(engine){
      m_njac = ninput;
      m_nparams = ninput*(degree+1);
    }

    void Initialize();
    void feed_forward(int nvec, double*x, int ldx, double* y, int ldy);
    void cache_jacobian(double* x, double* y);
    void propagate_backward(double* dCdy, double* dCdx, double scl);   
    void gradient(double* x, double* y, double* dCdy);                
    void print(){cout << "Hermite[" << m_degree << "](" << m_noutput << ")";}
  private:
    int m_degree;
    std::mt19937_64* m_engine;
  };

void Hermite::Initialize()
  {
    double xavier = sqrt(1.e0/m_ninput);
    double he = sqrt(6.e0/(m_ninput+m_noutput));
    std::uniform_real_distribution<double> m121(-he, he);
    auto randW = [&]() { return m121(*m_engine); };

    for (int i = 0; i < m_noutput; ++i){
      double* paramsi = m_params + i * (m_degree+1);
      for (int j = 0; j <= m_degree; ++j) paramsi[j] = randW();
    }
  }

void Hermite::feed_forward(
    int nvec, double*x, int ldx, double*y, int ldy)
  {
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i * ldx;
      double* yi = y + i * ldy;
      for (int j = 0; j < m_ninput; ++j){
	double* betaj = m_params + j * (m_degree+1);
	double hnm2 = 1;
	double hnm1 = 2*xi[j];
	yi[j] =  betaj[0]*hnm2;
	yi[j] += betaj[1]*hnm1;
        for (int k = 2; k <= m_degree; ++k){
          double hn = 2*xi[j]*hnm1-2*(k-1)*hnm2;
	  yi[j] += betaj[k]*hn;
	  hnm2 = hnm1;
	  hnm1 = hn;
	}
      }
    }
    //mprint("FF:Herm:Yn", m_noutput, nvec, y, ldy);
  }

void Hermite::cache_jacobian(
    double* x, double* y)
  {
    for (int j = 0; j < m_ninput; ++j){
      double* betaj = m_params + j * (m_degree+1);
      double hnm2_dot = 0, hnm2 = 1;
      double hnm1_dot = 2, hnm1 = 2*x[j];
      m_jac[j]  = betaj[0]*hnm2_dot;
      m_jac[j] += betaj[1]*hnm1_dot;
      for (int k = 2; k <= m_degree; ++k){
        double hn = 2*x[j]*hnm1-2*(k-1)*hnm2;
        double hn_dot = 2*hnm1+2*x[j]*hnm1_dot-2*(k-1)*hnm2_dot;
        m_jac[j] += betaj[k]*hn_dot;
        hnm2_dot = hnm1_dot; hnm2 = hnm1;
        hnm1_dot = hn_dot;   hnm1 = hn;
      }
    }
  }

void Hermite::propagate_backward(
    double* dCdy, double* dCdx, double scl)
  {
    if(scl == 0){
      for (int i = 0; i < m_ninput; ++i){ dCdx[i] = m_jac[i]*dCdy[i]; }
    }
    else{
      if (scl == 1)
        for (int i = 0; i < m_ninput; ++i){ dCdx[i] += m_jac[i]*dCdy[i]; }
      else
        for (int i = 0; i < m_ninput; ++i){ dCdx[i] = scl*dCdx[i] + m_jac[i]*dCdy[i]; }
    }
  }

void Hermite::gradient(
    double* x, double* y, double* dCdy)
  {
    for (int j = 0; j < m_ninput; ++j){
      double* dCdw = m_gradient + j*(m_degree+1);
      double hnm2 = 1;
      double hnm1 = 2*x[j];
      dCdw[0] += dCdy[j]*hnm2;
      dCdw[1] += dCdy[j]*hnm1;
      for (int k = 2; k <= m_degree; ++k){
        double hn = 2*x[j]*hnm1-2*(k-1)*hnm2;
        dCdw[k] += dCdy[j]*hn;
        hnm2 = hnm1;
        hnm1 = hn;
      }
    }
  }
//------------------------------------------------------------------------
//                             NETWORK
//------------------------------------------------------------------------
class Network: public Function{
  public:
    typedef enum {eFullrank=1, eSigmoid=2, eSoftmax=3, eReLU=4, eHermite=5, eLowrank = 6, 
                  ePolynomial = 7, eSeLU = 8, eLogSoftmax=9, eLogSigmoid=10, eGaussRBF = 11, eNetwork} LayerTypeId;
    typedef struct {
      double m_leak;
      int m_hdegree;
      int m_pdegree;
      int m_lrank;
      const int* m_prank;
    }LayerParam;

    Network(bool ishead = true):Function(eNetwork, 0, 0, ishead) {}
    Network(int nlayers, int nin0, int noutn, const int* ltp, int nnouti, const int* nouti, 
	    LayerParam lparam, mt19937_64* engine, bool ishead = true);

    virtual ~Network(){ for (auto layer : m_layers) delete layer; }
    void add(Function* layer){m_layers.push_back(layer);}
    
    
    virtual void isHead(bool ishead){
      m_ishead = ishead;
      if (m_layers.size()) m_layers.at(0)->isHead(m_ishead);
    }
    virtual bool isHead(){return m_ishead;}

    int setParams(double* params){
      int nparams = 0;
      for (auto layer : m_layers) 
	nparams += layer->setParams(params + nparams); 
      m_params = params;
      return m_nparams;
    }
    int setGradient(double* gradient){
      int nparams = 0;
      for (auto layer : m_layers) 
	nparams += layer->setGradient(gradient + nparams); 
      m_gradient = gradient;
      return m_nparams;
    }
    int setJacobian(double* jac){
      m_jac = jac;
      for (auto layer : m_layers) layer->setJacobian(m_jac); 
      return m_njac;
    }
    void setInputOutput(int nvec, double* x, int ldx, double* y, int ldy){
      double* xi = x, *yi = y; int ldxi = ldx;
      for (auto layer : m_layers){
        layer->setInputOutput(nvec, xi, ldxi, yi, ldy);
        xi = layer->getOutput(ldxi); 
        yi = xi + layer->nOutput();
      }
      m_nvec = nvec; m_x = x; m_ldx = ldx; m_y = y; m_ldy = ldy;
    }

    void InitializeGradient(){ for (auto layer : m_layers) layer->InitializeGradient();}
    void UpdateParameters(double scl){ for (auto layer: m_layers) layer->UpdateParameters(scl); } 
    void Initialize(){ for (auto layer : m_layers) layer->Initialize();}
    virtual void print();

    virtual double* PropagateBackward(int i, double* dCdy, double* dCdx, double scl = 0.e0);
    virtual double* FeedForward(){for (auto layer: m_layers) layer->FeedForward(); return m_y + (m_toutput - m_noutput);}
    virtual void Update();
  private:
  protected:
    vector<Function*> m_layers;

    bool isActivation(int tp){
     return (tp == eSigmoid || tp == eSoftmax || tp == eReLU || tp == eSeLU || 
	     tp == eHermite || tp == eLogSoftmax || tp == eLogSigmoid);
    }
  };

void Network::Update()
  { 
    m_ninput = m_layers.at(0)->nInput();
    m_noutput = m_layers.back()->nOutput();
    if (m_layers.size()) m_layers.at(0)->isHead(m_ishead);

    m_moutput = m_toutput = m_nparams = m_njac = 0;
    for (auto layer : m_layers) {
      m_nparams += layer->nParams(); 
      m_njac     = max(m_njac, layer->nJacobian()); 
      m_toutput += layer->tOutput();
      m_moutput  = max(layer->nOutput(), m_moutput);
    }
  }

void Network::print()
  {
    cout << "Network:";
    for (auto layer : m_layers){
      if (!layer->isHead()) cout << "--->";
      layer->print();
    }
    cout << endl;
  }

double* Network::PropagateBackward(
    int i, double* dCdy, double* dCdx, double scl)
  {
    if (scl == 0.e0){
      for (auto rit = m_layers.rbegin(); rit != m_layers.rend(); ++rit){
        (*rit)->PropagateBackward(i, dCdy, dCdx);
        swap(dCdx, dCdy); 
      }
    }
    else{
      assert(0);
    }
    return m_gradient;
  }
//------------------------------------------------------------------------
//                             POLYNOMIAL
//------------------------------------------------------------------------
class Polynomial : public Network{
  public:
    Polynomial(int id, bool ishead = false):Network(ishead){m_id = id;m_ishead = ishead;};
    Polynomial(int id, int ninput, int noutput, const int* rank, int degree, std::mt19937_64* engine, bool ishead = false, int nobias = 0);

    void isHead(bool ishead){
      m_ishead = ishead;
      for (auto layer : m_layers) layer->isHead(m_ishead);
    }
    bool isHead(){return m_ishead;}

    void setInputOutput(int nvec, double* x, int ldx, double* y, int ldy){
      double* yi = y;
      for (auto layer : m_layers){
        layer->setInputOutput(nvec, x, ldx, yi, ldy);
        yi += layer->tOutput();
      }
      m_nvec = nvec; m_x = x; m_ldx = ldx; m_y = y; m_ldy = ldy;
    }
    void Update();
    double* PropagateBackward(int i, double* dCdy, double* dCdx, double scl = 0.e0);
    double* FeedForward();
    void print();
  private:
  protected:
    int m_degree;
    int m_toutput_lin, m_njac_lin;
  };

Network::Network(
    int nlayers, int nin0, int noutn, const int* ltp, int nnouti, const int* nouti, LayerParam lparam, mt19937_64* engine, bool ishead):
    Function(eNetwork, nin0, noutn, ishead)
  {
    int nin = nin0, nout, error = 0;
    for (int i = 0; i < nlayers; ++i){
      if (i+1 == nlayers){
	nout = noutn;
      }
      else{
	if (isActivation(abs(ltp[i]))) {
	  nout = nin;
	}
	else {
	  if (i < nnouti){
            int j = i;
            while (j < nnouti && nouti[j] <= 0) ++j;
            if (j == nnouti) nout = noutn;
            else nout = nouti[j];
	  }
	  else{
	    nout = noutn;
	  }
	}
      }
      
      switch (abs(ltp[i])){
        case eFullrank:                               m_layers.push_back(new Fullrank(eFullrank, nin, nout, engine, ltp[i] < 0)); break;
        case eLowrank:                                m_layers.push_back(new Lowrank(eLowrank, nin, nout, lparam.m_lrank, engine, ltp[i] < 0)); break;
        case ePolynomial:                             m_layers.push_back(new Polynomial(ePolynomial, nin, nout, lparam.m_prank, lparam.m_pdegree, engine, false, ltp[i] < 0)); break;
        case eGaussRBF:                               m_layers.push_back(new GaussRBF(eGaussRBF, nin, nout, engine)); break;
        case eSigmoid:    if (nin != nout) error = 1; m_layers.push_back(new Sigmoid(eSigmoid, nin, ltp[i] < 0)); break;
        case eLogSigmoid: if (nin != nout) error = 1; m_layers.push_back(new LogSigmoid(eLogSigmoid, nin, ltp[i] < 0)); break;
        case eSoftmax:    if (nin != nout) error = 1; m_layers.push_back(new Softmax(eSoftmax, nin)); break;
        case eLogSoftmax: if (nin != nout) error = 1; m_layers.push_back(new LogSoftmax(eLogSoftmax, nin)); break;
        case eReLU:       if (nin != nout) error = 1; m_layers.push_back(new ReLU(eReLU, nin, lparam.m_leak, ltp[i] < 0)); break;
        case eSeLU:       if (nin != nout) error = 1; m_layers.push_back(new SeLU(eSeLU, nin)); break;
        case eHermite:    if (nin != nout) error = 1; m_layers.push_back(new Hermite(eHermite, nin, lparam.m_hdegree, engine)); break;
        default: assert(0); break;
      }
      nin = nout;
    }
    if (error) {
      print();
      assert(0);
    }
    Update();
  }

void Polynomial::print()
  {
    cout << "Polynomial:";
    m_layers[0]->print(); 
    for (int i = 1; i < m_degree; ++i){
      cout << "*("; m_layers[2*i-1]->print(); cout << ","; m_layers[2*i]->print(); cout << ")";
    }
  }

Polynomial::Polynomial(
  int id, int ninput, int noutput, const int* rank, int degree, std::mt19937_64* engine, bool ishead, int nobias):
    Network(ishead), m_degree(degree)
  { 
    m_id = id;
    m_ninput = ninput;
    m_noutput = ninput;

    for (int i = 0; i < 2*degree-1; ++i) {
      if (rank[i] >= min(ninput, noutput) || !rank[i])
        m_layers.push_back(new Fullrank(Network::eFullrank, ninput,  noutput, engine, nobias));
      else
        m_layers.push_back(new Lowrank(Network::eLowrank, ninput,  noutput, rank[i], engine, nobias));
    }
    Update();
  }

void Polynomial::Update()
  { 
    Network::Update();
    for (auto layer : m_layers) layer->isHead(m_ishead);

    m_njac_lin = m_njac;
    m_toutput_lin = m_toutput;
    if (m_degree > 1) m_njac += m_noutput;
    m_toutput += m_degree*m_noutput;
  }

double* Polynomial::PropagateBackward(
    int j, double* dCdy, double* dCdx, double scl)
  {
    assert(m_ishead == m_layers.at(0)->isHead());
    if (!isHead()){
      assert(scl == 0.e0);
      if (m_degree > 1) {
	memset(dCdx, 0, sizeof(double)*m_ninput);
	scl = 1.e0;
      }
    }

    double* dCdy_n = dCdy;
    double* dCdyn_ynm1 = m_jac + m_njac_lin;
    double* ynm1 = 0;

    if (m_degree > 1) 
      ynm1 = m_y + j*m_ldy + m_toutput - 2*m_noutput;

    for (int n = m_degree-1; n > 0; --n){
      int B = 2*n, L = B-1;
      m_layers.at(B)->PropagateBackward(j, dCdy_n, dCdx, 1.e0);

      double* y_L = m_layers.at(L)->getOutput() + j * m_ldy;
      for (int i = 0; i < m_noutput; ++i){
	dCdyn_ynm1[i] = dCdy_n[i] * ynm1[i];
	dCdy_n[i] *= y_L[i];
      }
      m_layers.at(L)->PropagateBackward(j, dCdyn_ynm1, dCdx, 1.e0);
      ynm1 -= m_noutput;
    } 
    m_layers.at(0)->PropagateBackward(j, dCdy_n, dCdx, scl);
    //mprint("PB:Poly:dCdx", m_ninput, 1, dCdx, m_ldx);
    //mprint("PB:Poly:G", m_nparams, 1, m_gradient, 1);
    return m_gradient;
  }


double* Polynomial::FeedForward()
  {
    //mprint("FF:Poly:Xn", m_ninput, m_nvec, m_x, m_ldx);
    for (auto layer : m_layers){ layer->FeedForward(); } 
    if (!m_degree) return getOutput();
    
    double *yn = getOutput(), *ynm1 = m_y + m_toutput_lin;
    double* y_B0 = m_layers.at(0)->getOutput();
    for (int j = 0; j < m_nvec; ++j){
      cblas_dcopy(m_noutput, y_B0+j*m_ldy, 1, ynm1+j*m_ldy, 1);
    }
     
    for (int n = 1; n < m_degree; ++n){
      int B = 2*n, L = B-1;
      yn = ynm1 + m_noutput;
      double* y_L = m_layers.at(L)->getOutput();
      double* y_B = m_layers.at(B)->getOutput();
      for (int j = 0; j < m_nvec; ++j){
	double* yn_j = yn + j * m_ldy;
	double* ynm1_j = ynm1 + j * m_ldy;
	double* y_Lj = y_L + j * m_ldy;
	double* y_Bj = y_B + j * m_ldy;
	for (int i = 0; i < m_noutput; ++i){
	  yn_j[i] = ynm1_j[i]*y_Lj[i] + y_Bj[i];
	}
      }
      ynm1 = yn;
    }
    //mprint("FF:Poly:Yn", m_noutput, m_nvec, yn, m_ldy);
    assert(yn == getOutput());
    return yn;
  }
//------------------------------------------------------------------------
//                             LOSS
//------------------------------------------------------------------------
class Loss {
  public:
    virtual ~Loss(){}
    Loss(int ninput): m_ninput(ninput){}
    int nInput(){return m_ninput;}

    void setTargets(double* ystar, int ldystar) {m_ystar = ystar; m_ldystar = ldystar;}
    double Evaluate(int nvec, double*y, int ldy, double*e = 0, int lde = 0){
      return this->evaluate(nvec, y, ldy, m_ystar, m_ldystar, e, lde);
    }
    void Jacobian(double* y, double* dCdy, int i){
      this->jacobian(y, dCdy, m_ystar + i* m_ldystar);
    }

    double evaluate(int nvec, double*y, int ldy, double* ystar, int ldystar, double*e, int lde);
    void jacobian(double* y, double* dCdy, double* ystar);
    virtual void print(){};
  protected:
    int m_ninput, m_ldystar;
    double* m_ystar;

    virtual double error(double y, double ystar) = 0;
    virtual double error_prime(double y, double ystar) = 0;
  };

double Loss::evaluate(
    int nvec, double*y, int ldy, double* ystar, int ldystar, double*e, int lde)
  {
    double loss = 0;
    for (int i = 0; i < nvec; ++i){
      double* yi = y + i*ldy;
      double* ystari = ystar + i*ldystar;
      double erri = 0;
      for (int j = 0; j < m_ninput; ++j){
	erri += error(yi[j], ystari[j]);
//	if (isnan(erri)) cout << yi[j] << " " << ystari[j] <<  endl;
      }
      if (e) e[i*lde] = erri;
      loss += erri;
    }
    return loss;
  }

void Loss::jacobian(
    double* y, double* dydx, double* ystar)
  {
    for (int j = 0; j < m_ninput; ++j){
      dydx[j] = error_prime(y[j], ystar[j]);
    }
  }
//------------------------------------------------------------------------
//                        MEAN SQUARE ERROR
//------------------------------------------------------------------------
class MSE : public Loss{
  public:
    MSE(int ninput): Loss(ninput){}
    void print(){cout << "Mean Square Error" << endl;}
  protected:
    double error(double y, double ystar)      {return (0.5/m_ninput)*pow(y-ystar,2);}
    double error_prime(double y, double ystar){return (1.0/m_ninput)*(y-ystar);}
  };
//------------------------------------------------------------------------
//                        LOG MEAN SQUARE ERROR
//------------------------------------------------------------------------
class MSEL : public Loss{
  public:
    MSEL(int ninput): Loss(ninput){}
    void print(){cout << "Mean Square Error (Log)" << endl;}
  protected:
    static constexpr double m_eps = 1.e-15;
    double error(double y, double ystar)      {return (0.5/m_ninput)*pow(log(y/(ystar+m_eps)+m_eps),2);}
    double error_prime(double y, double ystar){return (1.0/m_ninput)*log(y/(ystar+m_eps)+m_eps)/(y+m_eps*ystar+m_eps*m_eps);}
  };
//------------------------------------------------------------------------
//                       MEAN ABSOLUTE ERROR
//------------------------------------------------------------------------
class MAE : public Loss{
  public:
    MAE(int ninput): Loss(ninput){}
    void print(){cout << "Mean Absolute Error" << endl;}
  protected:
    double error(double y, double ystar)      {return (1.0/m_ninput)*fabs(y-ystar);}
    double error_prime(double y, double ystar){return error(y, ystar)/(y-ystar);}
  };
//------------------------------------------------------------------------
//                         CROSS ENTROPY
//------------------------------------------------------------------------
class CE : public Loss{
  public:
    CE(int ninput): Loss(ninput){}
    void print(){cout << "Mean Cross Entropy" << endl;}
  protected:
    double error(double y, double ystar)      {return -(1.0/m_ninput)*ystar*log(y+m_eps);}
    double error_prime(double y, double ystar){return -(1.0/m_ninput)*ystar/(y+m_eps);}
    static constexpr double m_eps = 1.e-15;
  };
//------------------------------------------------------------------------
//                         CROSS ENTROPY WITH LOG
//------------------------------------------------------------------------
class CEL : public Loss{
  public:
    CEL(int ninput): Loss(ninput){}
    void print(){cout << "Mean Cross Entropy (Log)" << endl;}
  protected:
    double error(double y, double ystar)      {return -(1.0/m_ninput)*ystar*y;}
    double error_prime(double y, double ystar){return -(1.0/m_ninput)*ystar;}
  };
//------------------------------------------------------------------------
//                      BINARY CROSS ENTROPY
//------------------------------------------------------------------------
class BCE : public Loss{
  public:
    BCE(int ninput): Loss(ninput){}
    void print(){cout << "Mean Binary Cross Entropy " << endl;}
  protected:
    double error(double y, double ystar)      {return -(1.0/m_ninput)*(ystar*log(y+m_eps)+(1-ystar)*log(1-y+m_eps));}
    double error_prime(double y, double ystar){return -(1.0/m_ninput)*(ystar/(y+m_eps)-(1-ystar)/(1-y+m_eps));}
    static constexpr double m_eps = 1.e-15;
  };
//------------------------------------------------------------------------
//                      BINARY CROSS ENTROY WITH LOG  
//------------------------------------------------------------------------
class BCEL : public Loss{
  public:
    BCEL(int ninput): Loss(ninput){}
    void print(){cout << "Mean Binary Cross Entropy (Log)" << endl;}
  protected:
    double error(double y, double ystar){
      double g = -(1.0/m_ninput)*(ystar*y+(1-ystar)*log(1-exp(y)+m_eps));
      //cout << scientific << std::setprecision(10);
      //sprint("g=", g, " ystar=", ystar, " y=", y, " exp(y)=", exp(y), " log=", log(1-exp(y)+m_eps),"\n");
      return g;
    }
    double error_prime(double y, double ystar){
      double gprime = -(1.0/m_ninput)*(ystar-(1-ystar)*exp(y)/(1-exp(y)+m_eps));
      //sprint("gprime=", gprime, " ystar=", ystar, " y=", y, " exp(y)=", exp(y), "\n");
      return gprime;
    }
    static constexpr double m_eps = 1.e-15;
  };
