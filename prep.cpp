//------------------------------------------------------------------------
//                             PREPROCESS
//------------------------------------------------------------------------
class Preprocessor{
  public:
    virtual ~Preprocessor(){}
    virtual void calc_scale(int nvec, int n, double* x, int ldx, double* scl_vecs) = 0;
    virtual void scale(int nvec, int n, double* x, int ldx, double* scl_vecs);
    virtual void unscale(int nvec, int n, double* x, int ldx, double* scl_vecs);
    virtual void print(){}
  private:
  protected:
  };

void Preprocessor::scale(
    int nvec, int n, double* x, int ldx, double* scl_vecs)
  {
    double* down_vec = scl_vecs, *up_vec = scl_vecs + n;
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i*ldx;
      for (int j = 0; j < n; ++j) 
	xi[j] = (xi[j] - up_vec[j])/down_vec[j]; 
    }
  }

void Preprocessor::unscale(
    int nvec, int n, double* x, int ldx, double* scl_vecs)
  {
    double* down_vec = scl_vecs, *up_vec = scl_vecs + n;
    for (int i = 0; i < nvec; ++i){
      double* xi = x + i*ldx;
      for (int j = 0; j < n; ++j) 
	xi[j] = xi[j]*down_vec[j] + up_vec[j];
    }
  }

class NoScaling : public Preprocessor{
  public:
    void calc_scale(int nvec, int n, double* x, int ldx, double* scl_vecs){};
    void scale(int nvec, int n, double* x, int ldx, double* scl_vecs){};
    void unscale(int nvec, int n, double* x, int ldx, double* scl_vecs){};
    void print(){sprint("No scaling\n");}
  private:
  protected:
  };

class MaxMinScaling : public Preprocessor{
  public:
    void calc_scale(int nvec, int n, double* x, int ldx, double* scl_vecs);
    void print(){sprint("MaxMin Scaling\n");}
  private:
  protected:
  };

void MaxMinScaling::calc_scale(
    int nvec, int n, double* x, int ldx, double* scl_vecs)
  {
    double* max_vec = scl_vecs, *min_vec = scl_vecs + n;
    memcpy(max_vec, x, sizeof(double)*n); 
    memcpy(min_vec, x, sizeof(double)*n); 

    for (int i = 1; i < nvec; ++i){
      double* xi = x + i*ldx;
      for (int j = 0; j < n; ++j) {
        min_vec[j] = std::min(min_vec[j], xi[j]);                      
        max_vec[j] = std::max(max_vec[j], xi[j]);                      
      }
    }
    for (int j = 0; j < n; ++j){
      max_vec[j] -= min_vec[j];
      if (max_vec[j] == 0.0) max_vec[j] = 1;
    }
  }

class ZscoreScaling : public Preprocessor{
  public:
    void calc_scale(int nvec, int n, double* x, int ldx, double* scl_vecs);
    void print(){sprint("Z-score Scaling\n");}
  private:
  protected:
  };

void ZscoreScaling::calc_scale(
    int nvec, int n, double* x, int ldx, double* scl_vecs)
  {
    double* std = scl_vecs, *mean = scl_vecs + n;
    memset(scl_vecs, 0, sizeof(double)*2*n); 

    for (int i = 0; i < nvec; ++i){
      double* xi = x + i*ldx;
      for (int j = 0; j < n; ++j) { mean[j] += xi[j]; }
    }
    for (int j = 0; j < n; ++j) { mean[j] /= nvec; }

    for (int i = 0; i < nvec; ++i){
      double* xi = x + i*ldx;
      for (int j = 0; j < n; ++j) { std[j] += pow((xi[j]-mean[j]),2); }
    }
    for (int j = 0; j < n; ++j){ 
      std[j] = sqrt(std[j]/nvec);
      if (std[j] == 0.0) std[j] = 1;
    }
  }
