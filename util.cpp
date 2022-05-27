//------------------------------------------------------------------------
//                             UTIL
//------------------------------------------------------------------------
template<typename T>
void sprint(const T& v){ cout << v;}

template<typename S, typename ...T>
void sprint(const S& arg, const T& ...args)
  {
    cout << arg;
    sprint(args...);
  }

template<typename T, typename S = T>
void vprint(int n, const T* v, int incv = 1)
  {
    for (int i = 0; i < n; ++i){
      cout << (S) (v[i*incv]) << " ";
    }
    cout << endl;
  }

template<typename T, typename S = T>
void mprint(const char* what, int n, int m, const T* v, int ldv = 0)
  {
    if (!ldv) ldv = n;
    cout << what << '(' << n << ',' << m << ')' << endl 
         << "--------------" << endl;
    for (int i = 0; i < m; ++i){
      vprint<T, S>(n, v + i* ldv);
    }
  }

void log_time_stamp(FILE* log)
  {
    time_t rawtime;
    struct tm timeinfo;
    char buf[26];
    time(&rawtime);
    localtime_r(&rawtime, &timeinfo);
    asctime_r(&timeinfo, buf);
    fprintf(log, "%s", buf); 
  }

string get_path(
    const char* path, const char* prefix, const char* suffix)
  {
    char slash = '/';
    const char* name = strrchr(path, slash);
    if (name) ++name;
    else name = path;
    int ncopy = strlen(name);

    char dot = '.';
    const char* end = strrchr(name, dot);
    if (end && end > name) ncopy -= strlen(end);

    string new_path(prefix);
    new_path.append(name, ncopy);
    new_path += suffix;
    
    return new_path;
  }

int do_user_stop(int del=0)
  {
    struct stat buffer;
    int exist = stat(".stop",&buffer);
    if(exist == 0) {
      if (del) remove(".stop");
      return 1;
    }
    else return 0;
  }
//------------------------------------------------------------------------
//                          DATAPERMUTATION
//------------------------------------------------------------------------
template<typename T>
void permute_inplace(
    int nvec, int n, T* x, int ldx, int* permv, T* wrk)
  {
    int i = 0, j;
    while(i < nvec){
      j = permv[i];
      if (j == i) {++i; continue;}
      T* xi = x + i*ldx;
      T* xj = x + j*ldx;
      memcpy(wrk, xj, sizeof(T)*n);
      memcpy(xj, xi, sizeof(T)*n);
      memcpy(xi, wrk, sizeof(T)*n);
      permv[i] = permv[j];
      permv[j] = j;
    }
  }
//------------------------------------------------------------------------
//                             PROGRESS
//------------------------------------------------------------------------
class Progress{
  public:
    Progress(int step = 1):m_step(step){}
    void show(double val, int iepoch, int ncycles, double elapsed_s, double timeout_s) {
      if (iepoch % m_step) return;

      double remain_time = (elapsed_s/iepoch)*(ncycles - iepoch); 
      double avail_time = timeout_s - elapsed_s;
      
      int done;
      double remain_s;
      if (remain_time > avail_time){
	remain_s = avail_time;
	done = (int)(elapsed_s/timeout_s*100);
      }
      else{
	remain_s = remain_time;
	done = (int)(100.*iepoch/ncycles);
      }
      done += 1;
      int todo = 100 - done;

      fprintf(stderr,"%1.5e/%5d", val, iepoch+1);
      for (int l = 0; l <= (int)done;++l)
        fprintf(stderr,".");
      fprintf(stderr,"%4.1fs/%4.1fs (%2d%%)", elapsed_s, remain_s, (int)done);
      for (int l = 0; l < todo;++l)
        fprintf(stderr,".");
      fprintf(stderr,"\r");
    }
    void stop(){ fprintf(stderr,"\n"); }
  private:
  protected:
    int m_step;
  };

//------------------------------------------------------------------------
//                           ACCURACY
//------------------------------------------------------------------------
double accuracy(
    int nrecords, const double* yn, int ldy, const double* ystar, int ldystar, 
    int noutput, const int* noutputi, double* accuracyi, int logit, int msg)
  {
    double error = 0;
    for (int j = 0; j < noutput; ++j){
      double accuracyij = nrecords;
      int nout = noutputi[j+1] - noutputi[j];

      for (int i = 0; i < nrecords; ++i){
        const double* yni = yn + i*ldy;
        const double* ystari = ystar + i*ldystar;

	int prediction = -1, target = -1;
	double predval = -1.e0, targval = -1e0;
	int predk = -1, targk = -1;
	double sum = 0.e0;
	for (int k = noutputi[j]; k < noutputi[j+1]; ++k){
	  double ynik = yni[k];
	  if (logit) ynik = exp(ynik);
	  assert(ystari[k] >= 0.e0);
	  assert(ynik >= 0.e0);
	  if (nout == 1){
	    target = 0; targval = ystari[k]; predval = ynik; 
	    if (round(ynik) == round(ystari[k])) { prediction = k; }
	    sum = 1;
	  }
	  else if (nout == 2){
	    if (round(ynik) == 1) { prediction = k; predval = ynik; }
	    if (round(ystari[k]) == 1) { target = k; targval = ystari[k]; }
	    sum += ynik;
	  }
          else {
	    if (ynik > predval){predval = ynik; prediction = k;}
	    if (ystari[k] > targval){targval = ystari[k]; target = k;}
	    sum += ynik;
	  }
	}
	if (fabs(sum-1.e0) > 1e-8){
	  cout << "ERROR:Sum-1e0:" << fabs(sum-1.e0) << endl;
//	  assert(fabs(sum-1.e0) < 1e-8);
	}

	if (target != prediction) {
	  --accuracyij;
//	  if (msg)
//            cout << "(" << i << "," << j << ") Prediction " << prediction << "(" << predval << ")" << 
//	                                  " Missed Target " << target << "(" << targval << ")" << endl;
	}
      }
      accuracyij /= nrecords;
      if (accuracyi) accuracyi[j] = accuracyij;
      error += accuracyij;
    }

    return error/noutput;
  }

void regress_prediction(
    int nrecords, int ntargets, const double* yn, int ldy, const double* ystar, int ldystar)
  {
     for (int i = 0; i < nrecords; ++i){
       for (int j = 0; j < ntargets; ++j){
         double e = fabs(yn[i*ldy+j] - ystar[i*ldystar+j]);
         cout << "Prediction " << yn[i*ldy+j] << " Target " << ystar[i*ldystar+j] 
              << " Error " << e << " (" << (100*e)/ystar[i*ldystar+j] << "%)" << endl;
       }
     }
  }
