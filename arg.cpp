//------------------------------------------------------------------------
//                       ARGDICTIONARY
//------------------------------------------------------------------------
class argval{
  public:
    virtual ~argval(){}
    argval(){}
    argval(const char* val){set(val);}
    virtual void set(const char* val){
      //std::cout << "to string: " << val << std::endl;
      m_val = std::string(val);
    }
    template<typename T> T get(){return std::any_cast<T>(m_val);}
    virtual int noEqual(){return 0;}
    virtual int noValue(){return 0;}
  protected:
    std::any m_val;
  };

class argint : public argval{
  public:
    argint(int val){m_val = val;}
    void set(const char* val){ 
      //std::cout << "to int: " << val << std::endl;
      m_val = std::stoi(val); 
    }
    int noEqual(){return 1;}
  };

class argdouble : public argval{
  public:
    argdouble(double val){m_val = val;}
    void set(const char* val){ 
      //std::cout << "to double: " << val << std::endl;
      m_val = std::stod(val); 
    }
  };

class argintlist : public argval{
  public:
    argintlist(){m_val = std::vector<int>();}
    argintlist(std::vector<int> val){m_val = val;}
    int noValue(){
      std::vector<int>& vals = std::any_cast<std::vector<int>& >(m_val);
      vals.clear();
      return 0;
    }
    void set(const char* val){
      //std::cout << "to list: " << val << std::endl;
      std::vector<int>& vals = std::any_cast<std::vector<int>& >(m_val);
      vals.clear();
      
      static constexpr int no_prev = 1234567890;
      int prev = no_prev;

      const char* p0 = val;
      int len = strlen(p0);

      auto add_value = [&](){
	int dash = 0;
	const char* pcomma = strchr(p0, ',');
	const char* pdash  = strchr(p0, ':');
	const char* p = pcomma;
	if (!pcomma || (pdash && pdash < pcomma)) {
	  p = pdash;
	  dash = 1;
	}
	if (p) len = p - p0;
        if (len) {
	  int next = std::stoi(string_view(p0, len).data());
	  if (prev == no_prev){
	    //std::cout << next << " ";
            vals.push_back(next);
	  }
	  else{
	    assert(next != 0 && prev*next >= 0); 
	    int one = (prev != 0 ? prev/abs(prev) : 
	              (next != 0 ? next/abs(next) : 1));
            
            for (int i = prev+one; i != next; i += one){
	      //std::cout << i << " ";
	      vals.push_back(i);
	    }
	    //std::cout << next << " ";
	    vals.push_back(next);
	    prev = no_prev;
	  }

	  if (dash) prev = next;
        }

	//std::cout << endl;
	return p;
      };

      const char* p = add_value();
      while(p){
	p0 = p+1;
        p = add_value();
      }
    }
  };

class ArgDictionary{
  public:
    ArgDictionary(std::map<std::string, argval*>& dictionary){
      //std::cout << "Copy" <<std::endl;
      m_dictionary = dictionary;
    }
    ArgDictionary(std::map<std::string, argval*>&& dictionary){
      //std::cout << "Move" <<std::endl;
      m_dictionary = dictionary;
    }
    argval* operator [] (const std::string& key) {return m_dictionary[key];}
    ~ArgDictionary(){for (auto p : m_dictionary) delete p.second;}


    int parse_args(int nargs, char** argc, const char* def)
    {
      int error = 0;
      int def_set = 0;
      char one = '1';
      for (int i = 0; i < nargs; ++i){
        if (argc[i][0] != '-') {
          if (!def_set){
            m_dictionary[def]->set(argc[i]);
	    def_set = 1;
          }
          else{
            std::cout << "Invalid argument:" << argc[i] << std::endl;
            error = 1;
          }
          continue;
        }
    
        char* equal = strchr(argc[i], '=');
        if (equal == 0){
          auto it = m_dictionary.find(argc[i]+1);
	  if (it == m_dictionary.end()) {
            std::cout << "Invalid argument:" << argc[i] << std::endl;
	    error = 1;
	  }
	  else{
            error = it->second->noEqual(); //set default
	  }
          continue;
        }
        
	*equal = '\0';
        auto it = m_dictionary.find(argc[i]+1);
        if (it == m_dictionary.end()){
          std::cout << "Invalid argument:" << argc[i] << std::endl;
          error = 1;
        }
	else{
          if (strlen(equal+1)){
	    it->second->set(equal+1);
	  }
	  else{
            error = it->second->noValue(); //set default
	  }
	}
	*equal = '='; //reset equal
      }
      return error;
    }
  private:
    std::map<std::string, argval*> m_dictionary;
  };

//------------------------------------------------------------------------
//                       ARGDICTIONARY
//------------------------------------------------------------------------
class Arguments{
  public:
    void print();
    string m_filename;
    string m_argfile;
    string m_parmfile;
    vector<int> m_output;
    vector<int> m_exclude;
    vector<int> m_include;
    vector<int> m_label;
    vector<int> m_layertypes;
    vector<int> m_layers;
    int m_batch;
    int m_ncycles;
    int m_nthreads;
    double m_thresh;
    double m_timeout; 
    double m_validate;
    double m_leak;
    int m_hdegree;
    int m_pdegree;
    int m_lrank;
    vector<int> m_prank;
    int m_scale;
    int m_seed;
    double m_alpha;
    double m_beta1;
    double m_beta2;
    int m_optim;
    int m_loss;
    double m_check;
    int m_encode;
    int m_nepoch;
    int m_natoms;
    double m_epsilon;
    double m_mixrate;
    int m_ga;
    double m_pso_c[5];
//
    int m_msglvl; 
  protected:
  };

void Arguments::print()
  {
     std::cout << "filename=" << m_filename << endl;
     std::cout << "-argfile=" << m_argfile << endl;
     std::cout << "-parmfile=" << m_parmfile << endl;
     std::cout << "-output="; vprint(m_output.size(), m_output.data()); 
     std::cout << "-exclude="; vprint(m_exclude.size(), m_exclude.data()); 
     std::cout << "-include="; vprint(m_include.size(), m_include.data()); 
     std::cout << "-label="; vprint(m_label.size(), m_label.data()); 
     std::cout << "-layertypes="; vprint(m_layertypes.size(), m_layertypes.data()); 
     std::cout << "-layerdim="; vprint(m_layers.size(), m_layers.data()); 
     std::cout << "-ncycles=" << m_ncycles << endl 
               << "-batch=" << m_batch << endl
               << "-nthreads=" << m_nthreads << endl
               << "-error=" << m_thresh << endl
               << "-validate=" << m_validate << endl
               << "-timeout=" << m_timeout << endl
               << "-leak=" << m_leak << endl
               << "-hdegree=" << m_hdegree << endl
               << "-pdegree=" << m_pdegree << endl
               << "-lrank=" << m_lrank << endl;
     std::cout << "-prank="; vprint(m_prank.size(), m_prank.data()); 
     std::cout << "-scale=" << m_scale << endl
               << "-seed=" << m_seed << endl
               << "-optimizer=" << m_optim << endl
               << "-loss=" << m_loss << endl
               << "-alpha=" << m_alpha << endl
               << "-beta1=" << m_beta1 << endl
               << "-beta2=" << m_beta2 << endl
               << "-check=" << m_check << endl
               << "-encode=" << m_encode << endl
               << "-epsilon=" << m_epsilon << endl
               << "-mixrate=" << m_mixrate << endl
               << "-natoms=" << m_natoms << endl
               << "-nepoch=" << m_nepoch << endl
               << "-ga=" << m_ga << endl;
     std::cout << "-pso_c="; vprint(4, m_pso_c);
     std::cout << "-pso_k=" << m_pso_c[4] << endl;
  }

ArgDictionary* getDefaultDictionary()
  {
    ArgDictionary* dict = new ArgDictionary({
      {"filename", new argval("")},
      {"argfile", new argval("")},
      {"parmfile", new argval("")},
      {"output", new argintlist({0})},
      {"exclude", new argintlist},
      {"include", new argintlist},
      {"label", new argintlist},
      {"layertypes", new argintlist({0,1,0})},
      {"layerdim", new argintlist({7,7})},
      {"ncycles", new argint(100000)},
      {"batch", new argint(32)},
      {"nthreads", new argint(1)},
      {"error", new argdouble(0.)},
      {"validate", new argdouble(0.1)},
      {"timeout", new argdouble(10.)},
      {"leak", new argdouble(0.001)},
      {"hdegree", new argint(5)},
      {"pdegree", new argint(3)},
      {"lrank", new argint(1)},
      {"prank", new argintlist({0,1})},
      {"scale", new argint(1)},
      {"seed", new argint(0)},
      {"optimizer", new argint(3)},
      {"loss", new argint(0)},
      {"alpha", new argdouble(0.01)},
      {"beta1", new argdouble(0.9)},
      {"beta2", new argdouble(0.999)},
      {"check", new argdouble(0)},
      {"encode", new argint(1)},
      {"epsilon", new argdouble(1.0)},
      {"mixrate", new argdouble(0.8)},
      {"nepoch", new argint(5000)},
      {"natoms", new argint(0)},
      {"ga", new argint(0)},
      {"pso_c1", new argdouble(1.)},
      {"pso_c2", new argdouble(1.)},
      {"pso_c3", new argdouble(1.)},
      {"pso_c4", new argdouble(1.)},
      {"pso_k",  new argdouble(0.1)},

      {"help", new argint(0)},
    });
    return dict;
  }

ArgDictionary* getDictionaryFromArguments(
    const Arguments& args)
  {
    ArgDictionary* dict = new ArgDictionary({
      {"filename", new argval(args.m_filename.c_str())},
      {"argfile", new argval(args.m_argfile.c_str())},
      {"parmfile", new argval(args.m_parmfile.c_str())},
      {"output", new argintlist(args.m_output)},
      {"exclude", new argintlist(args.m_exclude)},
      {"include", new argintlist(args.m_include)},
      {"label", new argintlist(args.m_label)},
      {"layertypes", new argintlist(args.m_layertypes)},
      {"layerdim", new argintlist(args.m_layers)},
      {"ncycles", new argint(args.m_ncycles)},
      {"batch", new argint(args.m_batch)},
      {"nthreads", new argint(args.m_nthreads)},
      {"error", new argdouble(args.m_thresh)},
      {"validate", new argdouble(args.m_validate)},
      {"timeout", new argdouble(args.m_timeout)},
      {"leak", new argdouble(args.m_leak)},
      {"hdegree", new argint(args.m_hdegree)},
      {"pdegree", new argint(args.m_pdegree)},
      {"lrank", new argint(args.m_lrank)},
      {"prank", new argintlist(args.m_prank)},
      {"scale", new argint(args.m_scale)},
      {"seed", new argint(args.m_seed)},
      {"optimizer", new argint(args.m_optim)},
      {"loss", new argint(args.m_loss)},
      {"alpha", new argdouble(args.m_alpha)},
      {"beta1", new argdouble(args.m_beta1)},
      {"beta2", new argdouble(args.m_beta2)},
      {"check", new argdouble(args.m_check)},
      {"encode", new argint(args.m_encode)},

      {"epsilon", new argdouble(args.m_epsilon)},
      {"mixrate", new argdouble(args.m_mixrate)},
      {"nepoch", new argint(args.m_nepoch)},
      {"natoms", new argint(args.m_natoms)},
      {"ga", new argint(args.m_ga)},
      {"pso_c1", new argdouble(args.m_pso_c[0])},
      {"pso_c2", new argdouble(args.m_pso_c[1])},
      {"pso_c3", new argdouble(args.m_pso_c[2])},
      {"pso_c4", new argdouble(args.m_pso_c[3])},
      {"pso_k", new argdouble(args.m_pso_c[4])},

      {"help", new argint(0)},
    });
    return dict;
  }

Arguments* getCLArguments(
    int nargs, char** argc, ArgDictionary& dictionary, char* exec)
  {
    constexpr const char* filename_error = {"Ambiguous or no filename specified."};

    int cmd_err = dictionary.parse_args(nargs, argc, "filename");
    
    int help = dictionary["help"]->get<int>();
    if (help || cmd_err){
      std::cout << exec 
              << " <filename.csv>"  << endl 
              << " [-argfile=<file with input arguments>]"  << endl 
              << " [-parmfile=<file with saved network parameters>]"  << endl 
	      << " [-output=<class columns in csv: <i1,i2,...,in> column ids, <=0 from end>]" << endl
	      << " [-exclude=<i1,i2,...,in> column ids to exclude, <=0 from end>]" << endl
	      << " [-include=<i1,i2,...,in> column ids to include, <=0 from end>]" << endl
	      << " [-label=<i1,i2,...,in> column ids to force categorical encoding, <=0 from end>]" << endl
              << " [-layertypes=<layertype list"
	         " (1:Fullrank,2:Sigmoid,3:Softmax,4:ReLu,5:Hermite,"
		 "  6:Lowrank,7:Polynomial,8:SeLU,9:LogSoftmax,10:LogSigmoid)>]" << endl
              << " [-layerdim=<layer dimensions>]" << endl
              << " [-ncycles=<Max NN training cycles>]" << endl 
              << " [-batch=<training batch>]" << endl
              << " [-nthreads=<nthreads>]" << endl
              << " [-error=<stopping error>]" << endl
              << " [-validate=<validation, -1<x<0 record %, 0 <x<1 cross record %, x<-1 last x records, x>1 infer last x records>]" << endl
              << " [-timeout=<timeout(s)>]" << endl
              << " [-leak=<ReLU leak>]" << endl
              << " [-hdegree=<Hermite degree>]" << endl
              << " [-pdegree=<Polynomial degree>]" << endl
              << " [-lrank=<Rank for Lowrank layers>]" << endl
              << " [-prank=<list of rank for Polynomial layers>]" << endl
              << " [-scale=<initial scaling (0:No Scaling, 1:MinMax, 2:Zscore)>]" << endl
              << " [-seed=<seed for RNG>]" << endl
              << " [-optimizer=<optimizer (0:GD, 1:Momentum, 2:RMSP, 3:Adam)>]" << endl
              << " [-loss=<loss function (0:MSE, 1:MAE, 2:CE, 3:BCE, 4:BCEL, 5:CEL, 6:LMSE)>]" << endl
              << " [-alpha=<learning rate, <0 adaptive>]" << endl
              << " [-beta1=<optimizer parameter>]" << endl
              << " [-beta2=<optimizer parameter>]" << endl
              << " [-check=<gradient threshold>]" << endl
              << " [-encode=<encoding (0: Binary, 1: Label, 2:One hot>]" << endl
              << " [-epsilon=<BSA epsilon>]" << endl
              << " [-mixrate=<BSA mixrate>]" << endl
              << " [-natoms=<BSA population size>]" << endl
              << " [-nepoch=<BSA number of epochs>]" << endl
              << " [-ga=<GA (0:BSAO, 1:SBMBSA, 2:SBPSO, 3:BiPSO, 4:BiMBSA)>]" << endl
              << " [-pso_ci=<PSO c parameter for i=1,2,3,4>]" << endl
	      << " [-pso_k=<PSO k-tournament %>]" << endl
              << " [-help]" << std::endl;
      return (Arguments*)0;
    }
    Arguments* args = new Arguments;
    args->m_filename   = dictionary["filename"]->get<std::string>();
    args->m_argfile    = dictionary["argfile"]->get<std::string>();
    args->m_parmfile   = dictionary["parmfile"]->get<std::string>();
    args->m_output     = dictionary["output"]->get<std::vector<int> >();
    args->m_exclude    = dictionary["exclude"]->get<std::vector<int> >();
    args->m_include    = dictionary["include"]->get<std::vector<int> >();
    args->m_label      = dictionary["label"]->get<std::vector<int> >();
    args->m_layertypes = dictionary["layertypes"]->get<std::vector<int> >();
    args->m_layers     = dictionary["layerdim"]->get<std::vector<int> >();
    args->m_ncycles    = dictionary["ncycles"]->get<int>();
    args->m_batch      = dictionary["batch"]->get<int>();
    args->m_nthreads   = dictionary["nthreads"]->get<int>();
    args->m_thresh     = dictionary["error"]->get<double>();
    args->m_validate   = dictionary["validate"]->get<double>();
    args->m_timeout    = dictionary["timeout"]->get<double>();
    args->m_leak       = dictionary["leak"]->get<double>();
    args->m_hdegree    = dictionary["hdegree"]->get<int>();
    args->m_pdegree    = dictionary["pdegree"]->get<int>();
    args->m_lrank      = dictionary["lrank"]->get<int>();
    args->m_prank      = dictionary["prank"]->get<std::vector<int> >();
    args->m_scale      = dictionary["scale"]->get<int>();
    args->m_seed       = dictionary["seed"]->get<int>();
    args->m_optim      = dictionary["optimizer"]->get<int>();
    args->m_loss       = dictionary["loss"]->get<int>();
    args->m_alpha      = dictionary["alpha"]->get<double>();
    args->m_beta1      = dictionary["beta1"]->get<double>();
    args->m_beta2      = dictionary["beta2"]->get<double>();
    args->m_check      = dictionary["check"]->get<double>();
    args->m_encode     = dictionary["encode"]->get<int>();
    args->m_nepoch     = dictionary["nepoch"]->get<int>();
    args->m_natoms     = dictionary["natoms"]->get<int>();
    args->m_ga         = dictionary["ga"]->get<int>();
    args->m_mixrate    = dictionary["mixrate"]->get<double>();
    args->m_epsilon    = dictionary["epsilon"]->get<double>();
    args->m_pso_c[0]   = dictionary["pso_c1"]->get<double>();
    args->m_pso_c[1]   = dictionary["pso_c2"]->get<double>();
    args->m_pso_c[2]   = dictionary["pso_c3"]->get<double>();
    args->m_pso_c[3]   = dictionary["pso_c4"]->get<double>();
    args->m_pso_c[4]   = dictionary["pso_k"]->get<double>();
    args->m_msglvl     = 0;
    return args;
  }

vector<char*> argsFromInputFile(
    const char* filename)
  {
    FILE* input = fopen(filename, "r");
    assert(input);
    vector<char*> argv;
    int nread = 0;
    while(1) {
      size_t len = 0;
      char* line = 0;
      int nread = getline(&line, &len, input); 
      if (nread == -1) break;
      if (!nread) {
	free(line);
	continue; //empty line
      }

      line[nread-1] = '\0'; //replace newline with '\0'
      int k = 0;
      for (int i = 0; i < nread-1; ++i){
	if (line[i] == '#') break;  //get rid of comment
        if (line[i] != ' ') line[k++] = line[i]; //trim white
      }
      line[k] = '\0';
      if (k) argv.push_back(line);
      else free(line);
    }
    fclose(input);
    return argv;
  }

Arguments* getArguments(
    int nargs, char** argv)
  {
    ArgDictionary* dictionary = getDefaultDictionary();
    Arguments* args = getCLArguments(nargs-1, argv+1, *dictionary, argv[0]); //arguments from program CL
    delete dictionary;
    if (!args) return 0;         
    if (args->m_argfile.empty()) return args;                          //no file arguments

    vector<char*> fargv = argsFromInputFile(args->m_argfile.c_str());  //pseudo CL from file
    delete args;                                                       //delete arguments from CL

    dictionary = getDefaultDictionary();
    char** fargv_ptr = fargv.data();
    args = getCLArguments(fargv.size(), fargv_ptr, *dictionary, argv[0]);  //file arguments from pseudo CL
    for (auto farg : fargv) free(farg);                                //delete pseudo CL
    delete dictionary;
    if (!args) return 0;

    dictionary = getDictionaryFromArguments(*args);                    //file dictionary from file arguments
    delete args;                                                       //delete file arguments
    
    args = getCLArguments(nargs-1, argv+1, *dictionary, argv[0]);      //arguments from program CL with file dictionaru 
    delete dictionary;
    return args;                                                   
  }

int processArguments(
    Arguments* args)
  {
    assert(!args->m_filename.empty());
    int ncols = cntCSVColumns(args->m_filename.c_str());
    assert(ncols);

    auto sort_unique = [](auto& ilst, int ncols)
    {
      int n = ilst.size();
      if (!n) return;

      for (auto&& elem : ilst)  if (elem <= 0) elem += ncols;
      sort(ilst.begin(), ilst.end());

      int i = 1, j = 0;
      for (; i < n; ++i){
	if (ilst[i] > ncols) break;
	if (ilst[i] != ilst[j]){
	  if (++j != i) ilst[j] = ilst[i];
	}
      }
      ilst.resize(j+1); 
    };

    sort_unique(args->m_output, ncols);
    //add output to include
    for (auto out : args->m_output) args->m_include.push_back(out); 
    sort_unique(args->m_include, ncols);
    sort_unique(args->m_exclude, ncols);
    sort_unique(args->m_label, ncols);

    //remove included from excluded
    int nexc = args->m_exclude.size();
    int ninc = args->m_include.size();
    int jexc = 0;
    for (int iexc = 0, iinc = 0; iexc < nexc; ++iexc){
      if (iinc < ninc && args->m_exclude[iexc] == args->m_include[iinc]){
        ++iinc;
      }
      else{
	if (jexc != iexc) args->m_exclude[jexc] = args->m_exclude[iexc];
	++jexc;
      }
    }
    args->m_exclude.resize(jexc); 

//  fill in polynomial degress
    for (int i = args->m_prank.size(); i < 2*args->m_pdegree-1; ++i) 
      args->m_prank.push_back(args->m_prank[i-1]); 
    return ncols;
  }

