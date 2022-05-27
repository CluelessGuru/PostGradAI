//------------------------------------------------------------------------
//                             TRAIN
//------------------------------------------------------------------------
typedef struct training_param{
  int m_loss;
  int m_ncycles;
  int m_batch;
  int m_optim;
  int m_seed;
  double m_check;
  double m_timeout; 
  double m_thresh; 
  double m_alpha; 
  double m_beta1;
  double m_beta2;
  int m_msglvl;
  const char* m_log;
  int m_adapt;
}TrainingParam;

void train(
    Network* network, TrainingParam* args, 
    int ntargets, int nfeatures,
    int nrecords, double* x, int ldx, double* ystar, int ldystar, 
    int ntest, double* t, int ldt, double* ystart, int ldystart)
  {
    Loss* loss = 0;
    switch (args->m_loss){
      case 0: loss = new MSE(ntargets); break;
      case 1: loss = new MAE(ntargets); break;
      case 2: loss = new CE (ntargets); break;
      case 3: loss = new BCE(ntargets); break;
      case 4: loss = new BCEL(ntargets); break;
      case 5: loss = new CEL(ntargets); break;
      case 6: loss = new MSEL(ntargets); break;
      default: assert(0); break;
    }
//--------Build Trainer
    int seed = args->m_seed;
    if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 engine(seed);

    Trainer* trainer = 0;
    TestTrainer* tt = 0;
    GradientTrainer* gd = 0, *sgd = 0, *gdptr = 0;

    switch (args->m_optim){
      case 0: gd = new GD(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha); break;
      case 1: gd = new MomentumGD(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha, args->m_beta1); break;
      case 2: gd = new RMSPGD(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha, args->m_beta1); break;
      case 3: gd = new AdamGD(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha, args->m_beta1, args->m_beta2); break;
      case 4: gd = new GSL_Trainer(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha); break;
      default: assert(0); break;
    }
    gd->network(network); 
    gd->loss(loss);
    trainer = gd;
    gdptr = gd;

    if (args->m_batch > 0 && args->m_batch < nrecords){ 
      sgd = new SGD(args->m_ncycles, args->m_thresh, args->m_timeout, args->m_alpha, args->m_batch, gdptr, &engine);
      trainer = sgd; 
      gdptr = sgd;
    }

    if (ntest){
      tt = new TestTrainer(args->m_ncycles, args->m_thresh, args->m_timeout, gdptr);
      tt->setTestData(ntest, t, ldt, ystart, ldystart);
      trainer = tt;
    }

//--------Build Progress
    Progress progress(1);
    if (args->m_msglvl) trainer->progress(&progress);

    int njac = network->nJacobian();
    int lwrk = trainer->lwrk(nrecords);
    double* wrk = (double*) malloc(sizeof(double)*(lwrk+njac));
    network->setJacobian(wrk);
    
    if (args->m_check > 0.e0) {
      sprint("Checking Gradient:", args->m_check,"\n");
      loss->print();
      gdptr->CheckGradient(nrecords, x, ldx, ystar, ldystar, wrk+njac, args->m_check); 
    }

    SaveBestCallback save(args->m_log);
    if (args->m_msglvl) {
      trainer->log(args->m_log);
      trainer->callback(&save);
    }
    trainer->InitTraining(args->m_adapt);
    trainer->Train(nrecords, x, ldx, ystar, ldystar, wrk+njac);

    free(wrk);
    if (sgd) delete sgd;
    if (gd) delete gd;
    if (tt) delete tt;
    delete loss;
  }
//------------------------------------------------------------------------
//                             RUN
//------------------------------------------------------------------------
double run_model(
    const Arguments* args, TaskData* data, int ntest0)
  {
    int ntest = abs(ntest0);
    int msg = args->m_msglvl & 4;
    int nrecords = data->m_nrecords-ntest;
    int ntargets = data->m_ntargets;
    int nfeatures = data->m_nfeatures;
    int ldystar = data->m_ntargets;
    int ldx = data->m_nfeatures;
    int scale = abs(args->m_scale);
    double* x = &(data->m_records[0]); 
    double* ystar = &(data->m_targets[0]); 

    bool is_classification = args->m_loss == 2 || args->m_loss == 3 || args->m_loss == 4 || args->m_loss == 5;
    bool is_logit = args->m_loss == 4 || args->m_loss == 5;
    bool is_mse = args->m_loss == 0 || args->m_loss == 6;
 
    Preprocessor* prepro;
    switch (scale){
      case 0: prepro = new NoScaling; break;
      case 1: prepro = new MaxMinScaling; break;
      case 2: prepro = new ZscoreScaling; break;
      default: assert(0); break;
    }
    if (msg) prepro->print();

    double* scale_vecs = (double*) malloc(sizeof(double)*2*max(ntargets, nfeatures));
    prepro->calc_scale(nrecords+max(0,ntest0), nfeatures, x, ldx, scale_vecs);
    prepro->scale(nrecords+ntest, nfeatures, x, ldx, scale_vecs);
    if (!is_classification && args->m_scale > 0){
      prepro->calc_scale(nrecords+max(0,ntest0), ntargets, ystar, ldystar, scale_vecs);
      prepro->scale(nrecords+ntest, ntargets, ystar, ldystar, scale_vecs);
    }
//-------Grab pointer from task data
    int batch = args->m_batch == 0 ? nrecords : args->m_batch;
    int nlayers = args->m_layertypes.size();
    const int* layer_types = args->m_layertypes.data();
    int nio = args->m_layers.size();
    const int* layer_nio = args->m_layers.data();

//-------Build Network
    int seed = args->m_seed;
    if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 engine(seed);

    Network::LayerParam lparam{args->m_leak, args->m_hdegree, args->m_pdegree, args->m_lrank, args->m_prank.data()};
    Network network(nlayers, data->m_nfeatures, data->m_ntargets, layer_types, nio, layer_nio, lparam, &engine); 
    if (msg) network.print();
    int nparams = network.nParams();
    double* params = (double*) malloc(sizeof(double)*nparams);
    network.setParams(params);
    network.Initialize();
    if (!args->m_parmfile.empty()){
      FILE* f = fopen(args->m_parmfile.c_str(), "r");
      assert(f);
      size_t found = fread(params, sizeof(double), nparams, f);
      assert(found == nparams);
      fclose(f); 
    }
    
    string train_log = get_path(args->m_filename.c_str(), "./logs/", to_string(omp_get_thread_num()).c_str());
    TrainingParam tparm{args->m_loss,
                        args->m_ncycles,
                        args->m_batch,
                        args->m_optim,
			args->m_seed,
                        args->m_check,
                        args->m_timeout, 
                        args->m_thresh, 
                        abs(args->m_alpha),
                        args->m_beta1,
                        args->m_beta2,
			msg,
                        train_log.c_str(),
                        args->m_alpha < 0 
                       };
    train(&network, &tparm, ntargets, nfeatures,
          nrecords, x, ldx, ystar, ldystar, 
	  (ntest0 > 0 ? ntest0 : 0), x + nrecords*ldx, ldx, ystar + nrecords*ldystar, ldystar);

    int ldy = network.tOutput();
    double* wrk = (double*) malloc(sizeof(double)*ldy*(data->m_nrecords));
    network.setInputOutput(data->m_nrecords, x, ldx, wrk, ldy);
    double* yn = network.FeedForward();
    if (!is_classification && args->m_scale > 0){
      prepro->unscale(data->m_nrecords, ntargets, ystar, ldystar, scale_vecs);
      prepro->unscale(data->m_nrecords, ntargets, yn, ldy, scale_vecs);
    }
    if (msg && !is_classification){
      if (!ntest){
        cout << "Training Data " << endl << "--------------" << endl;
        regress_prediction(nrecords, ntargets,yn, ldy, ystar, ldystar);
      }
      if (ntest){
        cout << "Test Data " << endl << "--------------" << endl;
        regress_prediction(ntest, ntargets,yn+nrecords*ldy, ldy, ystar+nrecords*ldystar, ldystar);
      }
    } /**/
    Loss* loss = 0;
    switch (args->m_loss){
      case 0: loss = new MSE(ntargets); break;
      case 1: loss = new MAE(ntargets); break;
      case 2: loss = new CE (ntargets); break;
      case 3: loss = new BCE (ntargets); break;
      case 4: loss = new BCEL(ntargets); break;
      case 5: loss = new CEL(ntargets); break;
      case 6: loss = new MSEL(ntargets); break;
      default: assert(0); break;
    }
    if (msg) loss->print();
    double* accuracyi = (double*) malloc(sizeof(double)*args->m_output.size());

    double error;
    if (!ntest || msg){
      loss->setTargets(ystar, ldystar);
      error = loss->Evaluate(nrecords, yn, ldy)/nrecords;
      if (is_mse){ error = sqrt(error); }

      if (msg) cout << "Training Loss " << error << endl; 
      if (is_classification){
        error = 1.e0 - accuracy(nrecords, yn, ldy, ystar, ldystar, args->m_output.size(), data->m_targeti.data(), accuracyi, is_logit, msg&&(!ntest));
        if (msg) mprint("Train Accuracy", args->m_output.size(), 1, accuracyi); 
      }
    }
    
    if (ntest0 > 0){
      loss->setTargets(ystar+nrecords*ldystar, ldystar);
      error = loss->Evaluate(ntest, yn+nrecords*ldy, ldy)/ntest;
      if (is_mse){ error = sqrt(error); }
      if (msg) cout << "Test Loss " << error << endl; 
      if (is_classification){
        error = 1.e0 - accuracy(ntest, yn+nrecords*ldy, ldy, ystar+nrecords*ldystar, ldystar, args->m_output.size(), data->m_targeti.data(), accuracyi, is_logit, msg);
        if (msg) mprint("Test Accuracy", args->m_output.size(), 1, accuracyi); 
      }
    } /**/

    delete loss;
    free(accuracyi);
    free(wrk);
    free(params);
    free(scale_vecs);
    delete prepro;
  
    return error;
  }
//------------------------------------------------------------------------
//                          Cross-Validate
//------------------------------------------------------------------------
double cross_validate(
    const Arguments* args, std::mt19937_64& engine, int npass)
  {
    Field_Labels labels = getFieldLabels(args->m_filename, args->m_label.size(), args->m_label.data());
    //printFieldLabels(labels);
    //read data
    TaskData* data = getTaskData(args->m_filename, labels, args->m_encode, 
	                         args->m_output.size(), args->m_output.data(),
	                         args->m_exclude.size(), args->m_exclude.data());
    //data->print();

    int nrecords = data->m_nrecords;
    int ntargets = data->m_ntargets;
    int nfeatures = data->m_nfeatures;

    int ldystar = data->m_ntargets;
    int ldx = data->m_nfeatures;
    double* x = data->m_records.data(); 
    double* ystar = data->m_targets.data(); 

    int* mix = 0, *perm_rec = 0, *perm_targ = 0; 
    double *perm_wrk = 0;
    int ntest = (args->m_validate > 0 && args->m_validate < 1 ? round(args->m_validate*nrecords) : 0);
    int infer = 1;
    //mix records
    if (ntest) {
      mix = (int*) malloc(sizeof(int)*3*nrecords);
      perm_rec = mix + nrecords;
      perm_targ = perm_rec + nrecords;
      perm_wrk = (double*) malloc(sizeof(double)*max(ntargets, nfeatures));

      for (int i = 0; i < nrecords; ++i) mix[i] = i;
      shuffle(mix, mix + nrecords, engine);
      if (npass <= 0) 
	npass = nrecords/ntest;
      else
	npass = min(nrecords/ntest, npass);
    }
    else{
      npass = 1;
      double validate = abs(args->m_validate);
      ntest = (round)(validate*(validate >= 1 ?  1: nrecords));
      infer = (args->m_validate >= 1 ? -1 : 1);
    }
    //sprint("Cross_validate ntest ", ntest, ", npass ", npass,"\n");

    //cross-validation
    double loss = 0;
    for (int i = 0; i < npass; ++i){ 
      if (mix){ 
	memcpy(perm_rec, mix, sizeof(int)*nrecords);
	if(i < npass-1){
          for (int j = 0; j < ntest; ++j){
	    swap(perm_rec[i*ntest+j], perm_rec[nrecords-ntest+j]);
	  }
        }
        memcpy(perm_targ, perm_rec, sizeof(int)*nrecords);
        permute_inplace(nrecords, nfeatures, x, ldx, perm_rec, perm_wrk);
        if (ntargets) permute_inplace(nrecords, ntargets, ystar, ldystar, perm_targ, perm_wrk);
      }
      loss += run_model(args, data, ntest*infer);
      //sprint("pass ", i+1, "/", npass," loss=", loss/(i+1),"\n");

      //reload
      if (i < npass-1){  
        delete data;
        data = getTaskData(args->m_filename, labels, args->m_encode, 
                           args->m_output.size(), args->m_output.data(),
                           args->m_exclude.size(), args->m_exclude.data());
        x = data->m_records.data(); 
        ystar = data->m_targets.data(); 
      }
    }
    loss /= npass;

    free(mix);
    free(perm_wrk);
    delete data;
    return loss;
  }

//------------------------------------------------------------------------
//                         RUN_FEATURE_SELECTION
//------------------------------------------------------------------------
typedef pair<int, int*> FeatureInfo;

void run_feature_selection(
    Arguments*args, const FeatureInfo& feature_info)
  {
    int seed = args->m_seed;
    if (seed < 0) seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937_64 engine(seed);

    int nfeatures = feature_info.first;
    int* feat2col = feature_info.second;
    mprint("Feature2Col", nfeatures, 1, feat2col);
    NN_Fitness<int8_t> fitness(args, nfeatures, feat2col, &engine);

    int8_t* best = (int8_t*) malloc(sizeof(int8_t)*nfeatures);
    double obj;
    if (args->m_ga == 0){
      BSAO<int8_t> select_features(args->m_natoms, args->m_nepoch, args->m_timeout, args->m_mixrate, args->m_epsilon, args->m_seed);
      obj = select_features(&fitness, best);
    }
    else if (args->m_ga == 1){
      SBMBSA<int8_t> select_features(args->m_natoms, args->m_nepoch, args->m_timeout, args->m_nthreads, args->m_pso_c, args->m_seed);
      obj = select_features(&fitness, best);
    }
    else if (args->m_ga == 2){
      SBPSO<int8_t> select_features(args->m_natoms, args->m_nepoch, args->m_timeout, args->m_nthreads, args->m_pso_c, args->m_seed);
      obj = select_features(&fitness, best);
    }
    else if (args->m_ga == 3){
      BiPSO<int8_t> select_features(args->m_natoms, args->m_nepoch, args->m_timeout, args->m_nthreads, args->m_pso_c, args->m_seed);
      obj = select_features(&fitness, best);
    }
    else if (args->m_ga == 4){
      BiMBSA<int8_t> select_features(args->m_natoms, args->m_nepoch, args->m_timeout, args->m_nthreads, args->m_mixrate, args->m_pso_c, args->m_seed);
      obj = select_features(&fitness, best);
    }
    else{
      assert(0);
    }
    //mprint("Best", nfeatures, 1, best);

    sprint("Objective:", obj, "\n"); 
    fitness.interpret(best);
    fitness.pick_exclude(nfeatures, feat2col, best, args->m_exclude, 0, 0);
    free(best);
    free(feat2col);
  }
