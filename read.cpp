//------------------------------------------------------------------------
//                             READSCV
//------------------------------------------------------------------------
class TaskData{
  public:
    vector<double> m_records;
    vector<double> m_targets;
    vector<int> m_targeti;
    int m_nrecords;
    int m_nfeatures;
    int m_ntargets;

    void print(int nvec = 0){
      cout << "nrecords=" << m_nrecords << " "
	   << "nfeatures=" << m_nfeatures << " "
	   << "ntargets=" << m_ntargets << endl;
      if (nvec) {
	if (nvec < 0) nvec = m_nrecords;
        mprint("records", m_nfeatures, nvec, &(m_records[0]));
        mprint("targets", m_ntargets,  nvec, &(m_targets[0]));
      }
    }
  private:
  protected:
  };

typedef map<int, set<string> > Field_Labels;

void printFieldLabels(
    const Field_Labels& labels)
  {
    cout << "Label mapping" << endl << "===================" << endl;
    for (auto field_labels: labels)
    {
      cout << field_labels.first << "-->";
      auto label1 = field_labels.second.begin();
      for (auto label = label1; label != field_labels.second.end(); ++label) 
	cout << *label << "(" << distance(label1, label) << ") ";
      cout << endl;
    }
  }

Field_Labels getFieldLabels(
    const string& filename, int ncateg, const int* categ)
  {
    csv::CSVRow row;
    csv::CSVFormat format; //format.no_header();
//******************ENCODING
    Field_Labels labels;
    csv::CSVReader reader(filename, format);
    bool next = reader.read_row(row);
    do {
      for (int k = 0, i = 0; i < row.size(); ++i){
        bool is_label = !row[i].is_num();
	if (k < ncateg && i+1 == categ[k]) {
	  is_label = true;
	  ++k;
	}
	if (is_label){
	  auto field_labels = labels.find(i);
	  if (field_labels != labels.end()){
	    field_labels->second.insert(row[i].get());
	  }
	  else{
	    labels.insert(make_pair(i, set<string>({row[i].get()})));
	  }
	}
      }
      next = reader.read_row(row);
    }while (next);
    return labels;
  }

TaskData* getTaskData(
    const string& filename, const Field_Labels& labels, int encoding, 
    int noutcols, const int* output_cols, int nexcl, const int* excl)
  {
    if (encoding == 0){
      int mxlabels = 0;
      for (auto field_labels : labels){
        mxlabels = max(mxlabels, (int)field_labels.second.size());
        assert(field_labels.second.size() <= 64); //8 bits/byte* 8 bytes/double
      }
      cout << "Maximum Number of Labels:" << mxlabels << endl;
    }

//******************READING
    TaskData* data = new TaskData;

    csv::CSVRow row;
    csv::CSVFormat format; //format.no_header();
    int irow;

    auto add_value = [&](int i, char what){
      vector<double>& destination = 
	(what == 'F' ? data->m_records : data->m_targets);

      auto field_labels_it = labels.find(i);
      bool is_num = (field_labels_it == labels.end());
      if (is_num){
	assert(row[i].is_num());
        destination.push_back(row[i].get<double>());
      }
      else{
	auto& field_labels = field_labels_it->second;
	const string& label = row[i].get();
	auto label_it = field_labels.find(label);
	if (label_it == field_labels.end()){
	  cout << "Row=" << irow << " Field=" << i+1 << " Undefined Label=" << label << endl;
	  cout << "Labels in Category are:"; 
	  for (auto field_label : field_labels) cout << field_label << " "; cout << endl;
	  assert(label_it != field_labels.end());
	}
	int n = field_labels.size();
	int ibit = distance(field_labels.begin(), label_it);
	if (encoding == 0){
	  destination.push_back(1 << ibit);
	}
	else if (encoding == 1){
	  destination.push_back(ibit);
	} 
	else if (encoding == 2){
	  int sizep = destination.size();
	  for (int j = 0; j < ibit; ++j) destination.push_back(0);
	  destination.push_back(1);
	  for (int j = ibit+1; j < n ; ++j) destination.push_back(0);
	}
	else {
	  assert(0);
	}
      }
    };

    csv::CSVReader reader(filename, format);
    bool next = reader.read_row(row), first = true;
    data->m_targeti.push_back(0);
    irow = 1;
    do{
      int ncols = row.size();
      int k = 0, j = 0, i = 0;
      for (; i < ncols; ++i) {
	if (k < nexcl){
	  if (i+1 == excl[k]) { ++k; continue; } //skip field
	}
	if (j < noutcols){
	  if (i+1 == output_cols[j]){ 
	    add_value(i, 'T'); ++j; 
	    if (first) data->m_targeti.push_back(data->m_targets.size());
	    continue;
	  } //add as target
	} 
	add_value(i, 'F');  //add as feature
      }
      first = false;
      next = reader.read_row(row);
      ++irow;
    }while (next);
    
    data->m_nrecords = reader.n_rows();
    assert(data->m_records.size()%data->m_nrecords == 0);
    assert(data->m_targets.size()%data->m_nrecords == 0);
    data->m_nfeatures = data->m_records.size()/data->m_nrecords;
    data->m_ntargets = data->m_targets.size()/data->m_nrecords;
    assert(data->m_targeti.back() == data->m_ntargets);
    return data;
  }

pair<int, int*> cntFeatures(
    int ncols, int noutcols, const int* output_cols, int nexcl, const int* excl)
  {
    int nfields = ncols;
    int* feat2col = (int*) malloc(sizeof(int)*nfields);
    
    int i = 0, k = 0, j = 0, nfeatures = 0;
    for (; i < ncols; ++i) {
      if (k < nexcl){
        if (i+1 == excl[k]) { ++k; continue; } //skip field
      }
      if (j < noutcols){
        if (i+1 == output_cols[j]){ ++j; continue;} //count target
      }
      feat2col[nfeatures++] = i; //count feature
    }
    return make_pair(nfeatures, feat2col);
  }

int cntCSVColumns(
    const char* filename)
  {
    FILE* csv = fopen(filename, "r");
    assert(csv);
    char* line = 0;
    size_t len = 0;
    ssize_t nread;
    int n_rows = 0;
    int n_cols = 0;
    nread = getline(&line, &len, csv);
    assert(nread > 0);
    char* toks = line;
    while(strtok(toks, ",")){ toks = 0; ++n_cols; }
    free(line);
    fclose(csv);
    return n_cols;
  }
