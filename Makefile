SRC=./src
OBJ=./obj
EXE = deep_ann
EXE_SMP = deep_ann_smp
EXE_DBG = deep_ann_dbg

CPP = /opt/intel/oneapi/compiler/latest/linux/bin/icpx
CPP_DBG = g++

OBJS = $(OBJ)/main.o 
OBJS_SMP = $(OBJ)/main_smp.o 
OBJS_DBG = $(OBJ)/main_dbg.o 

CSVINC = /home/skostas/Utopia/uDevLib/csv-parser-master/single_include
GSLINC = /usr/include/gsl/
INC = -I$(CSVINC) -I$(GSLINC)

#MKLROOT = /opt/intel/oneapi/mkl/latest
MKLROOT= /opt/intel/mkl
MKL_CFLAGS = -m64 -I"${MKLROOT}/include" 
MKL_LFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group 
CFLAGS = -std=c++17 -c $(MKL_CFLAGS) $(INC) 
LFLAGS = -lgsl $(MKL_LFLAGS) -liomp5 -lpthread -lm -ldl 

$(EXE): $(OBJS) $(OBJS_SMP) $(OBJS_DBG) 
	$(CPP) -O3 -fPIC -static-intel -o $(EXE) $(OBJS) $(LFLAGS) -Bstatic
	$(CPP) -O3 -fPIC -static-intel -qopenmp -o $(EXE_SMP) $(OBJS_SMP) $(LFLAGS) -Bstatic
	$(CPP_DBG) -O3 -o $(EXE_DBG) $(OBJS_DBG) $(LFLAGS) 

$(OBJ)/main.o:$(SRC)/*.cpp 
	$(CPP) -O3 -o $(OBJ)/main.o $(SRC)/main.cpp $(CFLAGS)

$(OBJ)/main_smp.o:$(SRC)/*.cpp 
	$(CPP) -O3 -qopenmp -pthread -o $(OBJ)/main_smp.o $(SRC)/main.cpp $(CFLAGS)

$(OBJ)/main_dbg.o:$(SRC)/*.cpp 
	$(CPP_DBG) -g -o $(OBJ)/main_dbg.o $(SRC)/main.cpp $(CFLAGS)

clean:
	rm $(OBJS) $(EXE) $(EXE_SMP) $(EXE_DBG) $(OBJS_SMP) $(OBJS_DBG)




