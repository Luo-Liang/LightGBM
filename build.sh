rm build -rf
mkdir -p build
cd build
make clean
cmake .. #-DUSE_MPI=1  .. 
make -j16
