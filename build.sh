mkdir -p build
cd build
cmake -DUSE_MPI=1 -DUSE_GPU=1 .. 
make -j16
cd ..
python postbuild.py
