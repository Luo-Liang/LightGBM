mkdir -p build
cd build
cmake -DUSE_MPI=ON ..
make -j$(nproc)
python postbuild.py
