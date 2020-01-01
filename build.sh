rm ./lightgbm
mkdir -p build
cd build
cmake -DUSE_MPI=ON ..
make -j$(nproc)
cd ..
python ./postbuild.py
