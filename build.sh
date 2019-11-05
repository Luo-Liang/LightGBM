rm build -rf
mkdir -p build
cd build
make clean
cmake -DUSE_SOCKET=1  ..
#cmake ..
make -j$(nproc)
cd ..
python postbuild.py
