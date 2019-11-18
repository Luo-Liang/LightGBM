rm build -rf
mkdir -p build
cd build
make clean
cmake -DUSE_MPI=1  ..
#cmake ..
make -j$(nproc) > /dev/null
cd ..
python postbuild.py
OUTPUT=`date; hostname;`
echo $OUTPUT
