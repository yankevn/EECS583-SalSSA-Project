if [ -z "$1" ]; then
NJOBS=$(nproc)
else
NJOBS=$1
fi
DIR=$(dirname $0)
cd ${DIR}



mkdir -p build
cd build
cmake -DLLVM_ENABLE_PROJECTS='clang;compiler-rt' -DCMAKE_BUILD_TYPE="Release" -DLLVM_ENABLE_ASSERTIONS=On ../llvm-project/llvm
make clang opt llvm-link -j${NJOBS}

