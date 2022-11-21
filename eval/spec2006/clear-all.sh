DIR=$(dirname $0)
cd ${DIR}

if [ $# -eq 0 ]; then
BENCHMARKS="473.astar 433.milc 462.libquantum 470.lbm 401.bzip2 450.soplex 471.omnetpp 482.sphinx3 400.perlbench 447.dealII 464.h264ref 456.hmmer 453.povray 445.gobmk 403.gcc 444.namd 429.mcf 458.sjeng 483.xalancbmk"
else
BENCHMARKS=$*
fi

for BENCH in ${BENCHMARKS}; do
  cd ${BENCH}
  make clean >/dev/null
  cd ..
done
