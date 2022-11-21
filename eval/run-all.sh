DIR=$(dirname $0)
cd ${DIR}

COALESCING=true

runCodeSize() {
  Bench=$1
  echo "Running ${Bench}"
  cp Makefile.config ${Bench}/
  cp Makefile.lto.default ${Bench}/Makefile.default
  for t in 1 5 10; do
    echo "Exploration Threshold: $t"
    sh config.sh ${Bench}/Experiment.config $t ${COALESCING}
    echo "Computing Code Size:"
    :>${Bench}/results.csv
    sh run-code-size.sh ${Bench}
    mkdir -p results/${Bench}/n$t
    cp ${Bench}/results.csv ${Bench}/*.txt results/${Bench}/n$t
  done
  python plot-code-size.py results/${Bench}/
}

runCompileTime() {
  Bench=$1
  echo "Running ${Bench}"
  cp Makefile.config ${Bench}/
  cp Makefile.lto.default ${Bench}/Makefile.default
  for t in 1 5 10; do
    echo "Exploration Threshold: $t"
    sh config.sh ${Bench}/Experiment.config $t ${COALESCING}
    echo "Computing Compilation Time:"
    :>${Bench}/compilation.csv
    sh run-compilation-time.sh ${Bench}
    mkdir -p results/${Bench}/n$t
    cp ${Bench}/compilation.csv results/${Bench}/n$t
  done
  python plot-compilation-time.py results/${Bench}/
}

if [ $# -eq 0 ]; then
BENCHMARKS="spec2006"
else
BENCHMARKS=$*
fi

for Bench in ${BENCHMARKS}; do
  #runCodeSize ${Bench}
  runCompileTime ${Bench}
done

