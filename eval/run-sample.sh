DIR=$(dirname $0)
cd ${DIR}

echo "Running SPEC2006"

for t in 1 5 10; do
  echo "Exploration Threshold: $t"
  sh config.sh spec2006/Experiment.config $t
  echo "Computing Code Size:"
  :>spec2006/results.csv
  sh spec2006/run-code-size.sh 473.astar 433.milc 462.libquantum
  mkdir -p results/spec2006/n$t

  cp spec2006/results.csv results/spec2006/n$t
done
python plot-code-size.py results/spec2006/

