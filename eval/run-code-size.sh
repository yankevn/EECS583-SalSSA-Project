DIR=$(dirname $0)
cd ${DIR}

BENCHDIR=$1

BENCHMARKS=$(cat ${BENCHDIR}/BenchNames)

BUILDS="fm2 fm baseline"

ERROR=0
for BENCH in ${BENCHMARKS}; do
  cd ${BENCHDIR}/${BENCH}
  echo ${BENCH}
  make clean >/dev/null
  for VERSION in ${BUILDS}; do
    make ${VERSION} 2>../${BENCH}.${VERSION}.txt 1>/dev/null
    if [ $? -ne 0 ]; then
      echo "$BENCH [ERROR] : ${VERSION} Compilation"
      ERROR=1
    fi
  done
  cd ../..
  if [ $ERROR -eq 0 ]; then
    python results.py ${BENCHDIR}/${BENCH} >> ${BENCHDIR}/results.csv
  fi
done
