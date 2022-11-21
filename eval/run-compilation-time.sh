DIR=$(dirname $0)
cd ${DIR}

BENCHDIR=$1

BENCHMARKS=$(cat ${BENCHDIR}/BenchNames)

BUILDS="baseline fm fm2"

ERROR=0
for BENCH in ${BENCHMARKS}; do
  cd ${BENCHDIR}/${BENCH}
  echo ${BENCH}
  for VERSION in ${BUILDS}; do
    make clean >/dev/null
    /usr/bin/time -f "${BENCH},${VERSION},%E" sh -c "make ${VERSION} 2>/dev/null 1>/dev/null" 2>> ../compilation.csv
    if [ $? -ne 0 ]; then
      echo "$BENCH [ERROR] : ${VERSION} Compilation"
      ERROR=1
    fi
  done
  cd ../..
done
