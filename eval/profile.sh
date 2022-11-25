DIR=$(dirname $0)
cd ${DIR}

BENCHDIR=spec2006

# TODO: create input data for other benchmarks
if [ $# -eq 0 ]; then
BENCHMARKS="462.libquantum"
else
BENCHMARKS=$*
fi

generateProfile() {
  Bench=$1
  echo "Profiling ${Bench}"
  make clean >/dev/null

  make profile 2>profile.txt 1>/dev/null
  if [ $? -ne 0 ]; then
    echo "$Bench [ERROR] : Profile Generation"
    # ERROR=1
  fi
}

cp Makefile.config ${BENCHDIR}/
cp Makefile.lto.default ${BENCHDIR}/Makefile.default
cd ${BENCHDIR}

for BENCH in ${BENCHMARKS}; do
  cd ${BENCH}
  generateProfile ${BENCH}
  cd ..
done
