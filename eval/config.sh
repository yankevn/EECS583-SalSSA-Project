DIR=$(dirname $0)
cd ${DIR}
DIR=$(pwd)

:>$1
echo "LLPATH=${DIR}/../build/bin" >>$1
echo "FMEXPLORATION=$2" >>$1
echo "SALSSA_COALESCING=$3" >>$1
