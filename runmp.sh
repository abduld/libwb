if [ $# -ne 1 ]
then
    echo "usage: $0 <mp number>" 1>&2
    exit 1
fi

MP=$1
TARGET=build/Debug/MP${MP}.exe

if [ ! -x ${TARGET} ]
then
    echo "unable to find executable ${TARGET}" 1>&2
    exit 1
fi

DATADIR=data/MP${MP}/data

if [ ! -x ${DATADIR} ]
then
    echo "unable to find directory ${DATADIR}" 1>&2
    exit 1
fi

case ${MP} in
    0|1|5)
        TYPE=vector
        ;;
    2|3)
        TYPE=matrix
        ;;
    4)
        TYPE=image
        ;;
    *)
        echo "unknonn type for MP${MP}" 1>&2
        exit 1
        ;;
esac

for dataset in ${DATADIR}/*
do
    echo "executing with dataset ${dataset}"
    inputs=`ls ${dataset}/input* 2>/dev/null | xargs echo | sed -e "s/  */,/g"`
    expected=`ls ${dataset}/output* 2>/dev/null `

    if [ -z "${inputs} ]
    then
        inputs=none
    fi

    if [ -z "${expected} ]
    then
        expected=none
    fi

    echo ${TARGET} -e ${expected} -i ${inputs} -o ${dataset}/result.raw -t ${TYPE}
    ${TARGET} -e ${expected} -i ${inputs} -o ${dataset}/result.raw -t ${TYPE}
done
