#!/bin/bash
# Halt if any command returns a non-zero status
set -e

CONFIG_FILE=config.env
source $CONFIG_FILE
DIRNAME="$TMPDIR/$PACKAGE_NAME"

DDD20_UTILS_DIR=$DIRNAME/$DDD20_UTILS_DIR
VENV_DIR=$DIRNAME/$VENV_DIR
CSV_FILE=$DIRNAME/$CSV_FILE


if [ ! -d "${OUT_DIR}" ]; then
    mkdir -p "${OUT_DIR}"
fi

BINSIZE_S=$(awk "BEGIN {printf \"%.3f\", $BINSIZE_MS / 1000}")
TODO_FILES=("$IN_DIR"/*.hdf5)

# Activate the virtual environment
source ${VENV_DIR}/bin/activate

for TODO_FILE in "${TODO_FILES[@]}"
do
    echo "### Working on $TODO_FILE ####"
    FILENAME=$(basename "$TODO_FILE")

    # pull the tstart and tstop from recording-start-stop.csv
    TSTART=$(awk -F, -v f="$FILENAME" '$1 == f {print $2}' "$CSV_FILE")
    TSTOP=$(awk -F, -v f="$FILENAME" '$1 == f {print $3}' "$CSV_FILE")    
    # Skip the file if TSTART is -1
    if [ "$TSTART" == "-1" ]; then
        echo "Skipping $TODO_FILE because TSTART is -1, marking unviable recording"
        continue
    fi
    
    OUT_FILE_NO_EXTENSION="${OUT_DIR}/${FILENAME%.*}"
    OUT_FILE_FULLPATH="${OUT_FILE_NO_EXTENSION}_bin${BINSIZE_MS}ms.hdf5"
    
    # Check if output file already exists
    if [ -f "${OUT_FILE_FULLPATH}" ]; then
        echo "Output file ${OUT_FILE_FULLPATH} already exists. Skipping export for $TODO_FILE."
        continue
    fi
    
    # Export data
    echo "### Exporting... ###"
    echo "Running command: python2 ${DDD20_UTILS_DIR}/export.py ${TODO_FILE} --binsize ${BINSIZE_S} --export_aps ${EXPORT_APS} --export_dvs ${EXPORT_DVS} --out_file ${OUT_FILE_FULLPATH}" #  --tstart ${TSTART} --tstop ${TSTOP}
    python2 ${DDD20_UTILS_DIR}/export.py ${TODO_FILE} --binsize ${BINSIZE_S} --export_aps ${EXPORT_APS} --export_dvs ${EXPORT_DVS} --out_file ${OUT_FILE_FULLPATH} # --tstart ${TSTART} --tstop ${TSTOP}

    # Prepare and resize
    # This works inplace
    python2 ${DDD20_UTILS_DIR}/prepare_cnn_data.py --filename ${OUT_FILE_FULLPATH} --rewrite 1 --skip_mean_std 1 --new_height ${HEIGHT} --new_width ${WIDTH}
done

# If needed create further directories for JUST DVS and JUST APS.