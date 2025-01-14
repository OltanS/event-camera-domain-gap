#!/bin/bash
# Halt if any command returns a non-zero status
set -e

CONFIG_FILE=config.env

source $CONFIG_FILE
# Delete everything in OUT_DIR
rm -rf "${OUT_DIR:?}"/*

BINSIZE_S=$(echo "scale=3; $BINSIZE_MS / 1000" | bc)
TODO_FILES=("$IN_DIR"/*)

# Activate the virtual environment
source ${VENV_DIR}/bin/activate

# Here on down should not require modification
for TODO_FILE in "${TODO_FILES[@]}"
do
    echo "### Working on $TODO_FILE ####"
    FILENAME=$(basename "$TODO_FILE")

    TSTART=$(awk -F, -v f="$FILENAME" '$1 == f {print $2}' "$CSV_FILE")
    TSTOP=$(awk -F, -v f="$FILENAME" '$1 == f {print $3}' "$CSV_FILE")

    OUT_FILE_NO_EXTENSION="${OUT_DIR}/${FILENAME%.*}"
    OUT_FILE_FULLPATH="${OUT_FILE_NO_EXTENSION}_bin${BINSIZE_MS}ms.hdf5"
    # Export data
    echo "### Exporting... ###"
    echo "Running command: python2 ${DDD20_UTILS_DIR}/export.py ${TODO_FILE} --binsize ${BINSIZE_S} --export_aps ${EXPORT_APS} --export_dvs ${EXPORT_DVS} --out_file ${OUT_FILE_FULLPATH} --tstart ${TSTART} --tstop ${TSTOP}"
    python2 ${DDD20_UTILS_DIR}/export.py ${TODO_FILE} --binsize ${BINSIZE_S} --export_aps ${EXPORT_APS} --export_dvs ${EXPORT_DVS} --out_file ${OUT_FILE_FULLPATH} --tstart ${TSTART} --tstop ${TSTOP}

    # Prepare and resize
    # This works inplace
    python2 ${DDD20_UTILS_DIR}/prepare_cnn_data.py --filename ${OUT_FILE_FULLPATH} --rewrite 1 --skip_mean_std 1 --new_height ${HEIGHT} --new_width ${WIDTH}
done

# If needed create further directories for JUST DVS and JUST APS.