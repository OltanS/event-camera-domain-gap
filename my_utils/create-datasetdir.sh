#!/bin/bash

# Creates the dataset dir, creates symlinks to the original dataset
source config.env


echo "Creating $DATA_DIR"
if [ ! -d "${DATA_DIR}" ]; then
    mkdir -p "${DATA_DIR}"
fi


cd $DATA_DIR

echo "Populating $DATA_DIR with symlinks to data from $PROCESSED_DIR"
TODO_FILES=("$PROCESSED_DIR"/*)
for TODO_FILE in "${TODO_FILES[@]}"
do
    ln -s "$TODO_FILE" .
done
echo "Done."