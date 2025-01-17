#!/bin/bash

# Creates the dataset dir, creates symlinks to the original dataset
source config.env

echo "Creating $IN_DIR"
if [ ! -d "${IN_DIR}" ]; then
    mkdir -p "${IN_DIR}"
fi

cd $IN_DIR

echo "Populating $IN_DIR with symlinks to data from $DATA_ORIGIN_DIR"
TODO_FILES=("$DATA_ORIGIN_DIR"/*)
for TODO_FILE in "${TODO_FILES[@]}"
do
    ln -s "$TODO_FILE" .
done
echo "Done."