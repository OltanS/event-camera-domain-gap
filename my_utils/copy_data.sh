#!/bin/bash
source config.env

if [ ! -d "${DATA_DIR}" ]; then
    mkdir -p "${DATA_DIR}"
fi

cp -r $PROCESSED_DIR/* $DATA_DIR
