#!/bin/bash

# run me from the main directory
cd my_utils/csv_files
# Check if both files exist
if [[ ! -f all-files.csv ]] || [[ ! -f recording-start-stop.csv ]]; then
    echo "Error: Both all-files.csv and recording-start-stop.csv must exist."
    exit 1
fi

# Extract the first column from recording-start-stop.csv and filter all-files.csv lines not present in it
echo "Following files don't have a corresponding entry in recording-start-stop.csv"
grep -Fvx -f <(awk -F, '{print $1}' recording-start-stop.csv) all-files.csv

echo "Following files in recording-start-stop.csv have a -1 in the second column (only if present in all-files.csv):"
awk -F, '
    BEGIN {
        while ((getline line < "all-files.csv") > 0) {
            files[line] = 1
        }
        close("all-files.csv")
    }
    $2 == "-1" && ($1 in files) { 
        print $1 
    }
' recording-start-stop.csv