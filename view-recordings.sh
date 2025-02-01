#!/bin/bash
# Halt if any command returns a non-zero status
set -e
CONFIG_FILE=config.env
source $CONFIG_FILE

# Activate the virtual environment
source ${VENV_DIR}/bin/activate

CSV_FILE="recording-start-stop.csv"
# sort the files alphabetically and exclude those listed in recording-start-stop.csv
EXCLUDED_FILES=$(cut -d',' -f1 $CSV_FILE)
echo "Excluded files:"
echo "$EXCLUDED_FILES"

TODO_FILES=($(ls "$IN_DIR" | grep -vxF -f <(echo "$EXCLUDED_FILES") | sort))

for f in "${TODO_FILES[@]}"; do
    read -p "Press Enter to load $f..." < /dev/tty
    python "${DDD20_UTILS_DIR}/view.py" "$IN_DIR/$f"
    read -p "Enter start time: " start_time
    read -p "Enter end time: " end_time
    read -p "1 if recording is flipped, 0 otherwise: " flipped
    echo "$f,$start_time,$end_time,$flipped" >> $CSV_FILE
    done

deactivate