#!/bin/bash
# Halt if any command returns a non-zero status
set -e

# Parse command line arguments for rotate
ROTATE=""
while getopts "r:" opt; do
    case "$opt" in
        r) ROTATE="$OPTARG" ;;
        *) echo "Usage: $0 [-r rotate_value]"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

CONFIG_FILE=config.env
source $CONFIG_FILE

# Activate the virtual environment
source ${VENV_DIR}/bin/activate

# sort the files alphabetically and exclude those listed in recording-start-stop.csv
EXCLUDED_FILES=$(cut -d',' -f1 $CSV_FILE)
echo "Excluded files:"
echo "$EXCLUDED_FILES"

TODO_FILES=($(ls "$IN_DIR" | grep -vxF -f <(echo "$EXCLUDED_FILES") | sort))

for f in "${TODO_FILES[@]}"; do
    read -p "Press Enter to load $f..." < /dev/tty
    if [ -n "$ROTATE" ]; then
        python "${DDD20_UTILS_DIR}/view.py" "$IN_DIR/$f" --rotate "$ROTATE"
    else
        python "${DDD20_UTILS_DIR}/view.py" "$IN_DIR/$f"
    fi
    read -p "Enter start time: " start_time
    read -p "Enter end time: " end_time
    if [ -n "$ROTATE" ]; then
        read -p "WARNING: The view has been rotated. Enter 1 if recording appears flipped (i.e., it's the right way around), 0 otherwise: " flipped
    else
        read -p "1 if recording is flipped, 0 otherwise: " flipped
    fi
    echo "$f,$start_time,$end_time,$flipped" >> $CSV_FILE
done

deactivate