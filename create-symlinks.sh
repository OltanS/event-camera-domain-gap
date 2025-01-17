IN_DIR="/mnt/c/users/z5230739/OneDrive - UNSW/H0425391/DDD17"
TODO_FILES=("$IN_DIR"/*)

for TODO_FILE in "${TODO_FILES[@]}"
do
    ln -s "$TODO_FILE" .
done

