#!/bin/bash

dir="$1"
fails=""
for path in "$dir"/*; do
    if [[ "$path" == *".mtx" ]]; then
        echo "processing $path"
        python convert_petsc.py "$path"
        if [[ "$?" -ne 0 ]]; then
            echo "$path failed"
            fails="${fails}${path} failed"$'\n'
        else
            echo "successfully processed $path"
        fi
    fi
done

echo "$fails"