#!/bin/sh

if test ! -f ./extract-bbs; then
    echo "Please compile extract-bbs.cpp to extract-bbs"
    echo "Run the following:"
    echo "cmake .; make"

    exit 1
fi

if test $# -ne 1 -o ! -d $1; then
    echo "Bad parameters"
    echo "Usage: ./main.sh DATASET_PATH"
    echo "Where DATASET_PATH is the path to a directory containing PNG images with bouding boxes"

    exit 2
fi

for file in $1/*.png; do
    echo "processing $file"
    ./extract-bbs $file
done
