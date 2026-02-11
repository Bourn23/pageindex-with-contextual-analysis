#!/bin/bash
paper_dir="$1"
paper_name=$(basename "$paper_dir")

if [[ "$paper_name" == *"fail"* ]]; then
    echo "Simulating failure for $paper_name"
    exit 1
else
    echo "Simulating success for $paper_name"
    exit 0
fi
