#!/bin/sh

filename=$1
i=$2
j=$3

pandoc --pdf-engine=xelatex --template=${j}.tex --from=markdown --output="${filename}.pdf" "${i}"
convert -density 200 "${filename}.pdf" "${filename}.jpg"
rm ${filename}.pdf
