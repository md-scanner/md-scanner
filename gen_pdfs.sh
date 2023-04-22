#!/bin/sh
# La Leonardi sarebbe orgogliosa

TEMPLATES_DIR=/usr/share/haskell-pandoc/data/templates

if test ! -d ${TEMPLATES_DIR}; then
	echo No templates dir found
	echo please download the TeX templates to ${TEMPLATES_DIR}
	exit 0
fi

for i in *.md; do
	echo processing ${i}
	# generate PDF using base theme
	pandoc --from=markdown --output=${i}-base.pdf ${i}
	convert -density 200 ${i}-base.pdf ${i}-base.jpg
	rm ${i}-base.pdf
	# generate PDFs using different themes
#	for j in ${TEMPLATES_DIR}/*.tex; do
#		echo generating ${i}-${j%.*}.pdf
#		pandoc --pdf-engine=xelatex --template=${j} --from=markdown --output=${i}-${j%.*}.pdf ${i}
#		convert -density 200 ${i}-${j%.*}.pdf ${i}-${j%.*}.jpg
#	done
done

