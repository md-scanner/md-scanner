#!/bin/sh
# La Leonardi sarebbe orgogliosa

TEMPLATES_DIR=/usr/share/haskell-pandoc/data/templates



if test ! -d ${TEMPLATES_DIR}; then
	echo No templates dir found
	echo please download the TeX templates to ${TEMPLATES_DIR}
	exit 0
fi

for i in *.md; do
	if test ! -f ${i}-base.pdf; then
		echo processing ${i}
		# generate PDF using base theme
		pandoc --pdf-engine=xelatex --template=./base.tex --from=markdown --output=${i}-base.pdf ${i}
		convert -density 200 ${i}-base.pdf ${i}-base.jpg
		rm ${i}-base.pdf
		# generate PDFs using different themes
		for j in ${TEMPLATES_DIR}/*.tex; do
			j="$(basename ${j})"
			j=${j%.*}
			filename="${i}-${j}"
			echo generating "${filename}.pdf"
			./generate_file.sh "${filename}" "${i}" "${j}" &
		done

	fi
done

