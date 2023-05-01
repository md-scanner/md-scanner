#!/bin/bash

# The directory where the Markdown files to render are retrieved
DATASET_DIR=/home/rutayisire/unimore/cv/md-scanner/dataset

BASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
THEMES=(
    "jekyll-theme-architect"
    "jekyll-theme-cayman"
    "jekyll-theme-dinky"
    "jekyll-theme-hacker"
    "jekyll-theme-leap-day"
    "jekyll-theme-merlot"
    "jekyll-theme-midnight"
    "jekyll-theme-minimal"
    "jekyll-theme-modernist"
    "jekyll-theme-primer"
    "jekyll-theme-slate"
    "jekyll-theme-tactile"
    "jekyll-theme-time-machine"
    )

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

# https://github.com/envygeeks/jekyll-docker/blob/master/README.md

JEKYLL_SITE_DIR=$BASE_DIR/jekyll/_site

# Clean Jekyll's src directory
rm $BASE_DIR/jekyll/*.md

# Clean Jekyll's dist directory
rm $JEKYLL_SITE_DIR/*.html
rm $JEKYLL_SITE_DIR/*.md

# Move non-renderized Markdown files to the Jekyll source directory
for MD_FILE in $DATASET_DIR/*.md
do
    if [[ -f "${MD_FILE%.md}.jpg" || -f "${MD_FILE%.md}.html" || -f "${MD_FILE%.md}.pdf" ]]
    then
        continue
    fi

    echo "Processing \"$MD_FILE\"..."

    MD_FILENAME_WITH_EXT=$(basename -- $MD_FILE)
    MD_OUT_FILE=$BASE_DIR/jekyll/$MD_FILENAME_WITH_EXT
    cp $MD_FILE $MD_OUT_FILE

    python3 $BASE_DIR/clean_md_for_jekyll.py $MD_OUT_FILE

    echo "Cleaned \"$MD_OUT_FILE\""
done

# Render the Markdown files to HTML
docker run \
    --rm \
    --volume $BASE_DIR/jekyll:/srv/jekyll \
    --volume $BASE_DIR/.jekyll-cache/bundle:/usr/local/bundle \
    --volume $BASE_DIR/.jekyll-cache/gem:/usr/gem \
    jekyll/jekyll \
    jekyll build

for HTML_FILE in $JEKYLL_SITE_DIR/*.html
do
    cat >> $HTML_FILE <<- EOM
<style>
    h1,h2,h3,h4,h5,h6 { border: 3px solid red !important; }
    p { border: 3px solid blue !important; }
    pre { border: 3px solid green !important; }
    * {
      color: transparent !important;
      text-shadow: none !important;
      text-decoration: none !important;
      background: black !important;
      border: none !important;
    }
</style>
EOM
    echo "BoundingBox style injected in \"$HTML_FILE\""
done

# TODO iterate over all the themes

docker run \
    --rm \
    --volume $BASE_DIR/jekyll:/srv/jekyll \
    --volume $BASE_DIR/.jekyll-cache/bundle:/usr/local/bundle \
    --volume $BASE_DIR/.jekyll-cache/gem:/usr/gem \
    --publish [::1]:4000:4000 \
    jekyll/jekyll \
    jekyll serve --skip-initial-build

# TODO Visit every generated HTML web-page using an headless browser and convert it to PDF
# TODO Convert each PDF to Jpg

