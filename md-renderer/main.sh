#!/bin/bash

# The prefix used for Docker containers and images (not currently the case) created by this script
DOCKER_PREFIX=md-renderer
OUTPUT_IMAGE_DENSITY=200

if test $# -lt 1; then
    echo Usage: main.sh DATASET_DIR
    exit 1
fi

# The directory where the Markdown files to render are retrieved
DATASET_DIR=$1

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

JEKYLL_PORT=4000

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

source $BASE_DIR/.venv/bin/activate
pip install -r "$BASE_DIR/requirements.txt"

# https://github.com/envygeeks/jekyll-docker/blob/master/README.md

JEKYLL_SITE_DIR=$BASE_DIR/jekyll/_site

# Clean Jekyll's src directory
rm -f $BASE_DIR/jekyll/*.md

# Clean Jekyll's dist directory
rm -f $JEKYLL_SITE_DIR/*.html
rm -f $JEKYLL_SITE_DIR/*.md

# Move non-rendered Markdown files to the Jekyll source directory
I=1
#COUNT=
for MD_FILE in $DATASET_DIR/*.md
do
    if [[ -f "${MD_FILE%.md}.jpg" || -f "${MD_FILE%.md}.html" || -f "${MD_FILE%.md}.pdf" ]]
    then
        continue
    fi

    echo -n "$I Pre-processing $MD_FILE for Jekyll..."

    MD_FILENAME_WITH_EXT=$(basename -- $MD_FILE)
    MD_OUT_FILE=$BASE_DIR/jekyll/$MD_FILENAME_WITH_EXT
    cp $MD_FILE $MD_OUT_FILE

    python3 $BASE_DIR/scripts/clean_md_for_jekyll.py $MD_OUT_FILE

    echo " Done!"

    (( I++ ))
done

# TODO iterate over all the themes
for theme in  ${THEMES[@]}; do
    echo Generating with theme $theme

    # Generate Jekyll config
    cat > $BASE_DIR/jekyll/_config.yml << EOF
theme: $theme

repository: MD-Scanner/$theme
author: Filippo Marinelli, Lorenzo Rutayisire and Carmine Zaccagnino

render_with_liquid: false
EOF

    # ------------------------------------------------------------------------------------------------
    # Markdown to Html
    # ------------------------------------------------------------------------------------------------
    # Render the Markdown files to HTML
    docker run \
        --rm \
        --volume $BASE_DIR/jekyll:/srv/jekyll \
        --volume $BASE_DIR/.jekyll-cache/bundle:/usr/local/bundle \
        --volume $BASE_DIR/.jekyll-cache/gem:/usr/gem \
        jekyll/jekyll \
        jekyll build

    # ------------------------------------------------------------------------------------------------
    # Markdown to BoundingBox-Html (or BB-Html)
    # ------------------------------------------------------------------------------------------------

    I=1
    COUNT=$(ls $JEKYLL_SITE_DIR/*.html | wc -w)
    for HTML_FILE in $JEKYLL_SITE_DIR/*.html
    do
        HTML_FILENAME_WITH_EXT=$(basename -- $HTML_FILE)
        DOC_NAME="${HTML_FILENAME_WITH_EXT%.html}"

        BB_HTML_FILE="${HTML_FILE%.html}-bb.html"

        echo -n "$I/$COUNT Generating Bb-Html for $DOC_NAME... "

        cp $HTML_FILE $BB_HTML_FILE

        cat >> $BB_HTML_FILE <<- EOM
<style>
    h1 { outline: 3px solid #FF0000 !important; }
    h2 { outline: 3px solid #FF00FF !important; }
    h3 { outline: 3px solid #AA10FF !important; }
    h4 { outline: 3px solid #10BBFF !important; }
    h5 { outline: 3px solid #00FFFF !important; }
    h6 { outline: 3px solid #0090FF !important; }
    img { outline: 3px solid #50AA90 !important; }
    p { outline: 3px solid #0000FF !important; }
    pre { outline: 3px solid #00FF00 !important; }
    ul { outline: 3px solid #FFFF00 !important; }
    ol { outline: 3px solid #FFC0CB !important; }
</style>
EOM
        echo "Done!"

        (( I++ ))
    done

    # ------------------------------------------------------------------------------------------------
    # Html to PDF to JPGs
    # ------------------------------------------------------------------------------------------------

    # Start Jekyll webserver so that we can access its pages through Selenium to save the PDFs
    docker rm --force $DOCKER_PREFIX-jekyll-server
    docker run \
        --rm \
        --detach \
        --name $DOCKER_PREFIX-jekyll-server \
        --volume $BASE_DIR/jekyll:/srv/jekyll \
        --volume $BASE_DIR/.jekyll-cache/bundle:/usr/local/bundle \
        --volume $BASE_DIR/.jekyll-cache/gem:/usr/gem \
        --publish $JEKYLL_PORT:$JEKYLL_PORT \
        jekyll/jekyll \
        jekyll serve --skip-initial-build --port $JEKYLL_PORT

    # Wait for Jekyll to be listening on that port
    echo -n "Waiting for Jekyll server to be available..."

    while ! nc -z localhost $JEKYLL_PORT
    do
    sleep 0.1
    echo -n "."
    done

    sleep 0.1

    echo " Done!"

    # For every HTML file we have on the site, we access it through Selenium and save the PDF
    I=1
    COUNT=$(ls $JEKYLL_SITE_DIR/*.html | wc -w)
    for HTML_FILE in $JEKYLL_SITE_DIR/*.html
    do
        HTML_FILENAME_WITH_EXT=$(basename -- $HTML_FILE)
        
        PDF_FILE="${HTML_FILE%.html}-$theme.pdf"
        JPG_FILE="${HTML_FILE%.html}-$theme.jpg"
        PNG_FILE="${HTML_FILE%.html}-$theme.png"

        JEKYLL_DOC_URL="http://localhost:$JEKYLL_PORT/$HTML_FILENAME_WITH_EXT"

        echo -n "$I/$COUNT Taking PDF of Html $HTML_FILENAME_WITH_EXT from $JEKYLL_DOC_URL... "
        
        python3 $BASE_DIR/scripts/convert_web_page_to_pdf.py $JEKYLL_DOC_URL "$PDF_FILE"
        
        echo "Done!"

        echo -n "$I/$COUNT Converting PDF to image(s)... "
        # ------------------------------------------------------------------------------------------------
        # PDF to PNG/JPG
        # ------------------------------------------------------------------------------------------------

        # check whether we are processing a bb or non-bb file and pick output format accordingly
        if [[ $HTML_FILE =~ \-bb.html$ ]]; then
            convert -density $OUTPUT_IMAGE_DENSITY $PDF_FILE $PNG_FILE
        else
            convert -density $OUTPUT_IMAGE_DENSITY $PDF_FILE $JPG_FILE
        fi

        echo "Done!"

        ((I++))
    done

    cp $JEKYLL_SITE_DIR/*.jpg $DATASET_DIR
    cp $JEKYLL_SITE_DIR/*.png $DATASET_DIR

    # Finally we forcefully stop the Jekyll server by deleting its container
    docker rm --force $DOCKER_PREFIX-jekyll-server
done


deactivate
