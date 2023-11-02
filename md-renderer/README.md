# MD-renderer

This is the tool responsible of converting an input Markdown file to an output HTML file with a given theme.

We've chosen to use Jekyll instead of Pandoc for the wide variety of themes it offers. Although the usage of a whole
Static Site Generator would probably affect the performance of the conversion.

## Usage

```
sudo bash ./main.sh <md-dataset-dir>
```

Where `md-dataset-dir` is the directory containing the .md files to renderer into images.

System requirements:
- docker
- imagemagick
