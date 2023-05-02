import re
import sys


def clean_frontmatter(md_content: str):
    regex_pattern = re.compile(r"---(.|\n)*?---", re.MULTILINE)
    
    res = re.sub(regex_pattern, "", md_content)
    return res


def clean_liquid_like_syntax(md_content: str):
    regex_list = [
        re.compile(pattern, re.MULTILINE) for pattern in [
            r"{{(.|\n)*?}}",
            r"{{",
            r"}}",
            r"{%(.|\n)*?%}"
        ]
    ]
    
    for regex in regex_list:
        md_content = re.sub(regex, "", md_content)
    return md_content


def main(md_file: str):
    with open(md_file, "rt") as f:
        md_content = f.read()
    
    md_content = clean_frontmatter(md_content)
    md_content = clean_liquid_like_syntax(md_content)

    with open(md_file, "wt") as f:
        f.write(md_content)


if __name__ == "__main__":
    main(sys.argv[1])
