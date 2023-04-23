import github
from github import Github
from dotenv import load_dotenv
import os
import re
from colorama import Fore
import time
import math
import random
import json
from urllib.parse import urlparse


load_dotenv()


class InvalidMdFile(Exception):
    pass


class DatasetRetriever:
    def __init__(self,
                 access_token: str,
                 output_directory: str,
                 retrieval_interval: int  # In seconds
                 ):
        self.per_page = 30
        self.g = Github(access_token, per_page=self.per_page)

        self.output_directory = output_directory
        self.last_retrieve_at = None
        self.retrieval_interval = retrieval_interval
        
        self.paginated_list = self.g.search_code(query="filename:.md")
        self.num_pages = math.ceil(self.paginated_list.totalCount / self.per_page)

        self.current_page = []
        self.current_page_idx = 0
        self.current_page_offset = 0
        self.set_page(random.randint(0, min(100, self.num_pages)))

        self.processed_file_keys = set({})

        self.num_samples = 0


    def try_to_create_directory(self):
        try:
            os.mkdir(self.output_directory)
        except:
            pass

    
    def set_page(self, page_idx: int):
        print(f"Fetching page {page_idx + 1} of {self.num_pages}, totalCount={self.paginated_list.totalCount}...")

        self.current_page = self.paginated_list.get_page(page_idx)
        self.current_page_idx = page_idx
        self.current_page_offset = 0


    def assert_md_file_content(self, md_file_content: str):
        # Throws an exception if the given MD file contains a non-ASCII character. Supposing the MD file 
        # is UTF-8 encoded, a non-ASCII character will be a character whose UTF-8 binary representation
        # has more than one byte
        for char in md_file_content:
            if len(char.encode('utf-8')) > 1:
                raise InvalidMdFile(f"Non-ASCII character found: {char}")
        
        # Throws an exception if the given MD file contains an HTML tag. This is quite common
        # in Github's Markdown files
        # NOTE: At the moment we ignore whether the tag is found inside an HTML code section. Instead,
        # we should accept MD files whose HTML tags are found inside a ``` ... ``` section
        match = re.search(r"<([^\s]+)(?:\s[^>]*)?>(?:.|\n)*?<\/\1>", md_file_content)
        if match != None:
            raise InvalidMdFile(f"HTML tags found: {match.group(0)}")


    def make_image_urls_absolute(self, md_file, md_file_content: str):
        image_url_regex = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

        offset = 0
        for match in image_url_regex.finditer(md_file_content):
            image_url = match.group(1)
            match_start, match_end = match.span(1)

            if image_url.startswith("http://") or image_url.startswith("https://"):
                continue

            parsed_url = urlparse(md_file.download_url)
            abs_image_url = parsed_url._replace(path=os.path.dirname(parsed_url.path) + "/" + image_url).geturl()

            md_file_content = md_file_content[:match_start + offset] + abs_image_url + md_file_content[match_end + offset:]
            offset += len(abs_image_url) - len(image_url)

        return md_file_content


    def save_md_file(self, md_file, processed_content):
        filename = hex(hash(processed_content))[2:]

        desc_path = os.path.join(self.output_directory, filename + ".json")
        md_path = os.path.join(self.output_directory, filename + ".md")

        if os.path.exists(desc_path) or os.path.exists(md_path):
            raise InvalidMdFile(f"Filename \"{filename}\" is already stored")

        # Store the descriptor file
        with open(desc_path, "w") as f:
            raw_data = md_file.raw_data
            del raw_data['content']

            f.write(json.dumps(raw_data, indent=2))

        # Store the MD file
        with open(md_path, "wb") as f:
            f.write(processed_content.encode('utf-8'))


    def retrieve_one(self):
        if self.current_page_offset >= len(self.current_page):
            self.set_page(self.current_page_idx + 1)

        md_file = self.current_page[self.current_page_offset]
        self.current_page_offset += 1

        print(f"{self.num_samples}. Processing {Fore.CYAN}{md_file.repository.full_name}{Fore.RESET} {Fore.LIGHTWHITE_EX}{md_file.path}{Fore.RESET}... ", end='')

        file_key = md_file.repository.name + "-" + md_file.path
        if file_key in self.processed_file_keys:  # We have already processed this file, it's likely coming from a fork
            print(f" {Fore.YELLOW}Skipped!{Fore.RESET}")
            return False

        try:
            # Roughly check that the file wasn't processed yet by checking its path and the repository name.
            # This is done to avoid forks of the same repository
            if file_key in self.processed_file_keys:
                raise InvalidMdFile(f"Already processed")
            self.processed_file_keys.add(file_key)

            processed_content = md_file.decoded_content.decode('utf-8')

            processed_content = self.make_image_urls_absolute(md_file, processed_content)

            self.assert_md_file_content(processed_content)

            self.save_md_file(md_file, processed_content)

            print(f" {Fore.GREEN}Saved!{Fore.RESET}")

            self.num_samples += 1
            return True

        except InvalidMdFile as e:
            print(f" {Fore.RED}Invalid MD-file: {e}{Fore.RESET}")
            return False


    def retrieve(self):
        self.try_to_create_directory()

        while True:
            now = time.time()

            # If the elapsed time since last retrieval is higher than the minimum threshold, wait
            # and then check again
            if self.last_retrieve_at != None and (now - self.last_retrieve_at) < self.retrieval_interval:
                if self.last_retrieve_at != None:
                    time.sleep(self.retrieval_interval - (now - self.last_retrieve_at))
                continue

            self.last_retrieve_at = now

            try:
                self.retrieve_one()
            except github.RateLimitExceededException as _:
                return self.num_samples


def main(output_directory: str):
    dataset_retriever = DatasetRetriever(
        os.environ["GITHUB_ACCESS_TOKEN"],
        output_directory,
        0.5
        )
    num_samples = dataset_retriever.retrieve()
    print(f"Retrieved a total of: {num_samples}")


if __name__ == "__main__":
    main("tmp")

