# https://github.com/kumaF/pyhtml2pdf/blob/master/pyhtml2pdf/converter.py

import json
import base64
from time import time
import sys

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService

def convert(
    source: str,
    target: str,
    print_options: dict = {},
):
    """
    Convert a given html file or website into PDF

    :param str source: source html file or website link
    :param str target: target location to save the PDF
    :param bool compress: whether PDF is compressed or not. Default value is False
    :param int power: power of the compression. Default value is 0. This can be 0: default, 1: prepress, 2: printer, 3: ebook, 4: screen
    :param dict print_options: options for the printing of the PDF. This can be any of the params in here:https://vanilla.aslushnikov.com/?Page.printToPDF
    """

    result = __get_pdf_from_html(
        source, print_options)
    
    with open(target, "wb") as file:
        file.write(result)


def __send_devtools(driver, cmd, params={}):
    resource = "/session/%s/chromium/send_command_and_get_result" % driver.session_id
    url = driver.command_executor._url + resource
    body = json.dumps({"cmd": cmd, "params": params})
    response = driver.command_executor._request("POST", url, body)

    if not response:
        raise Exception(response.get("value"))

    return response.get("value")


def __get_pdf_from_html(
    path: str, print_options: dict
):
    # Setup the headless Browser
    webdriver_options = Options()
    webdriver_prefs = {}
    driver = None

    webdriver_options.add_argument("--headless")
    webdriver_options.add_argument("--disable-gpu")
    webdriver_options.add_argument("--no-sandbox")
    webdriver_options.add_argument("--disable-dev-shm-usage")
    webdriver_options.experimental_options["prefs"] = webdriver_prefs

    webdriver_prefs["profile.default_content_settings"] = {"images": 2}

    driver = webdriver.Chrome(
        options=webdriver_options
    )

    # Get the given website
    driver.get(path)

    calculated_print_options = {
        "landscape": False,
        "displayHeaderFooter": False,
        "printBackground": True,
        "preferCSSPageSize": True,
    }
    calculated_print_options.update(print_options)
    result = __send_devtools(
        driver, "Page.printToPDF", calculated_print_options)
    
    driver.quit()
    pdf_data = base64.b64decode(result["data"])

    return pdf_data

if __name__ == "__main__":
    # <web-page-url> <pdf-file>
    if len(sys.argv) < 3:
        print(f"Invalid syntax: {sys.argv[0]} <web-page-url> <pdf-file>")
        exit(1)

    webpage_url = sys.argv[1]
    pdf_file = sys.argv[2]
    convert(webpage_url, pdf_file)
