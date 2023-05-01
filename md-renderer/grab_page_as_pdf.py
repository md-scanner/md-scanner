import json
import base64
from time import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import staleness_of
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By


def convert(
    source: str,
    target: str,
    timeout: int = 2,
    install_driver: bool = True,
    print_options: dict = {},
):
    """
    Convert a given html file or website into PDF

    :param str source: source html file or website link
    :param str target: target location to save the PDF
    :param int timeout: timeout in seconds. Default value is set to 2 seconds
    :param bool compress: whether PDF is compressed or not. Default value is False
    :param int power: power of the compression. Default value is 0. This can be 0: default, 1: prepress, 2: printer, 3: ebook, 4: screen
    :param dict print_options: options for the printing of the PDF. This can be any of the params in here:https://vanilla.aslushnikov.com/?Page.printToPDF
    """

    result = __get_pdf_from_html(
        source, timeout, install_driver, print_options)
    
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
    path: str, timeout: int, install_driver: bool, print_options: dict
):
    # Setup the headless Browser
    started_at = time()

    webdriver_options = Options()
    webdriver_prefs = {}
    driver = None

    webdriver_options.add_argument("--headless")
    webdriver_options.add_argument("--disable-gpu")
    webdriver_options.add_argument("--no-sandbox")
    webdriver_options.add_argument("--disable-dev-shm-usage")
    webdriver_options.experimental_options["prefs"] = webdriver_prefs

    webdriver_prefs["profile.default_content_settings"] = {"images": 2}

    if install_driver:
        driver = webdriver.Chrome(
            ChromeDriverManager().install(), options=webdriver_options
        )
    else:
        driver = webdriver.Chrome(options=webdriver_options)

    print(f"Browser initialized in: {time() - started_at:.3f}s")

    # Get the given website
    started_at = time()

    driver.get(path)

    print(f"Page ready in: {time() - started_at:.3f}s")

    started_at = time()

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

    print(f"PDF generated in: {time() - started_at:.3f}s")

    return pdf_data


convert('http://localhost:4000/', 'sample.pdf', 2, True)

