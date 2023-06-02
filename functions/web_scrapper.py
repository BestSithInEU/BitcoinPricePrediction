from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

import csv
import os
import threading
import time


class StoppableThread(threading.Thread):
    """
    A thread that can be stopped by setting the internal stop flag
    through stop() method.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the thread and set up the stop event.
        """

        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        """
        Set the stop flag to stop the thread.
        """

        self._stop_event.set()

    def stopped(self):
        """
        Check if the thread is stopped.

        Returns
        -------
            bool
                True if the thread is stopped, False otherwise.
        """

        return self._stop_event.is_set()


class WebScraper:
    """
    A web scraper for scraping headlines from coindesk.

    The WebScraper class provides methods for starting, stopping, pausing, and resuming the scraping process.
    It uses the Selenium library to navigate through web pages, find required elements, and write scraped data to a CSV file.
    """

    def __init__(
        self,
        url="https://www.coindesk.com/search?s=bitcoin&sort=1",
        starting_page_number=1,
        ending_page_number=3096,
        output_file_name="headlines",
    ):
        """
        Initialize the web scraper with the URL to scrape from, the range of pages
        to scrape, and the file to output the scraped headlines.

        Parameters
        ----------
        url : str, optional
            URL to scrape from. Defaults to "https://www.coindesk.com/search?s=bitcoin&sort=1".
        starting_page_number : int, optional
            Page number to start scraping from. Defaults to 1.
        ending_page_number : int, optional
            Page number to end scraping at. Defaults to 3096.
        output_file_name : str, optional
            Name of the file to output scraped headlines. Defaults to "headlines".
        """

        self.url = url
        self.starting_page_number = starting_page_number
        self.ending_page_number = ending_page_number
        self.output_file_name = output_file_name
        self.pause = False
        self.stop = False

    def start_scraping(
        self, url, starting_page_number, ending_page_number, output_file_name
    ):
        """
        Start the scraping process.

        Parameters
        ----------
        url : str
            URL to scrape from.
        starting_page_number : int
            Page number to start scraping from.
        ending_page_number : int
            Page number to end scraping at.
        output_file_name : str
            Name of the file to output scraped headlines.
        """

        self.thread = StoppableThread(
            target=self.webScrapper,
            args=(url, starting_page_number, ending_page_number, output_file_name),
        )
        self.thread.start()

    def stop_scraping(self):
        """
        Stop the scraping process by stopping the thread in which it runs.
        """

        self.thread.stop()

    def pause_scraping(self):
        """
        Pause the scraping process by setting the pause flag to True.
        """

        self.pause = True

    def resume_scraping(self):
        """
        Resume the scraping process by setting the pause flag to False.
        """

        self.pause = False

    def webScrapper(
        self, url, starting_page_number, ending_page_number, output_file_name
    ):
        """
        The main method for scraping the website. It navigates through the pages
        of the given url, finds the required elements, and writes the data to a CSV file.
        It also handles cases like file not found, consent button, and handles exceptions like
        TimeoutException. The scraping process can also be paused, resumed and stopped.

        Parameters
        ----------
        url : str
            The base url from where the scraping starts.
        starting_page_number : int
            The starting page number for scraping.
        ending_page_number : int
            The ending page number for scraping.
        output_file_name : str
            The name of the output file where the scraped data is stored.
        """

        output_file_name = f"dataset/{output_file_name}.csv"

        options = webdriver.ChromeOptions()
        options.add_argument("log-level=3")
        driver = webdriver.Chrome(options=options)

        driver.set_page_load_timeout(30)

        seen_articles = set()
        if os.path.isfile(output_file_name):
            with open(output_file_name, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=";")
                next(reader)
                for row in reader:
                    seen_articles.add(tuple(row))
        else:
            with open(output_file_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["News", "Date"])

        consent_clicked = False

        for page in range(starting_page_number, ending_page_number + 1):
            if self.thread.stopped():
                break

            while self.pause:
                time.sleep(1)

            if page == 1:
                url_ = url
            else:
                url_ = url + "&i=" + str(page)

            try:
                driver.get(url_)
                if not consent_clicked:
                    try:
                        consent_button = WebDriverWait(driver, 2).until(
                            EC.element_to_be_clickable(
                                (By.ID, "CybotCookiebotDialogBodyButtonAccept")
                            )
                        )
                        consent_button.click()
                        consent_clicked = True
                    except TimeoutException:
                        print("No consent button found on page, continuing")

                soup = BeautifulSoup(driver.page_source, "html.parser")

                for container in soup.find_all(
                    "div", {"class": "Box-sc-1hpkeeg-0 jBwlRs"}
                ):
                    headline = container.find(
                        "h6", {"class": "typography__StyledTypography-owin6q-0 hjHKEC"}
                    ).text.strip()
                    headline = headline.replace(
                        ",", ""
                    )  # Remove commas from the headline
                    date = container.find(
                        "h6", {"class": "typography__StyledTypography-owin6q-0 lfNAOh"}
                    ).text.strip()
                    article = (headline, date)
                    if article not in seen_articles:
                        seen_articles.add(article)
                        with open(
                            output_file_name, "a", newline="", encoding="utf-8"
                        ) as f:
                            writer = csv.writer(f, delimiter=";")
                            writer.writerow(article)

            except TimeoutException:
                print(f"Page {page} timed out, continuing to next page")
                continue

            except Exception as e:
                print(f"An error occurred on page {page}: {e}")
                continue

        driver.quit()
