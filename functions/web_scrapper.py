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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class WebScraper:
    def __init__(
        self,
        url="https://www.coindesk.com/search?s=bitcoin&sort=1",
        starting_page_number=1,
        ending_page_number=3096,
        output_file_name="headlines",
    ):
        self.url = url
        self.starting_page_number = starting_page_number
        self.ending_page_number = ending_page_number
        self.output_file_name = output_file_name
        self.pause = False
        self.stop = False

    def start_scraping(
        self, url, starting_page_number, ending_page_number, output_file_name
    ):
        self.thread = StoppableThread(
            target=self.webScrapper,
            args=(url, starting_page_number, ending_page_number, output_file_name),
        )
        self.thread.start()

    def stop_scraping(self):
        self.thread.stop()

    def pause_scraping(self):
        self.pause = True

    def resume_scraping(self):
        self.pause = False

    def webScrapper(
        self, url, starting_page_number, ending_page_number, output_file_name
    ):
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
