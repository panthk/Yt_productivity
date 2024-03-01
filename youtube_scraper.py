import sqlite3
from sqlite3 import Error
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common import TimeoutException

# Function to create a connection to the SQLite database
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

# Function to create a table in the SQLite database
def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

# Function to insert a new row into the SQLite database
def insert_data(conn, data):
    sql = ''' INSERT INTO videos(title, description, views, tags, keywords, url)
              VALUES(?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, data)
    conn.commit()
    return cur.lastrowid

# Path to the SQLite database
database = "videoData.db"

# SQL statement to create the videos table
create_table_sql = """ CREATE TABLE IF NOT EXISTS videos (
                                id INTEGER PRIMARY KEY,
                                title TEXT NOT NULL,
                                description TEXT,
                                views INTEGER,
                                tags TEXT,
                                keywords TEXT,
                                url TEXT NOT NULL
                            ); """

# Create a database connection
conn = create_connection(database)
if conn is not None:
    # Create videos table
    create_table(conn, create_table_sql)
else:
    print("Error! cannot create the database connection.")

# enable the headless mode
options = Options()
# options.add_argument('--headless=new')

# initialize a web driver instance to control a Chrome window
driver = webdriver.Chrome(
    service=ChromeService(ChromeDriverManager().install()),
    options=options
)

# the URL of the target page
url = 'https://www.youtube.com/watch?v=kuDuJWvho7Q'
# visit the target page in the controlled browser
driver.get(url)

try:
    # wait up to 15 seconds for the consent dialog to show up
    consent_overlay = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, 'dialog'))
    )

    # select the consent option buttons
    consent_buttons = consent_overlay.find_elements(By.CSS_SELECTOR, '.eom-buttons button.yt-spec-button-shape-next')
    if len(consent_buttons) > 1:
        # retrieve and click the 'Accept all' button
        accept_all_button = consent_buttons[1]
        accept_all_button.click()
except TimeoutException:
    pass

# wait for YouTube to load the page data
WebDriverWait(driver, 15).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, 'h1.ytd-watch-metadata'))
)

# scraping logic
title = driver \
    .find_element(By.CSS_SELECTOR, 'h1.ytd-watch-metadata') \
    .text

# scrape the channel info attributes
channel_element = driver.find_element(By.ID, 'owner')

channel_url = channel_element \
              .find_element(By.CSS_SELECTOR, 'a.yt-simple-endpoint') \
              .get_attribute('href')
channel_name = channel_element \
              .find_element(By.ID, 'channel-name') \
              .text
channel_image = channel_element \
              .find_element(By.ID, 'img') \
              .get_attribute('src')
channel_subs = channel_element \
              .find_element(By.ID, 'owner-sub-count') \
              .text \
              .replace(' subscribers', '')

# click the description section to expand it
driver.find_element(By.ID, 'description-inline-expander').click()

info_container_elements = driver \
    .find_elements(By.CSS_SELECTOR, '#info-container span')
views = info_container_elements[0] \
    .text \
    .replace(' views', '')
publication_date = info_container_elements[2] \
    .text

description = driver \
    .find_element(By.CSS_SELECTOR, '#description-inline-expander .ytd-text-inline-expander span') \
    .text

# Placeholder for tags and keywords, you need to implement logic to extract them
tags = ""
keywords = ""

# Close the browser and free up the resources
driver.quit()

# Data to be inserted into the database
data = (title, description, int(views.replace(',', '')), tags, keywords, url)

# Insert data into the SQLite database
with conn:
    insert_data(conn, data)