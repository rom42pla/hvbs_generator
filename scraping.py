import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import chromedriver_binary
from fake_useragent import UserAgent

options = Options()
ua = UserAgent()
userAgent = ua.random
# options.add_argument(f'user-agent={userAgent}')
options.add_argument(f'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64')
# options.add_argument("--enable-javascript")
options.add_argument("--incognito")
driver = webdriver.Chrome(executable_path=f"/home/rom42pla/Downloads/chromedriver", options=options)
driver.get(f"https://www.dndbeyond.com/homebrew/magic-items?filter-type=4&filter-search=&filter-rarity=1&filter-requires-attunement=&filter-effect-type=&filter-effect-subtype=&filter-has-charges=&filter-author=&filter-author-previous=&filter-author-symbol=&filter-rating=-13&page=1&sort=rating")
# elem = driver.find_element(By.XPATH, ('//ul[@class="listing listing-rpgmagic-item rpgmagic-item-listing"]'))
# elem.click()
# elem.clear()
# elem.send_keys("pycon")
# elem.send_keys(Keys.RETURN)
# assert "No results found." not in driver.page_source
time.sleep(30)
driver.close()