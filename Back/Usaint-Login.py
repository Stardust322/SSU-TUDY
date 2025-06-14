import os
import time
import sys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc
import json
import requests
from bs4 import BeautifulSoup
import io
sys.stdout.reconfigure(encoding='utf-8')
sys.stdout = io.StringIO()
URL = "https://smartid.ssu.ac.kr/Symtra_sso/smln.asp?apiReturnUrl=https%3A%2F%2Fsaint.ssu.ac.kr%2FwebSSO%2Fsso.jsp"
options = uc.ChromeOptions()
options.add_argument("--disable-gpu") 
options.add_argument("--headless")
options.add_argument('--window-size=600,592')
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-blink-features=AutomationControlled")
driver = uc.Chrome(options=options)

driver.get(URL)
id = ""
pw = ""
user_id = driver.find_element(By.ID,'userid')
user_id.send_keys(id)
user_pw = driver.find_element(By.ID, 'pwd')
user_pw.send_keys(pw)

btn = driver.find_element(By.XPATH, "//*[@id=\"sLogin\"]/div/div[1]/form/div/div[2]/a")
btn.click()
driver.implicitly_wait(5)
time.sleep(1)
s = requests.Session()
login_url = "https://smartid.ssu.ac.kr/Symtra_sso/smln_pcs.asp"
login_data = {
    "userid": id,
    "pwd": pw,
    "in_tp_bit": "0",
    "rqst_caus_cd": "03"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
    "Referer": "https://saint.ssu.ac.kr/symtra_Sso/smln_login.asp",
    "Content-Type": "application/x-www-form-urlencoded",
}
s.headers.update(headers)

response_post = s.post(login_url, data=login_data, headers=headers)
response_post.raise_for_status()

for cookie in driver.get_cookies():
    c = {cookie['name'] : cookie['value']}
    s.cookies.update(c)

response = s.get("https://saint.ssu.ac.kr/webSSUMain/main_student.jsp")
soup = BeautifulSoup(response.text, 'html.parser')
name = str(soup.find_all("p", class_="main_title")[0].text).split("님")[0]
main_info = soup.find("div").find_all_next("strong")
print(name, end="/")
for i in main_info:
    print(str((i.text)).replace("과정", "과정 "), end="/")