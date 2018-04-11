from selenium import webdriver
import os
import time
import io

final_page_num = 57
word = 'activity'

os.chdir('C:/Users/Santi/Downloads/')

driver=webdriver.Chrome("C:/Users/Shiva/Downloads/chromedriver.exe")

driver.get(f'https://www.powerthesaurus.org/{word}/narrower/1')

# Click alphabet

words = []

for div_element in driver.find_elements_by_css_selector(".pt-thesaurus-card__term-title"):
    word = div_element.find_element_by_tag_name('a')
    words.append(word.text)

for i in range(2, final_page_num):
    driver.get(f'https://www.powerthesaurus.org/{word}/narrower/{i}')
    for div_element in driver.find_elements_by_css_selector(".pt-thesaurus-card__term-title"):
        word = div_element.find_element_by_tag_name('a')
        words.append(word.text)

print(words)

