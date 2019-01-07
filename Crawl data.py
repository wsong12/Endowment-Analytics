from urllib.request import urlopen 
from bs4 import BeautifulSoup 
import pandas as pd


web_page='https://en.wikipedia.org/wiki/List_of_colloquial_names_for_universities_and_colleges_in_the_United_States'
page =urlopen(web_page)
soup = BeautifulSoup(page, 'html.parser')

text=[]
for link in soup.find_all('li'):
#print(link.get('title')) #if len(link)>2: text.append(link.text)
d={}
for i in text:
if i.count(" - ")>0:
idx=i.split(" - ",1)[0].lstrip() name=i.split(" - ",1)[1].lstrip()
\
if name.count(",")>0 and len(name)>35: n=name.split(",")
#print(n)
for item in n:
        item=item.lstrip()
        d[item]=idx
else: d[name]=idx