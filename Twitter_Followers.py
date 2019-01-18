
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as pls 


# In[364]:


df = pd.read_csv("endowment_2018.csv")


# In[351]:


df.head()


# In[7]:


result1 = df.groupby(['Public/Private']).mean()
result1.rename(index={0:'Private',1:'Public'},inplace=True)
result1


# In[356]:



id_list = []
for keyword in df.University:
    url = 'https://twitter.com/search?f=users&vertical=news&q='+ keyword + '&src=typd&lang=en'
    url = url.replace(' ', '%20')
    htmlpage = requests.get(url)
    soup=BeautifulSoup(htmlpage.text,'html.parser')
    button = soup.find('div',{'class':'js-stream-item'})['data-item-id']

    id_list.append(button)


# In[366]:


df['TwitterID'] = id_list


# In[369]:


import tweepy

# Keys, tokens and secrets
consumer_key = "V4AByC2mPcLVm06jQdu5fGfxY"
consumer_secret = "Y9tIo9gUEiZHUeZrPPriKDMCuSrTljCCyOVCk1PwCImodb9qSv"
access_token = "826549909-Ba6AEl4orwqSWkgjjm79kM5dQNX8fJT5mymdd4de"
access_token_secret = "qMXbCIKtjexsLsjLussX8yfTb0OB0h6QhJufSevRLZYap"

# Tweepy OAuthHandler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

followers = []
for item in df.TwitterID:    
    user = api.get_user(item)
    followers_count = user.followers_count
    
    followers.append(followers_count)
#     print(user.name)
#     print(user.id)
#     print(user.followers_count)


# In[371]:


df['Followers'] = followers


# In[391]:


df.to_csv("endowments_2018_2.csv")

