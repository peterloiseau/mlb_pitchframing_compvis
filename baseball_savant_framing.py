#all these packages must be install before the script can run
from bs4 import BeautifulSoup
from selenium import webdriver
from urllib.request import urlopen
import time
import csv
import pandas as pd
import re
import requests
from selenium.webdriver import FirefoxOptions

opts=FirefoxOptions()
opts.add_argument("--headless")
browser = webdriver.Firefox(options=opts)

def get_links():
    root='https://baseballsavant.mlb.com'
    #statcast search for only called balls and strikes (excluding pitches in dirt)
    spider='/statcast_search?hfPT=&hfAB=&hfGT=R%7C&hfPR=ball%7Ccalled%5C.%5C.strike%7C&hfZ=&stadium=&hfBBL=&hfNewZones=&hfPull=&hfC=&hfSea=2021%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hfBBT=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=name&sort_col=pitches&player_event_sort=api_p_release_speed&sort_order=asc&min_pas=0#results'
    browser.get(root+spider)
    #find the css to all the rows of the search
    rows_to_expand=browser.find_elements_by_css_selector('td.player_name')
    #expand all the rows aggregated by pitchers to get individual pitches
    for row in rows_to_expand:
        row.click()
        time.sleep(5)
    #scrape rows of expanded tables
    expanded_tables=browser.find_elements_by_css_selector('div.ajax-table')
    rows=[]
    url_list=[]
    for table in expanded_tables:
        raw_html = table.get_attribute('innerHTML')
        soup = BeautifulSoup(raw_html,'html.parser')
        for row in soup.find('tbody').find_all('tr'):
            hyp = row.find_all('a')
            cols = row.find_all('td')
            row=[]
            #scrape info about pitches
            for col in cols:
                row.append(col.text)               
            rows.append(row)
            #scrape urls to videos of pitches
            #note some pitches don't have videos
            if len(hyp)==1:
                url_list.append(root+hyp[0]['href'])
            else:
                url_list.append('')
    #save the pitch data to a dataframe and save to file                     
    df = pd.DataFrame(rows)
    fdf = pd.concat([df.reset_index(drop=True), pd.DataFrame(url_list)], axis=1)
    #filter for any rows without video
    pdf = fdf.loc[fdf.iloc[:,len(fdf.columns)-1]!='']
    fin_df = pdf.sample(n=5000,random_state=42)
    furl_list = list(fin_df.iloc[:,len(fin_df.columns)-1])
    return fin_df, furl_list

def scrape(url_list):
    #go to each of the videos and download the videos to file
    count=1
    for url in url_list:
        #there are some urls which won't load videos, I found one here: https://baseballsavant.mlb.com/sporty-videos?playId=59903391-68ba-4990-be3c-5a4982849c61
        try:
            page = urlopen(url).read()
            soup = BeautifulSoup(page,'html.parser')
            vid_soup = soup.find('source')
            name = vid_soup['src'].split('/')[-1]
            r = requests.get(vid_soup['src'], stream=True)
            #download the video to file
            with open('C:/Users/peter/Documents/baseball-databases/savant_video/vids/'+name,'wb') as f:
                for chunk in r.iter_content(chunk_size=1024): 
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
        except:
            pd.DataFrame().to_csv('C:/Users/peter/Documents/baseball-databases/savant_video/vids/err-'+str(count)+'.csv')
            count+=1

    return name

        
if __name__ == '__main__':
    #df,urls=get_links()
    #df.to_csv(r'C:\Users\peter\Documents\baseball-databases\savant_video\pitch_desc_samp.csv',index=False)
    url_df = pd.read_csv(r'C:\Users\peter\Documents\baseball-databases\savant_video\pitch_desc_samp.csv')
    urls = list(url_df.iloc[1145:5000,-1])
    scrape(urls)
    
