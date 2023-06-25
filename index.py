import os
import time
import requests
import sys

def data_retrieve():
    for year in range(2013,2019):
        for month in range(1,13):
            if month<10:
                url = 'https://en.tutiempo.net/climate/0{}-{}/ws-421820.html'.format(month, year)
            else:
                url = 'https://en.tutiempo.net/climate/{}-{}/ws-421820.html'.format(month, year)


            input_text = requests.get(url)
            # incode data to utf-8 format
            text_utf = input_text.text.encode('utf-8')

            if not os.path.exists("data/data_html/{}".format(year)):
                os.makedirs("data/data_html/{}".format(year))
            with open("data/data_html/{}/{}.html".format(year,month),"wb") as output:
                output.write(text_utf)

            sys.stdout.flush()

if __name__=="__main__":
    start_time = time.time()
    data_retrieve()
    end_time = time.time()
    print("time for extracting data {}".format(end_time - start_time))