from Air_quality import get_yearly_avg_data

import requests
import sys
import pandas as pd
from bs4 import BeautifulSoup
import os
import csv

def met_data(month, year):
    
    file_html = open('data/data_html/{}/{}.html'.format(year,month), 'rb')
    plain_text = file_html.read()

    data_temporary = []
    Data_final = []

    soup = BeautifulSoup(plain_text, "lxml")
    ## extracting data from html format
    for table in soup.findAll('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                data_item = tr.get_text()
                data_temporary.append(data_item)

    ## initial number of features
    rows = len(data_temporary) / 15

    for times in range(round(rows)):
        newdata_temporary = []
        for i in range(15):
            newdata_temporary.append(data_temporary[0])
            data_temporary.pop(0)
        Data_final.append(newdata_temporary)

    length = len(Data_final)

    Data_final.pop(length - 1)
    Data_final.pop(0)

    for a in range(len(Data_final)):
        Data_final[a].pop(6)
        Data_final[a].pop(13)
        Data_final[a].pop(12)
        Data_final[a].pop(11)
        Data_final[a].pop(10)
        Data_final[a].pop(9)
        Data_final[a].pop(0)

    return Data_final

def data_combine(year, cs):
    for a in pd.read_csv('data/final-data/finalize_data' + str(year) + '.csv', chunksize=cs):
        df = pd.DataFrame(data=a)
        mylist = df.values.tolist()
    return mylist


if __name__ == "__main__":
    if not os.path.exists("data/final-data"):
        os.makedirs("data/final-data")
    for year in range(2013, 2019):
        final_data = []
        with open('data/final-data/finalize_data' + str(year) + '.csv', 'w') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            wr.writerow(
                ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        for month in range(1, 13):
            temp = met_data(month, year)
            final_data = final_data + temp
            
        pm = get_yearly_avg_data(year)

        if len(pm) == 364:
            pm.insert(364, '-')

        for i in range(len(final_data)-1):
            # final[i].insert(0, i + 1)
            final_data[i].insert(8, pm[i])

        with open('data/final-data/finalize_data' + str(year) + '.csv', 'a') as csvfile:
            wr = csv.writer(csvfile, dialect='excel')
            for row in final_data:
                flag = 0
                for elem in row:
                    if elem == "" or elem == "-":
                        flag = 1
                if flag != 1:
                    wr.writerow(row)
                    
    data_2013 = data_combine(2013, 600)
    data_2014 = data_combine(2014, 600)
    data_2015 = data_combine(2015, 600)
    data_2016 = data_combine(2016, 600)
    data_2017 = data_combine(2017, 600)
    data_2018 = data_combine(2018, 600)
     
    total=data_2013+data_2014+data_2015+data_2016+data_2017+data_2018
    
    with open('data/final-data/finalize_dataCombine.csv', 'w') as csvfile:
        wr = csv.writer(csvfile, dialect='excel')
        wr.writerow(
            ['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM', 'PM 2.5'])
        wr.writerows(total)
        
        
df=pd.read_csv('data/final-data/finalize_dataCombine.csv')