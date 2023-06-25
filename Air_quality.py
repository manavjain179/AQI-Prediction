import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

def get_yearly_avg_data(year):
    ith_temperature = 0
    mean_temp = []
    for rows in pd.read_csv(f'data/air_quality/aqi{year}.csv', chunksize=24):
        add_var = 0
        temp_average = 0.0
        result = []
        df = pd.DataFrame(data=rows)
        for index, row in df.iterrows():
            result.append(row['PM2.5'])
        for item in result:
            if type(item) is float or type(item) is int:
                add_var += item
            elif type(item) is str:
                if item not in ['NoData', 'PwrFail', '---', 'InVld']:
                    temp = float(item)
                    add_var += temp
        temp_average = add_var / 24
        ith_temperature += 1
        mean_temp.append(temp_average)
    return mean_temp
    

if __name__=="__main__":
    list_2013=get_yearly_avg_data(2013)
    list_2014=get_yearly_avg_data(2014)
    list_2015=get_yearly_avg_data(2015)
    list_2016=get_yearly_avg_data(2015)
    list_2017=get_yearly_avg_data(2015)
    list_2018=get_yearly_avg_data(2015)

    plt.plot(range(0,365),list_2013,label="year 2013 data")
    plt.plot(range(0,364),list_2014,label="year 2014 data")
    plt.plot(range(0,365),list_2015,label="year 2015 data")
    plt.plot(range(0,365),list_2016,label="year 2016 data")
    plt.plot(range(0,365),list_2017,label="year 2017 data")
    plt.plot(range(0,365),list_2018,label="year 2018 data")
    
    plt.xlabel('No of Days')
    plt.ylabel('PM 2.5')
    plt.legend(loc='upper right')
    plt.show()
