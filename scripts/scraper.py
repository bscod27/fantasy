import pandas as pd
import numpy as np
import regex as re
import time
import sys

try:
    maxyear = int(sys.argv[1])
except:
    raise Exception("Error: no argument specifying the max year was provided")

df = pd.DataFrame()
for i in range(maxyear-20, maxyear+1): # due to zero index
    print(str(i) + '...')
    page = 'https://www.pro-football-reference.com/years/' + str(i) + '/fantasy.htm'
    data = pd.read_html(page)[0]
    data.columns = [i[1] for i in data.columns]
    data['Year'] = i
    data['ProBowl'] = data.Player.str.contains('\*').astype(int)
    data['AllPro'] = data.Player.str.contains('\+').astype(int)
    data['Player'] = data.Player.str.strip('*+')
    df = pd.concat([df, data], axis=0)
    time.sleep(5)


df = df[df.Player != 'Player']
df.to_csv('../data/data_'+ str(maxyear-20) + '_' + str(maxyear) + '.csv', index=False)
