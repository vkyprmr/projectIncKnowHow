#Coder: Vicky Parmar

#Imports
import pandas as pd
import numpy as np


#creating a list of odd and even nums
numlist = list(range(0,343))
odds = []
evens = []

for x in numlist:
    if x % 2 == 0:
        evens.append(x)
    else:
        odds.append(x)
    


#Data
file = 'path'
data = pd.read_csv(file) #or read_excel 'For more see pandas documentation'
data.iloc[rows, columns] #positional indices returns a dataframe 'using .values will return a numpy array' "columns = -1 == last column"
data.loc[rows, columns] #where rows and columns are names of the indices
    rows = ['',''] and columns = ['','']
data.ix[rows, columns] #works the same way as .iloc when passed integers and as .loc when passed strings

data.X = data['X'] #selecting the X column in a dataframe
new_data = data[(data.X == 0)] #can be used for filtering data
new_data.astype(float).round(4) #taking values as float and rounds upto 4 decimals

#to get the frequency of values in a dataframe column
data.a.value_counts()
data['a'].value_counts()

#to keep all the rows with some values in a list
to_keep = [a,b,c,d,e,f]
data = data.loc[data['X'].isin(to_keep)] #will look in column X and keep the rows where values are in the list

#droping
data.drop(['a','b'], axis=0) #drops rows a and b
data.drop(['a','b'], axis=1) #drops columns a and b
data.dropna() #drops all rows if any one column has NaN values
data.drop(subset=['a'], how='all', inplace=True) #keeps one row with x in 'a' and drops the rest (kinda removing duplicates)

'''
data.fillna(0) #fills NaN values with 0
data.fillna(method='ffill') #fills NaN values with forward fill or previous values
data.fillna(method='bfill') #backward fill or next values
data.interpolate(method='linear') #interpolates values from start to end

Learn more from documentation

'''

#combining dataframes
combined = pd.concat([data, new_data], axis=0) #combines dataframe along rows (axis=1 will do the same for columns)

#exporting files
combined.to_excel('filename.xlsx') #or *csv


#Used for selecting data based on certain conditions
col = data.X
#replacing 0 if it repeats for a certain number of time (threshold)
def f(col, threshold=6):
    mask = col.groupby((col != col.shift()).cumsum()).transform('count').lt(threshold)
    mask &= col.eq(0) #whatever value you want to replace
    col.update(col.loc[mask].replace(0,0.00001)) #replacing 0 by 0.00001
    return col

data = data.apply(f, threshold=4) #applying the created function

def sn (row):
    if row['X'] == 0: #can use 'or', 'and' to combine multiple conditions
        return 'New'
    return 'Old'

data['X1'] = data.apply(lambda row: sn (row), axis=1)

def time_window (row):
    if row['X'] == 0: #can use 'or', 'and' to combine multiple conditions
        return '1'
    return '0'

data['X2'] = data.apply(lambda row: time_window (row), axis=1)

#rolling mean
data['X3'] = data['X2'].rolling(50).mean()

def nr (row):
    if row['X'] == 0 and row['X2']  < 0.27 or row['X1'] == 'New':
        return '1'
    return '0'

data['N'] = data.apply(lambda row: nr (row), axis=1) #will have 1s for few rows where conditions meet and the 0s for the rest

data['Nr'] = (data.N != data.N.shift(1)).astype(int).cumsum() #will order the previous column as 1,2,3,4 where 1 will contain all the values for 1s and 2 - for 0s and then 3 - for 1s and so on

data = data[data['Nr'].isin(odds)]
data['Nr'] = (data.N != data.N.shift(1)).astype(int).cumsum() #final list of the values you want


    















