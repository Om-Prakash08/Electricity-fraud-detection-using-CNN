import pandas as pd
from sklearn.preprocessing import MinMaxScaler

rawData = pd.read_csv('C:/Users/HP/Downloads/data.csv')

#data preprocessing 
#removing column 1 and 2(making InfoData)
rawData1_=rawData.iloc[:100,:]
rawData2_=rawData.iloc[-100:,:]
rawData=pd.concat([rawData1_, rawData2_], ignore_index=True)
infoData = pd.DataFrame()
infoData['FLAG'] = rawData['FLAG']
infoData['CONS_NO'] = rawData['CONS_NO']
data = rawData.drop(['FLAG', 'CONS_NO'], axis=1)   #axis 1 column ,axis 0 row

#droping duplicate row
dropIndex = data[data.duplicated()].index  # duplicates drop
data = data.drop(dropIndex, axis=0)   #droping duplicate value present wen two row are same
infoData = infoData.drop(dropIndex, axis=0) #droping duplicate index infodata

#removing row with all zero(Nan) value
zeroIndex = data[(data.sum(axis=1) == 0)].index  # zero rows drop
data = data.drop(zeroIndex, axis=0) 
infoData = infoData.drop(zeroIndex, axis=0)  

#change column name to dates(2014/1/1 to 2014-01-01)
data.columns = pd.to_datetime(data.columns)  #columns reindexing according to dates

#sort data accoding to date( as previusoly column are unsorted)
data = data.reindex(sorted(data.columns), axis=1)
cols = data.columns

# reindex row name (as some row has been remove till this step due to duplicate or all nan values)
data.reset_index(inplace=True, drop=True)  # index sorting
infoData.reset_index(inplace=True, drop=True)

#filling nan value using neighbouring value (middle missing value replace by average 
#and other by maximum 2 distance element)
data = data.interpolate(method='linear', limit=2, limit_direction='both', axis=0).fillna(0) 


#removing erronoues value(fixing outliers)
for i in range(data.shape[0]):  # outliers treatment
    m = data.loc[i].mean()
    st = data.loc[i].std()
    data.loc[i] = data.loc[i].mask(data.loc[i] > (m + 3 * st), other=m + 3 * st)

# save preprocessed data without scaling
data.to_csv(r'visualization.csv', index=False, header=True)  # preprocessed data without scaling

#noramalisation process
scale = MinMaxScaler()
scaled = scale.fit_transform(data.values.T).T
mData = pd.DataFrame(data=scaled, columns=data.columns)
preprData = pd.concat([infoData, mData], axis=1, sort=False)  # Back to initial format
print("Noramalised data")
print(preprData)

# save preprocessed data after scaling
preprData.to_csv(r'preprocessedR.csv', index=False, header=True)
