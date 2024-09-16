# data_processing.py

import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
class DataProcessing:
    
    def load_and_clean_data(self,filepath):
        data = pd.read_csv(filepath)
        # Removing duplicates
        data = data.drop_duplicates(keep="first")
        return data
    def replace_outliers_with_mean(self,data, z_threshold=3):
        # Iterate through each numeric column
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in ['Bearer Id', 'IMSI', 'MSISDN/Number','IMEI']:
                col_data = data[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Replace outliers in the original DataFrame
                # Need to align the original index with the calculated z-scores
                data.loc[data[col].notna() & (abs(zscore(data[col].fillna(0))) > z_threshold), col] = mean_value
    def replace_missing_with_mean_or_mode(self,data):
        columns_to_drop = ['UnderwrittenCoverID','Country','Bank','IsVATRegistered','Citizenship']
        # Drop the specified columns from the DataFrame
        dataframe = dataframe.drop(columns=columns_to_drop, errors='ignore')
        #iterate over the columns
        for col in data.columns:
            if col in ['TotalPremium','TotalClaims']:
                data = data.dropna(subset=[col]) #drop the rows which don't have 'Bearer Id'or'IMSI'
            elif data[col].dtype == 'float64':  # If the column is numeric (float)
                mean_value = data[col].mean()
                data.loc[:,col]=data[col].fillna(mean_value)  # Replace NaN with mean
            elif data[col].dtype == 'object':  # If the column is object (string)
                mode_value = data[col].mode()[0]
                data.loc[:,col]=data[col].fillna(mode_value) # Replace NaN with mode
        
        return data
    def encoder(self,method, dataframe, columns_label, columns_onehot):
        if method == 'labelEncoder':      
            df_lbl = dataframe.copy()
            for col in columns_label:
                label = LabelEncoder()
                label.fit(list(dataframe[col].values))
                df_lbl[col] = label.transform(df_lbl[col].values)
            return df_lbl
        
        elif method == 'oneHotEncoder':
            df_oh = dataframe.copy()
            df_oh= pd.get_dummies(data=df_oh, prefix='ohe', prefix_sep='_',
                        columns=columns_onehot, drop_first=True, dtype='int8')
            return df_oh

    def scaler(self,method, data, columns_scaler):    
        if method == 'standardScaler':        
            Standard = StandardScaler()
            df_standard = data.copy()
            df_standard[columns_scaler] = Standard.fit_transform(df_standard[columns_scaler])        
            return df_standard
            
        elif method == 'minMaxScaler':        
            MinMax = MinMaxScaler()
            df_minmax = data.copy()
            df_minmax[columns_scaler] = MinMax.fit_transform(df_minmax[columns_scaler])        
            return df_minmax
        
        elif method == 'npLog':        
            df_nplog = data.copy()
            df_nplog[columns_scaler] = np.log(df_nplog[columns_scaler])        
            return df_nplog
        
        return data
