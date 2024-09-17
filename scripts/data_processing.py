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
    def replace_outliers_with_mean(self, data,z_threshold=3):
        columns_to_drop = ['UnderwrittenCoverID','Country','Bank','IsVATRegistered','Citizenship']
        # Drop the specified columns from the DataFrame
        data = data.drop(columns=columns_to_drop, errors='ignore')
        # Iterate through each numeric column
        for col in data.select_dtypes(include=[np.number]).columns:
                col_data = data[col].dropna()
                col_zscore = zscore(col_data)
                
                # Create a boolean mask for outliers
                outlier_mask = abs(col_zscore) > z_threshold
                
                # Calculate the mean of non-outlier values (excluding NaNs)
                mean_value = col_data[~outlier_mask].mean()
                
                # Replace outliers in the original DataFrame
                # Need to align the original index with the calculated z-scores
                data.loc[data[col].notna() & (abs(zscore(data[col].fillna(0))) > z_threshold), col] = int(mean_value)
                return data
    def replace_missing_with_mean_or_mode(self,data):
        #drop the colum since it have all null values
        # data.drop('NumberOfVehiclesInFleet', axis=1)
        for col in data.columns:
            # if col in ['TotalPremium','TotalClaims']:
            #     data = data.dropna(subset=[col]) #drop the rows which don't have 'Bearer Id'or'IMSI'
            if data[col].dtype == 'float64':  # If the column is numeric (float)
                mean_value = data[col].mean()
                data.loc[:,col]=data[col].fillna(mean_value)  # Replace NaN with mean
            elif data[col].dtype == 'object':  # If the column is object (string)
                mode_value = data[col].mode()
                if not mode_value.empty:
                    data.loc[:,col]=data[col].fillna(mode_value[0]) # Replace NaN with mode
        
        return data
    def catagorize_columns(self,data):
        numerical_columns=[]
        catagorical_columns=[]
        for col in data.columns:
            if data[col].dtype in ['float64','int64']:
                numerical_columns.append(col)
            else :
                catagorical_columns.append(col)
        return numerical_columns,catagorical_columns
    def encoder(self,method, dataframe, columns_label, columns_onehot):
        if method == 'labelEncoder':      
            df_lbl = dataframe.copy()
            for col in columns_label:
                label = LabelEncoder()
                label.fit(list(dataframe[col].values))
                df_lbl[col] = label.transform(df_lbl[col].values)
            return df_lbl
        
        elif method == 'oneHotEncoder':
            df_oh = dataframe.copy()  # Create a copy of the original DataFrame
            for col in columns_onehot:
                # Apply one-hot encoding to each column in columns_onehot
                df_oh = pd.get_dummies(data=df_oh, prefix=f'ohe_{col}', prefix_sep='_',
                                    columns=[col], drop_first=True, dtype='int8')
            return df_oh


    def scaler(self,method, data, columns_scaler):    
        if method == 'standardScaler':        
            df_standard = data.copy()
            Standard = StandardScaler()
            for col in columns_scaler:
                df_standard[col] = Standard.fit_transform(df_standard[[col]])  # Scaling each column individually
            return df_standard
            
        elif method == 'minMaxScaler':        
            df_minmax = data.copy()
            MinMax = MinMaxScaler()
            for col in columns_scaler:
                df_minmax[col] = MinMax.fit_transform(df_minmax[[col]])  # Scaling each column individually
            return df_minmax
        
        elif method == 'npLog':        
            df_nplog = data.copy()
            for col in columns_scaler:
                df_nplog[col] = np.log(df_nplog[col])  # Applying log transform to each column individually
            return df_nplog
        
        return data
