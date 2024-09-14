import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class InsuranceEDA:
    def txt_to_csv(self,input_file, output_file, delimiter='|'):
        if not isinstance(input_file, str):
            raise ValueError(f"Invalid file path: expected a string, got {type(input_file)}")

        input_file = input_file.replace('/', '\\')
        try:
            # Read the .txt file into a pandas DataFrame
            df = pd.read_csv(input_file, delimiter=delimiter,low_memory=False)
            
            # Save the DataFrame as a .csv file
            df.to_csv(output_file, index=False)
            
            print(f"File successfully converted and saved as {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def summarize_data(self,data):
        #  List of columns to summarize
        columns_to_summarize = ['SumInsured', 'TermFrequency', 'CalculatedPremiumPerTerm', 
                                'ExcessSelected', 'TotalPremium', 'TotalClaims']
        
        # Check if these columns exist in the data
        existing_columns = [col for col in columns_to_summarize if col in data.columns]
        # Apply .describe() only to the specified columns
        summary = data[existing_columns].describe()

        return summary

    def check_missing_values(self,data):
        """Check for missing values in the dataset"""
        missing_values = data.isnull().sum()
        print("Missing Values:\n", missing_values)
        # return missing_values

    def univariate_analysis(self,data):
        # Exclude the following columns from the analysis
        exclude_columns = ['UnderwrittenCoverID', 'PolicyID', 'PostalCode', 'mmcode', 'SubCrestaZone', 'Model','CapitalOutstandingRange']
        data_filtered = data.drop(columns=exclude_columns, errors='ignore')
        col_to_edit=['TransactionMonth','Gender','VehicleIntroDate','CapitalOutstandingRange']
        # Plot TransactionMonth with Month name format
        data_filtered['TransactionMonth'] = pd.to_datetime(data_filtered['TransactionMonth'])
        data_filtered['MonthName'] = data_filtered['TransactionMonth'].dt.strftime('%B')  # Display month names
        plt.figure(figsize=(10, 6))
        data_filtered['MonthName'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.title('Transaction Count by Month')
        plt.xlabel('Month')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
        
        # Plot Gender with custom logic for 'Title'
        def custom_gender(row):
            if row['Gender'] == 'Not specified':
                if row['Title'] == 'Mr':
                    return 'Male'
                elif row['Title'] in ['Mrs', 'Miss', 'Ms']:
                    return 'Female'
            return row['Gender']
        
        data_filtered['CustomGender'] = data_filtered.apply(custom_gender, axis=1)
        plt.figure(figsize=(6, 4))
        data_filtered['CustomGender'].value_counts().plot(kind='bar', color='lightgreen')
        plt.title('Gender Distribution')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.show()
        
        # Plot VehicleIntroDate by year range
        data_filtered['VehicleIntroDate'] = pd.to_datetime(data_filtered['VehicleIntroDate'], errors='coerce')
        data_filtered['VehicleIntroYear'] = data_filtered['VehicleIntroDate'].dt.year
        plt.figure(figsize=(10, 6))
        data_filtered['VehicleIntroYear'].value_counts().sort_index().plot(kind='bar', color='orange')
        plt.title('Vehicle Introduction Year Distribution')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
        
        # # Plot CapitalOutstanding by 10,000 range
        # plt.figure(figsize=(10, 6))
        # data_filtered['CapitalOutstandingRange'] = (data_filtered['CapitalOutstanding'] // 10000) * 10000
        # data_filtered['CapitalOutstandingRange'].value_counts().sort_index().plot(kind='bar', color='purple')
        # plt.title('Capital Outstanding Distribution by 10,000 Range')
        # plt.xlabel('Capital Outstanding (Grouped by 10,000)')
        # plt.ylabel('Count')
        # plt.xticks(rotation=45)
        # plt.show()

        """Plot histograms for numerical columns and bar charts for categorical columns"""
        # Numerical columns
        numeric_cols = data_filtered.select_dtypes(include=['float64', 'int64']).columns
        data_filtered[numeric_cols].hist(bins=15, figsize=(15, 10))
        plt.show()
        
        # Categorical columns
        categorical_cols = data_filtered.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in col_to_edit:
                plt.figure(figsize=(10, 5))
                sns.countplot(y=data_filtered[col])
                plt.title(f"Distribution of {col}")
                plt.show()
    
    def bivariate_analysis(self,data):
        """Perform correlation analysis for numeric variables"""
        numeric_cols = ['TotalPremium', 'TotalClaims']
        corr = data[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation between TotalPremium and TotalClaims")
        plt.show()

    def outlier_detection(self,data):
        """Detect outliers using boxplots"""
        numeric_cols = ['TotalPremium', 'TotalClaims']
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=data[col])
            plt.title(f"Boxplot for {col}")
            plt.show()

    def geographic_trend_analysis(self,data):
        """Explore trends across different provinces or zipcodes"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Province', y='TotalPremium', data=data)
        plt.xticks(rotation=90)
        plt.title("Premiums by Province")
        plt.show()

    def generate_creative_plots(self,data):
        """Produce creative plots based on EDA findings"""
        # Example 1: Distribution of premiums by car make
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Make', y='TotalPremium', data=data)
        plt.xticks(rotation=90)
        plt.title("Premiums by Car Make")
        plt.show()

        # Example 2: Claims by Province
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Province', y='TotalClaims', data=data)
        plt.xticks(rotation=90)
        plt.title("Claims by Province")
        plt.show()

        # Example 3: Premium vs Claims Scatter Plot
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='TotalPremium', y='TotalClaims', data=data, hue='VehicleType')
        plt.title("Total Premium vs Total Claims")
        plt.show()
