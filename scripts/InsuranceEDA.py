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
        """Perform basic summarization and descriptive statistics"""
        print(data.describe())
        print(data.info())
        return data.describe()

    def check_missing_values(self,data):
        """Check for missing values in the dataset"""
        missing_values = data.isnull().sum()
        print("Missing Values:\n", missing_values)
        return missing_values

    def univariate_analysis(self,data):
        """Plot histograms for numerical columns and bar charts for categorical columns"""
        # Numerical columns
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_cols].hist(bins=15, figsize=(15, 10))
        plt.show()
        
        # Categorical columns
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            plt.figure(figsize=(10, 5))
            sns.countplot(y=data[col])
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
