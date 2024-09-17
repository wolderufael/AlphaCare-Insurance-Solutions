import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind,f_oneway
class ABTest:
    def __init__(self, data):
        self.data = data
    def catagorize_province_into_2(self):
        Test_1_df=self.data[['Province','TotalClaims']].copy()
        # drop the row if province is missing
        Test_1_df = Test_1_df.dropna(subset=['Province'])
        #randomly catagorize the provinces into two
        control_province=['Gauteng','Limpopo','North West' ,'Mpumalanga','Eastern Cape']
        test_province=['Western Cape','KwaZulu-Natal','Free State','Northern Cape']
        
        Test_1_df.loc[:,'Province_Category'] = Test_1_df['Province'].apply(lambda x: '1' if x in control_province else '2')
        return Test_1_df
    def calculate_profit(self):
        self.data['ProfitMargin'] = self.data.apply(
            lambda row: (row['TotalPremium'] - row['TotalClaims']) / row['TotalPremium']
        if row['TotalPremium'] > 0 else 0, axis=1)
    
    def identify_gender(self):
        self.data['Gender'] = self.data.apply(
            lambda row: 'Male' if row['Gender'] == 'Not specified' and row['Title'] == 'Mr'
            else ('Female' if row['Gender'] == 'Not specified' and row['Title'] in ['Mrs', 'Miss', 'Ms'] 
                else row['Gender']),
            axis=1)
        
    def chi_squared_test(self, group_col, metric_col):
        contingency_table = pd.crosstab(self.data[group_col], self.data[metric_col])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        return chi2, p_value
    def t_test(self, df,group_col, metric_col, group_A, group_B):
        group_A_data = df[df[group_col] == group_A][metric_col]
        group_B_data = df[df[group_col] == group_B][metric_col]
        
        # Perform a t-test
        t_stat, p_value = ttest_ind(group_A_data, group_B_data)
        print('The p-value of T-Test is: ',p_value)
        
        return t_stat, p_value
    def anova_test(self, group_col, metric_col):
        # Perform ANOVA across multiple groups
        groups = [self.data[self.data[group_col] == group][metric_col] for group in self.data[group_col].unique()]
        f_stat, p_value = f_oneway(*groups)
        
        return f_stat, p_value