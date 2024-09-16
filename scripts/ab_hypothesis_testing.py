import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind,f_oneway
class ABTest:
    def __init__(self, data):
        self.data = data
    def chi_squared_test(self, group_col, metric_col):
        contingency_table = pd.crosstab(self.data[group_col], self.data[metric_col])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        return chi2, p_value
    def t_test(self, group_col, metric_col, group_A, group_B):
        group_A_data = self.data[self.data[group_col] == group_A][metric_col]
        group_B_data = self.data[self.data[group_col] == group_B][metric_col]
        
        # Perform a t-test
        t_stat, p_value = ttest_ind(group_A_data, group_B_data)
        
        return t_stat, p_value
    def anova_test(self, group_col, metric_col):
        # Perform ANOVA across multiple groups
        groups = [self.data[self.data[group_col] == group][metric_col] for group in self.data[group_col].unique()]
        f_stat, p_value = f_oneway(*groups)
        
        return f_stat, p_value