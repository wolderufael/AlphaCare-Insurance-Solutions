# AlphaCare Insurance Solutions

This repository contains the analysis and optimization of marketing strategies for AlphaCare Insurance Solutions (ACIS), focused on car insurance policies. The goal of the project is to identify low-risk clients, reduce premiums, and improve overall customer segmentation through advanced data analytics, machine learning models, and predictive analysis.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Project Overview

This project involves optimizing the marketing strategy of ACIS through:

- **Exploratory Data Analysis (EDA)**: Understanding client demographics, car information, claims data, and insurance policies.
- **A/B Testing**: Assessing marketing campaigns and policy changes to determine effective strategies.
- **Predictive Analytics & Machine Learning**: Identifying low-risk clients, predicting claims, and recommending premium adjustments.
- **Correlation Analysis**: Exploring relationships between monthly premium changes and monthly claims, by province and other demographic factors.

## Data

The dataset includes:

- **Client Information**: Age, gender, and title (used to infer gender).
- **Car Information**: Make, model, and introduction date.
- **Insurance Policies**: Policy start and end dates, premium amounts, and claims history.
- **Claims Data**: Information about claims made, including monthly changes and claims severity.

Certain columns, such as `UnderwrittenCoverID`, `PolicyID`, `PostalCode`, and others, are excluded in univariate analysis.

## Exploratory Data Analysis (EDA)

EDA includes visualizations and insights drawn from:

- Monthly premium changes (`MonthlyPremiumChange`) vs monthly claims changes (`MonthlyClaimsChange`).
- Gender analysis: where unspecified titles are inferred based on common naming conventions (Mr = Male, Mrs/Ms/Miss = Female).
- Vehicle introduction date analysis by year range.
- Capital outstanding range, grouped in 10,000 units.
- Transaction month formatting for clear representation of seasonal trends.

## Modeling

Machine learning models are developed to:

- Predict customer risk and claims probability.
- Optimize premiums for low-risk clients.
- Evaluate A/B testing results to improve marketing strategy.

## Key Insights

- Correlations between monthly premium and claims changes by province.
- Gender and vehicle type influence on risk factors.
- Seasonal trends in policy renewals and claims.

## Technologies Used

- **Python**: Data analysis, modeling, and visualization.
- **Pandas & NumPy**: Data manipulation and transformation.
- **Matplotlib & Seaborn**: Visualizations for EDA.
- **Scikit-learn**: Machine learning model development.
- **Statsmodels**: A/B testing and statistical analysis.

## Usage

Clone the repository:

```bash
git clone https://github.com/wolderufael/AlphaCare-Insurance-Solutions.git
