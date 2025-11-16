#%% importing libraries
import pandas as pd
import numpy as np

# %% importing datasets
#PDs = pd.read_csv("data /PDs.csv", index_col=0) --> old Pds file
PDs = pd.read_excel("data /LSEG_Monthly_PDs.xlsx", index_col=0)
gdp = pd.read_csv("data /monthly_MA5_gdp.csv", index_col=0)
debt = pd.read_csv("data /FINAL_debt_outstanding_2010_2025.csv", index_col=0)
Mark_Access = pd.read_excel("data /Market_Access_2010_2025_with_NonEuroAreaPlayers.xlsx", index_col=0)

# %% Cleaning data
debt = debt.drop(columns=['TIME PERIOD'])
debt.index = pd.to_datetime(debt.index).strftime("%Y-%m")

PDs.index = pd.to_datetime(PDs.index).strftime("%Y-%m")
PDs = PDs.sort_index()
# %% Keeping only the relevant dates and countries 

#Dates until 2024-12, as there is no GDP data after that
PDs = PDs[PDs.index <= '2024-12']
gdp = gdp[gdp.index <= '2024-12']
debt = debt[debt.index <= '2024-12']
Mark_Access = Mark_Access[Mark_Access.index <= '2024-12']

#Keep only countries that are in all datasets, i.e. drop Luxembourg and Malta as there is no CDS data for them
#debt = debt.drop(columns=['Luxembourg', 'Malta'])
#gdp = gdp.drop(columns=['Luxembourg MA-5', 'Malta MA-5'])
#Mark_Access = Mark_Access.drop(columns=['Luxembourg', 'Malta'])

# %% Merging all dataframes
# Create a new column for every country called per dataframe to differentiate them after merging
debt = debt.rename(columns={col: f"{col} Debt" for col in debt.columns})
PDs = PDs.rename(columns={col: f"{col} PDs" for col in PDs.columns})
Mark_Access = Mark_Access.rename(columns={col: f"{col} Market Access" for col in Mark_Access.columns})

merged_df = debt.join([gdp, PDs, Mark_Access], how='outer')

#%%
merged_df.to_csv("data /merged_df.csv")


