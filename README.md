# Charles-Carneiro-Thesis-Online-Appendix-
This repository contains the code and data used for the thesis: “Securitised Safe Assets in the Euro Area: A Monte Carlo Analysis of European Sovereign Bonds” by Charles Carneiro, University of St. Gallen (HSG), November 2025.

The appendix provides complete transparency of the modelling pipeline, including:
Data preprocessing and weighting logic,
Simulation code for baseline, contagion, and stress scenarios,

The simulations extend Brunnermeier et al. (2017)’s ESBies model using recent data (2010–2024) and introduce t-copula dependence, contagion dynamics, and systemic stress calibration.

The raw data are found in these files: LSEG_motnhly_PDs.xlsx for the PDs, FINAL_debt_outstanding_2010_2025,csv for the debt oustanding per country, Market_access_2010_2025_with_NonEUroAreaPlayers.xlsx for market access and monthly_MA5_gdp.csv for the 5y moving average GDP.

Merging_all.py merges the raw dataframes and returns merged_df.csv. In turn, the pool's weight are calculated in weight.py which returns the file PDs+weights.csv, as well as final_weights_k25 and capped_flags.py.

The simulation are run in the files MC_t_copula.py (model 1), MC_tcopula_contagiondefaultGEFRITES.py (model 2), and MC_tcopula_stress.py (model 3). The respective ressults are shwon in the MC_results_ csv files.
