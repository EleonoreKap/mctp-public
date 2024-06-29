#%%###########################################
##                 IMPORTS                  ##
##############################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import time
import datetime as dt
import calendar
import pickle

plt.style.use("bmh")

tic = time.time()


#%%###########################################
##          SIMULATION PARAMETERS           ##
##############################################

epsilon = 50 #ppm


#%%###########################################
##          SOME HELPER FUNCTIONS           ##
##############################################

def find_nearest(array, value):
    '''Rounds a given value to the nearest point in a given array.'''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def days_in_year(year):
    return 365 + calendar.isleap(year)


#%%###########################################
##               FORCERS DATA               ##
##############################################

# simple exponential decay forcers
forcers_data = pd.read_csv("slcf_data.csv",).set_index("Name")
forcers_data
forcers_data = forcers_data.loc[[
    #'carbon dioxide',
    #'methane', 'nitrous oxide',
    #'isoprene', 'ammonia', 'sulfur dioxide', 'acetaldehyde',
    #'trichloroethylene', 'methylethylketone', 'black carbon', 'benzene',
    #'tropospheric ozone', 'ethanol', 'acetone', 'carbon monoxide',
    #'HFC-161', 'HFC-152', 'HCFC-151', 'HCFC-141a', 'HCFC-131',
    #'methylbromide', 'HCFC-31'
    "ethanol"
    ]]

n_forcers = len(forcers_data)

## CO2 atmospheric decay parameters
a_0 = 2.123e-01 # [unitless] % Joos et al., 2013, SI (PD100, no climate feedbacks)
a_1 = 2.444e-01 # [unitless] 
a_2 = 3.360e-01 # [unitless] 
a_3 = 2.073e-01 # [unitless] 

alpha_1 = 3.364e+02 # [years]  
alpha_2 = 2.789e+01 # [years] 
alpha_3 = 4.055e+00 # [years] 

A_CO2_kg = 1.671361653979217e-15  #[Wm-2kg-1]
A_CO2_ppm = 1.3086e-02 # [W m2 ppm CO21] radiative efficiency of CO2

#ozone
A_ozone_kg = 1.124e-12
lifetime_ozone = 23.4/365

forcers_data["Lifetime [years]"] = forcers_data["Lifetime [days]"]/365


#%%###########################################
##                 SSP DATA                 ##
##############################################
# everything is called RCP to avoid changing names everywhere, but it is SSP

t_in = 2024

# choose scenario
scenario_name = "SSP3-7.0"

def read_SSP_data(scenario_name):
    # get data
    RCP_temperatures = pd.read_csv(f"SSP/temp-{scenario_name}.csv",
                index_col=0, skiprows= 1,
                names=["Year","Temperature"])

    RCP_concentrations = pd.read_excel(
        "SSP/CO2-ppm.xlsx", sheet_name=scenario_name,
        usecols="A:B", skiprows=12, header=None, index_col=0,
        names=["Year","RCP concentration"]
        )

    RCP_dataframe = pd.concat([RCP_temperatures,RCP_concentrations],axis=1)


    # start date and end dates
    RCP_dataframe.drop(RCP_dataframe[RCP_dataframe.index<t_in].index, inplace=True)
    RCP_dataframe.dropna(inplace=True)
    last_year_RCP = RCP_dataframe.index[-1]


    # maxima
    max_temp_RCP = RCP_dataframe["Temperature"].max()
    max_conc_RCP = RCP_dataframe["RCP concentration"].max()
    ini_temp_RCP = RCP_dataframe["Temperature"].values[0]


    return RCP_dataframe, max_temp_RCP, max_conc_RCP, ini_temp_RCP, last_year_RCP

RCP_dataframe, max_temp_RCP, max_conc_RCP, ini_temp_RCP, last_year_RCP = read_SSP_data(scenario_name)

last_year_RCP = RCP_dataframe.index[-1]
argmax_temp_RCP = RCP_dataframe[RCP_dataframe["Temperature"]>=max_temp_RCP].index[0]




#%%###########################################
##            USEFUL DATE TABLE             ##
##############################################

# approximately 5 seconds, improve efficiency

date_table = pd.DataFrame()
date_table["Year"] = range(t_in,2301)
date_table['Number days in year'] = date_table["Year"].map(days_in_year)
df = pd.DataFrame(index=[365, 366])
df["Day in year"]=df.index.map(lambda index: range(1,index+1))
date_table = date_table.join(df, on="Number days in year")
date_table = date_table.explode("Day in year")
date_table['Day in year'] = date_table['Day in year'].astype(str).str.pad(3,fillchar='0')

'''date_table["Date"] = date_table.apply(
    lambda row: dt.datetime.strptime(str(row["Year"])+'-'+str(row["Day in year"]), '%Y-%j').strftime('%Y-%m-%d'),
    axis=1
    )'''
date_table["Date as fraction"] = date_table["Year"] + (date_table["Day in year"].astype(int)-1) / date_table["Number days in year"]

date_table.reset_index(drop=True, inplace=True, names="Day index")
#date_table.drop("Number days in year", axis=1, inplace=True)
date_table.drop(date_table[
    (date_table["Year"] == date_table["Year"].max())
    & (date_table["Day in year"].astype(int) > 1)
    ].index, inplace=True) # stops at 1st day of 2300 - useful for interpolation
date_table




#%%###########################################
##           TIPPING POINTS DATA            ##
##############################################

tipping_points_original_data = pd.DataFrame({
    "Name":["Greenland IS", "West Antarctic IS", "Marine East Antarctic IS", "Non marine East Antarctic IS",
            "Antarctic SI", "Glaciers","Land permafrost",
            "Amazon F", "Boreal F S", "Boreal F N", "Temperate F",
            "Drylands", "Coral Reefs", "Mangroves", "Marine communities",
            "AMOC", "SPG", "S Ocean Circulation",
            "WA Monsoon"
            ], 
    "Tipping point id":range(1,20),
    "Temp. triang. lower":    [0.8, 1.0, 2, 6.0,  np.nan, 1.5, 1.0, 2,   1.4, 1.5, np.nan, np.nan, 1.0, np.nan, np.nan, 2.8, 1.1, 1.75, 2.9],
    "Temp. triang. central":  [2.1, 1.5, 3, 7.5,  np.nan, 2.0, 1.5, 3.5, 4.0, 4.0, np.nan, np.nan, 1.2, 1.5,    np.nan, 4.0, 1.8, 2.28, 3.7],
    "Temp. triang. upper":    [3.0, 3.0, 6, 10.0, np.nan, 3.0, 2.3, 6,   5.0, 7.2, np.nan, np.nan, 1.5, np.nan, np.nan, 5.2, 3.8, 3.00, 4.4]
    }).set_index("Tipping point id")


def special_cases_tipping_thresholds(RCP_dataframe, max_conc_RCP, ini_temp_RCP, max_temp_RCP):

    tipping_points_fixed_data = tipping_points_original_data.copy()

    # Antarctic Sea Ice
    # It is given in CO2 ppm, depending on the SSP it can be converted to a temperature
    ASI_lower_ppm = 770
    ASI_upper_ppm = 1112

    if max_conc_RCP >= ASI_lower_ppm: #lower is reachable
        ASI_lower_temp = RCP_dataframe[RCP_dataframe["RCP concentration"]>=ASI_lower_ppm]["Temperature"].values[0]

        if max_conc_RCP >= ASI_upper_ppm: # lower and upper reachable
            ASI_upper_temp = RCP_dataframe[RCP_dataframe["RCP concentration"]>=ASI_upper_ppm]["Temperature"].values[0]
        else: # only lower reachable
            ASI_upper_temp = (max_temp_RCP - ASI_lower_temp)*(ASI_upper_ppm-ASI_lower_ppm)/(max_conc_RCP-ASI_lower_ppm)+ASI_lower_temp # same scale as for ppm

    else: # none reachable
        ASI_lower_temp = max_temp_RCP+1
        ASI_upper_temp = max_temp_RCP+2


    ASI_id = tipping_points_fixed_data[tipping_points_fixed_data["Name"]=="Antarctic SI"].index[0]
    tipping_points_fixed_data.loc[ASI_id,"Temp. triang. lower"] = ASI_lower_temp
    tipping_points_fixed_data.loc[ASI_id,"Temp. triang. upper"] = ASI_upper_temp
    tipping_points_fixed_data.loc[ASI_id,"Temp. triang. central"] = (ASI_lower_temp+ASI_upper_temp)/2

    # Mangroves
    # Only has a central estimate
    mean_dist_to_min = (tipping_points_fixed_data["Temp. triang. central"] - tipping_points_fixed_data["Temp. triang. lower"] ).mean()
    mean_dist_to_max = (tipping_points_fixed_data["Temp. triang. upper"] - tipping_points_fixed_data["Temp. triang. central"] ).mean()
    mangroves_id = tipping_points_fixed_data[tipping_points_fixed_data["Name"]=="Mangroves"].index[0]
    tipping_points_fixed_data.loc[mangroves_id,"Temp. triang. lower"] = tipping_points_fixed_data.loc[mangroves_id,"Temp. triang. central"] - mean_dist_to_min
    tipping_points_fixed_data.loc[mangroves_id,"Temp. triang. upper"] = tipping_points_fixed_data.loc[mangroves_id,"Temp. triang. central"] + mean_dist_to_max


    # To avoid setting tipping points in the past
    for tipping_point_id in tipping_points_fixed_data.index:
        if tipping_points_fixed_data.loc[tipping_point_id, "Temp. triang. lower"] < ini_temp_RCP:
            tipping_points_fixed_data.loc[tipping_point_id, "Temp. triang. lower"] = ini_temp_RCP
            if tipping_points_fixed_data.loc[tipping_point_id, "Temp. triang. central"] < ini_temp_RCP:
                tipping_points_fixed_data.loc[tipping_point_id, "Temp. triang. central"] = (tipping_points_fixed_data.loc[tipping_point_id,"Temp. triang. lower"] + tipping_points_fixed_data.loc[tipping_point_id,"Temp. triang. upper"])/2

    return tipping_points_fixed_data.dropna()

tipping_points_fixed_data = special_cases_tipping_thresholds(RCP_dataframe, max_conc_RCP, ini_temp_RCP, max_temp_RCP)
tipping_points_fixed_data

#%%###########################################
##       TIPPING POINTS IMPACTS DATA        ##
##############################################

#tipping_impacts_path = "Serenas_CTP_calc_data_v4.xlsx"
tipping_impacts_path = "Tipping_impacts.xlsx"
tipping_impacts_dataframe = pd.read_excel(tipping_impacts_path, sheet_name='Data',
                     usecols="G:Y", skiprows=6, nrows=500, header=None)
tipping_impacts_dataframe.columns = range(1,20)
tipping_impacts_dataframe.index = range(500) # first row of the dataframe is year 0, year of tipping = year of impact
tipping_impacts_dataframe


#%%###########################################
##     RANDOMIZE TEMPERATURE THRESHOLDS     ##
##############################################

def randomize_temperature_thresholds(tipping_points_original_data:pd.DataFrame, RCP_dataframe:pd.DataFrame):

    """
    Takes the original tipping points data and returns a new dataframe with the selected tipping points along with their trigger temperature (picked at random).
    
    Parameters:
    tipping_points_original_data (pd.DataFrame): DataFrame containing information on probability distributions of trigger temperatures of tipping points.
    RCP_dataframe_daily_timestep (pd.DataFrame): RCP temperature and concentration

    Returns:
    selected_tipping_points_dataframe (pd.DataFrame): DataFrame containing information on tipping points with a new "Random temperature" column.
    
    Assumes triangular distribution of probabilities within the original range of probable trigger temperatures.
    Tipping points whose selected temperature lies outside of the attainable temperatures within the chosen RCP are discarded.
    """

    selected_tipping_points_dataframe = tipping_points_original_data.copy() # Copy the original dataframe to avoid modifying it
    RCP_dataframe_short = RCP_dataframe.loc[:argmax_temp_RCP]

    selected_tipping_points_dataframe["Temp. triang. loc"] = selected_tipping_points_dataframe["Temp. triang. lower"]
    selected_tipping_points_dataframe["Temp. triang. scale"] = selected_tipping_points_dataframe["Temp. triang. upper"] - selected_tipping_points_dataframe["Temp. triang. lower"]
    selected_tipping_points_dataframe["Temp. triang. c"] = (selected_tipping_points_dataframe["Temp. triang. central"] - selected_tipping_points_dataframe["Temp. triang. lower"]) / selected_tipping_points_dataframe["Temp. triang. scale"]

    selected_tipping_points_dataframe["Random temperature"] = np.nan # Initialize a column for random temperatures

    # Iterate over each tipping point in the dataframe
    for tipping_point_id in selected_tipping_points_dataframe.index:
        
        c, loc, scale = selected_tipping_points_dataframe.loc[tipping_point_id][["Temp. triang. c", "Temp. triang. loc", "Temp. triang. scale"]]
        triang_pd = stats.triang(c, loc, scale)
        temperature = triang_pd.rvs() # Generate a random temperature

        if temperature < max_temp_RCP:  # Check if the temperature falls within the attainable temperatures in the chosen RCP
            temperature = find_nearest(RCP_dataframe_short["Temperature"], temperature) # Find the nearest attainable temperature in the RCP
            while any(temperature == t for t in selected_tipping_points_dataframe["Random temperature"]): # Check if the temperature has already been selected for another tipping point
                temperature = triang_pd.rvs() # If so, generate a new random temperature
                if temperature < max_temp_RCP:
                    temperature = find_nearest(RCP_dataframe_short["Temperature"], temperature)
                else:
                    temperature = np.nan  # Set temperature to NaN if it exceeds the atteinable temperatures, since the tipping point cannot be triggered
        else:
            temperature = np.nan

        selected_tipping_points_dataframe.loc[tipping_point_id,"Random temperature"] = temperature

    selected_tipping_points_dataframe = selected_tipping_points_dataframe.dropna() # Discard tipping points with NaN temperature

    return selected_tipping_points_dataframe.drop(["Temp. triang. c","Temp. triang. loc","Temp. triang. scale"], axis=1)

selected_tipping_points_dataframe = randomize_temperature_thresholds(tipping_points_fixed_data, RCP_dataframe)
selected_tipping_points_dataframe


#%%###########################################
##       FIND ORDER OF TIPPING POINTS       ##
##          RETRIEVE CONCENTRATION          ##
##############################################
    
def order_tipping_points(selected_tipping_points_dataframe:pd.DataFrame, RCP_dataframe:pd.DataFrame):
    """
    Assigns an order to selected tipping points based on the order of their trigger temperatures,
    and joins the corresponding year and concentration in the given RCP, based on these temperatures.
    
    Parameters:
    selected_tipping_points_dataframe (pd.DataFrame): DataFrame containing selected tipping points and their trigger temperatures.
    RCP_dataframe_daily_timestep (pd.DataFrame): RCP temperature and concentration
    
    Returns:
    selected_tipping_points_dataframe[["Order", "RCP concentration"]] (pd.DataFrame): DataFrame with order and concentration for each tipping point.
    """

    selected_tipping_points_dataframe["Order"] = selected_tipping_points_dataframe["Random temperature"].rank().astype(int) # Assign order based on the rank of trigger temperatures
    selected_tipping_points_dataframe["RCP concentration"] = selected_tipping_points_dataframe["Random temperature"].map(
        lambda temp: RCP_dataframe[RCP_dataframe["Temperature"]==temp]["RCP concentration"].values[0]
        )

    return selected_tipping_points_dataframe[["Order", "RCP concentration"]]

selected_tipping_points_dataframe[["Order", "RCP concentration"]] = order_tipping_points(selected_tipping_points_dataframe, RCP_dataframe)
selected_tipping_points_dataframe


#%%###########################################
##       COMPUTE CONCENTRATIONS WITH        ##
##       TIPPING IMPACTS AND COMPUTE        ##
##       TIPPING YEAR ACCOUNTING FOR        ##
##           INCREASED BACKGROUND           ##
##############################################

def compute_concentrations_with_tipping_impacts_and_rectify_year(RCP_dataframe:pd.DataFrame, selected_tipping_points_dataframe:pd.DataFrame, tipping_impacts_dataframe:pd.DataFrame, include_tipping_impacts:bool=True):
    """
    Computes the year for each tipping point, as the year where the sum of the RCP and the cascading effects from previous tipping points reaches the threshold.

    Parameters:
    tipping_points_original_data (pd.DataFrame): DataFrame containing information on probability distributions of trigger temperatures of tipping points.
    RCP_dataframe_daily_timestep (pd.DataFrame): RCP temperature and concentration
    tipping_impacts_dataframe (pd.DataFrame): DataFrame containing data of cascading effects
    include_tipping_impacts (bool): where to include cascading effects
    """
    
    n_tipping_points = len(selected_tipping_points_dataframe) # Number of selected tipping points

    RCP_including_impacts = RCP_dataframe.copy()
    RCP_including_impacts["Concentration including impacts"] = RCP_including_impacts["RCP concentration"] # Initialize column for updated concentrations with impacts with original concentrations
    selected_tipping_points_dataframe["Year (with tipping impacts)"] = np.nan # Initialize column for rectified years

    # Iterate over each tipping point by its order
    for tipping_point_rank in range(1, n_tipping_points+1):
        tipping_point_id = selected_tipping_points_dataframe[selected_tipping_points_dataframe["Order"] == tipping_point_rank].index[0] # Get tipping point ID
        concentration_at_tipping = selected_tipping_points_dataframe.loc[tipping_point_id]["RCP concentration"] # Get concentration at tipping point

        # Compute day, year, and day in year of tipping
        year = RCP_including_impacts[concentration_at_tipping <= RCP_including_impacts["Concentration including impacts"]].index[0] # Find the rectified year as the first year where the background concentration (including impacts) exceeds the tipping concentration

        # Store day, year, and day in year of tipping
        selected_tipping_points_dataframe.loc[tipping_point_id,"Year (with tipping impacts)"] = year

        # Tipping impacts
        RCP_including_impacts.loc[year:, "Concentration including impacts"] += tipping_impacts_dataframe[tipping_point_id][:last_year_RCP-year+1].to_numpy() # Update background concentration from the rectified year onward, adding the impact of the tipping point

        #if (impacts_one_tipping_point[tipping_point_id]>0).sum()>0: ## for plotting only
        #    test.append(RCP_including_impacts["Concentration including impacts"].rename(tipping_point_rank))

    return selected_tipping_points_dataframe["Year (with tipping impacts)"], RCP_including_impacts.drop("Temperature", axis=1)

(
    selected_tipping_points_dataframe["Year (with tipping impacts)"],
    RCP_including_impacts
    ) = compute_concentrations_with_tipping_impacts_and_rectify_year(
        RCP_dataframe, selected_tipping_points_dataframe, tipping_impacts_dataframe,
        )

#RCP_including_impacts
selected_tipping_points_dataframe.dropna()

#%%###########################################
##     COMPUTE INSTANTANEOUS CAPACITIES     ##
##############################################

def compute_instantaneous_capacities(RCP_including_impacts:pd.DataFrame, selected_tipping_points_dataframe:pd.DataFrame):
    """
    For each tipping point, computes the carrying capacity over time,
    given as the difference between the tipping concentration and the background concentration
    
    Parameters:
    RCP_including_impacts (pd.DataFrame): DataFrame containing computed the background concentration including tipping impacts over time.
    selected_tipping_points_dataframe (pd.DataFrame): DataFrame containing selected tipping points and associated information.

    Returns:
    RCP_including_impacts (pd.DataFrame): carrying capacity over time of each tipping point are added to the DataFrame.
    """

    n_tipping_points = len(selected_tipping_points_dataframe) # Number of selected tipping points
    capacities = pd.DataFrame(index = RCP_including_impacts.index,
                              columns = [f"capacityTP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1)]
                              )

    # Iterate over each tipping point by its order
    for tipping_point_rank in range(1,n_tipping_points+1):
        tipping_point_id = selected_tipping_points_dataframe[selected_tipping_points_dataframe["Order"] == tipping_point_rank].index[0] # Get tipping point ID
        year = selected_tipping_points_dataframe.loc[tipping_point_id]["Year (with tipping impacts)"] # Get day

        concentration_at_tipping = selected_tipping_points_dataframe.loc[tipping_point_id]["RCP concentration"] # Get concentration at tipping point

        capacities[f"capacityTP{tipping_point_rank}"] = (concentration_at_tipping - RCP_including_impacts["Concentration including impacts"])[RCP_including_impacts.index < year] # Compute difference between tipping concentration and background concentration (including tipping impacts), for each year until the rectified tipping year
        if year>t_in:
            smallest_capacity = capacities.loc[year-1,f"capacityTP{tipping_point_rank}"]
            capacities[f"capacityTP{tipping_point_rank}"] = capacities[f"capacityTP{tipping_point_rank}"]- smallest_capacity
    return capacities.reset_index(drop=True) 

instantaneous_capacities = compute_instantaneous_capacities(RCP_including_impacts, selected_tipping_points_dataframe)

instantaneous_capacities.plot()





#%%###########################################
##  COMPUTE INSTANTANEOUS EMISSIONS IMPACT  ##
##############################################

# impact is the radiative forcing cause by a pulse emission
# (radiative efficiency * abundance defined by the impulse response function)

# storing the impacts for all dates, all tipping points, all forcers, requires a very large dataframe, Python can complain that it's fragmented; ignore the warning.

def compute_instantaneous_emission_impacts(emission_times:pd.Index, include_ozone:bool=True):
    """
    Computes emission impacts for all forcers as instantaneous radiative forcing
    
    Parameters:
    emission_times (pd.Index): Index containing emission days for which we want to compute impacts.
    selected_tipping_points_dataframe (pd.DataFrame): DataFrame containing selected tipping points and associated information.
    include_ozone (bool): whether to include indirect ozone impacts.

    Returns:
    pd.impacts_daily_timestep (pd.DataFrame): impacts over time for each tipping point and each forcer
    """

    if include_ozone:
        instantaneous_impacts = pd.DataFrame(#index=emission_times,
                                            columns=([f"Impact {forcer_name}" for forcer_name in forcers_data.index]
                                                    + [f"Impact {forcer_name} with ozone" for forcer_name in forcers_data.index]
                                                    ))
    else:
        instantaneous_impacts = pd.DataFrame(#index=emission_times,
                                            columns=([f"Impact {forcer_name}" for forcer_name in forcers_data.index]
                                                    ))
    # iterate over each simple forcer
    for forcer_id in range(n_forcers):
        '''if forcers_data.iloc[forcer_id].name=="CO2":
            instantaneous_impactsf"Impact CO2 TP{tipping_point_rank}"] = emission_times.map(lambda emission_time: compute_impact_CO2_model(emission_time, rectified_year)) # Apply the 'compute_impact_CO2_model' function to obtain the impact of CO2 on each tipping point over all emission years
        else:'''
        forcer_name = forcers_data.iloc[forcer_id].name # Get forcer name
        A_forcer_kg = forcers_data["Radiative efficiency [W m-2 kg-1]"][forcer_name] # Get forcer radiative efficiency
        lifetime_forcer = forcers_data["Lifetime [years]"][forcer_name] # Get forcer lifetime
        MIR_forcer = forcers_data["MIR [g_ozone/g_compound]"][[forcer_name]].fillna(0).values[0] # Get forcer MIR

        if forcer_name=="carbon dioxide":
            arg_of_exp1 = (- pd.Series(range(len(emission_times))) / alpha_1); term1 = a_1 * arg_of_exp1.loc[arg_of_exp1 <= 0].rpow(np.e)
            arg_of_exp2 = (- pd.Series(range(len(emission_times))) / alpha_2); term2 = a_2 * arg_of_exp2.loc[arg_of_exp2 <= 0].rpow(np.e)
            arg_of_exp3 = (- pd.Series(range(len(emission_times))) / alpha_3); term3 = a_3 * arg_of_exp3.loc[arg_of_exp3 <= 0].rpow(np.e)
            instantaneous_impacts[f"Impact {forcer_name}"] = A_CO2_kg * (a_0 + term1 + term2 + term3) / A_CO2_ppm
        else:
            arg_of_exp = (- pd.Series(range(len(emission_times))) / lifetime_forcer)
            instantaneous_impacts[f"Impact {forcer_name}"] = A_forcer_kg * arg_of_exp.loc[arg_of_exp <= 0].rpow(np.e) / A_CO2_ppm

        if include_ozone:
            instantaneous_impacts[f"Impact {forcer_name} with ozone"] = instantaneous_impacts[f"Impact {forcer_name}"]
            arg_of_exp_with_ozone = (- pd.Series(range(len(emission_times))) / lifetime_ozone)
            instantaneous_impacts.loc[arg_of_exp_with_ozone <= 0, f"Impact {forcer_name} with ozone"] += MIR_forcer * A_ozone_kg * arg_of_exp_with_ozone.loc[arg_of_exp_with_ozone <= 0].rpow(np.e) / A_CO2_ppm

    return instantaneous_impacts

instantaneous_impacts = compute_instantaneous_emission_impacts(RCP_dataframe.index, include_ozone=False)
instantaneous_impacts.plot()


#%%###########################################
##         COMPUTE INSTANTANEOUS CTP        ##
##############################################

# 47s for 1 forcer, all TP

def compute_instantaneous_CTP(instantaneous_capacities:pd.DataFrame, instantaneous_impacts:pd.DataFrame, n_tipping_points:int, epsilon:int=0, include_ozone:bool=True):
    """
    Computes Climate Tipping Potential (CTP) values for each tipping point and each forcer, based on computed impacts and capacities.
    
    Parameters:
    capacities_daily_timestep:pd.DataFrame: capacities at each timestep for all tipping points
    impacts_daily_timestep:pd.DataFrame: impacts at each time step for all tipping points and all forcers
    n_tipping_points:int: number of tippingp oints in the selected tipping points dataframe
    epsilon:int: a convergence parameter (default: 0) to be added to the denominator of the computation for avoiding diverging ctp
    include_ozone:bool: whether to include indirect impacts from ozone formation
    
    Returns:
    CTP (pd.DataFrame): CTP values over time are added to the DataFrame for each tipping point.
    """

    if not include_ozone:
        CTP = pd.DataFrame(index = instantaneous_impacts.index,
                       columns=([f"CTP {forcer_name} TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1) for forcer_name in forcers_data.index]
                                ))
        
    else:
        CTP = pd.DataFrame(index = instantaneous_impacts.index,
                       columns=([f"CTP {forcer_name} TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1) for forcer_name in forcers_data.index]
                                + [f"CTP {forcer_name} with ozone TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1) for forcer_name in forcers_data.index]
                                ))
        
    n_days = len(instantaneous_capacities.index)

    # Iterate over each tipping point by its order
    for tipping_point_rank in range(1,n_tipping_points+1): #attention#
        year = selected_tipping_points_dataframe.loc[selected_tipping_points_dataframe["Order"]==tipping_point_rank, "Year (with tipping impacts)"].values[0]-t_in

        # Iterate over each simple forcer
        for forcer_id in range(n_forcers): #attention#
            forcer_name = forcers_data.iloc[forcer_id].name

            capa_inv = pd.Series(index=instantaneous_capacities.index)
            capa_inv.loc[instantaneous_capacities[f"capacityTP{tipping_point_rank}"]>0] = 1/(instantaneous_capacities[f"capacityTP{tipping_point_rank}"]+epsilon)
            capa_inv.fillna(0, inplace=True)

            capa_inv_flip = np.flip(capa_inv.to_numpy())

            CTP_flip = np.convolve(instantaneous_impacts[f"Impact {forcer_name}"], capa_inv_flip)
            CTP[f"CTP {forcer_name} TP{tipping_point_rank}"] = np.flip(CTP_flip)[n_days-1:]
            CTP[f"CTP {forcer_name} TP{tipping_point_rank}"] = CTP[f"CTP {forcer_name} TP{tipping_point_rank}"] / (pd.Series(year-instantaneous_capacities.index).map(lambda x: max(x,0)))

            if include_ozone:
                CTP_flip = np.convolve(instantaneous_impacts[f"Impact {forcer_name} with ozone"], capa_inv_flip)
                CTP[f"CTP {forcer_name} with ozone TP{tipping_point_rank}"] = np.flip(CTP_flip)[n_days-1:]
                CTP[f"CTP {forcer_name} with ozone TP{tipping_point_rank}"] = CTP[f"CTP {forcer_name} with ozone TP{tipping_point_rank}"] / (pd.Series(year-instantaneous_capacities.index).map(lambda x: max(x,0)))


    return CTP #CTP.join(date_table).groupby("Year")[[column for column in CTP.columns if "CTP" in column]].mean()

CTP = compute_instantaneous_CTP(instantaneous_capacities, instantaneous_impacts,
                                len(selected_tipping_points_dataframe), epsilon=0.01, include_ozone=False
                                )
CTP.plot()

#%%###########################################
##               COMPUTE MCTP               ##
##############################################
  

def compute_MCTP(CTP:pd.DataFrame, n_tipping_points:int, include_ozone:bool=True): # shouldn't we divide by the number of tipping points?
    """
    Computes Multi Climate Tipping Potential (MCTP) values for each forcer, as the sum of computed CTP values accross all tipping points.
    
    Parameters:
    CTP (pd.DataFrame): CTP dataframe
    n_tipping_points: number of selected tippingpoints

    Returns:
    MCTP_dataframe (pd.DataFrame): DataFrame with computed MCTP values for CO2 and each simple forcer.
    """

    MCTP = pd.DataFrame(index = CTP.index)

    # Iterate over each simple forcer
    for forcer_id in range(n_forcers):
        '''if forcers_data.iloc[forcer_id].name=="CO2":
            MCTP["MCTP CO2"] = CTP[[f"CTP CO2 TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1)]].sum(min_count=1, axis=1) # Compute MCTP values for CO2 by summing CTP values for each tipping point
        else:'''
        forcer_name = forcers_data.iloc[forcer_id].name # Get forcer name
        MCTP[f"MCTP {forcer_name}"] = CTP[[f"CTP {forcer_name} TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1)]].sum(min_count=1, axis=1)

        if include_ozone == True:
            MCTP[f"MCTP {forcer_name} with ozone"] = CTP[[f"CTP {forcer_name} with ozone TP{tipping_point_rank}" for tipping_point_rank in range(1,n_tipping_points+1)]].sum(min_count=1, axis=1)


    # Return only the MCTP columns
    return(MCTP[MCTP>0])

MCTP = compute_MCTP(CTP, len(selected_tipping_points_dataframe), include_ozone=False)
#MCTP = MCTP.reset_index().set_index(date_table["Date as fraction"])
MCTP.plot()




#%%###########################################
##                   MAIN                   ##
##############################################

#Parameters
n_iterations = 2
epsilon = 1
include_ozone = False
str_ozone = " with ozone" if include_ozone else ""
include_tipping_impacts = True

# To save
save = True
save_fig = False
date = "26-06"
## First create folders with the desired names on your computer: "Results/{date}-{scenario_name}-noozone" and "Results/{date}-{scenario_name}-with-ozone"


# Main
for scenario_name in [
    "SSP1-1.9",
    "SSP1-2.6",
    "SSP2-4.5",
    "SSP3-7.0",
    "SSP5-8.5"
    ]:

    tipping_points = tipping_points_original_data.copy()

    # read RCP data
    RCP_dataframe, max_temp_RCP, max_conc_RCP, ini_temp_RCP, last_year_RCP = read_SSP_data(scenario_name)
    # convert to daily timestep
    # RCP_dataframe_daily_timestep, last_day_RCP_daily_timestep, argmax_temp_RCP = convert_RCP_daily_timestep(RCP_dataframe)
    last_year_RCP = RCP_dataframe.index[-1]
    argmax_temp_RCP = RCP_dataframe[RCP_dataframe["Temperature"]>=max_temp_RCP].index[0]

    # Thresholds of tipping points out of bounds
    tipping_points_fixed_data = special_cases_tipping_thresholds(RCP_dataframe, max_conc_RCP, ini_temp_RCP, max_temp_RCP)

    ######### Monte-Carlo starts here #########
    tic = time.time()

    list_of_MCTP_for_all_forcers = [[] for _ in range (n_forcers)] # list that contains Multi Climate Tipping Potential of other forcers emissions over time, at each iteration
    if include_ozone:
        list_of_MCTP_for_all_forcers_with_ozone = [[] for _ in range (n_forcers)]

    print(f"starting iteration no. 0, time: {time.time()-tic} sec.")

    # Iterate n_iterations times
    for it in range(n_iterations):

        selected_tipping_points_dataframe = randomize_temperature_thresholds(tipping_points_fixed_data, RCP_dataframe) # Randomize temperature thresholds and determine selected tipping points
        selected_tipping_points_dataframe[["Order", "RCP concentration"]] = order_tipping_points(selected_tipping_points_dataframe, RCP_dataframe) # Assign an order to tipping points and joins corresponding temperature and year

        selected_tipping_points_dataframe["Year (with tipping impacts)"], RCP_including_impacts = compute_concentrations_with_tipping_impacts_and_rectify_year(RCP_dataframe, selected_tipping_points_dataframe, tipping_impacts_dataframe, include_tipping_impacts)
        
        tipping_points = pd.concat([
            tipping_points,
            selected_tipping_points_dataframe["Year (with tipping impacts)"].rename(it),
            selected_tipping_points_dataframe["RCP concentration"].rename(f"{it}-conc")
            ],axis=1)
        
        instantaneous_capacities = compute_instantaneous_capacities(RCP_including_impacts, selected_tipping_points_dataframe) # compute carrying capacity over time of each tipping element
        instantaneous_impacts = compute_instantaneous_emission_impacts(RCP_dataframe.index, include_ozone=include_ozone) # compute impacts of each forcer on each tipping point, over all emission years
        CTP = compute_instantaneous_CTP(instantaneous_capacities, instantaneous_impacts, len(selected_tipping_points_dataframe), epsilon=epsilon, include_ozone=include_ozone)
    
        MCTP = compute_MCTP(CTP, len(selected_tipping_points_dataframe), include_ozone) # compute the climate tipping potential of each forcer, over all emission years, as the sum of climate tipping potentials for individual tipping point

        # Iterate for each simple forcer
        for forcer_id in range (n_forcers):
            forcer_name = forcers_data.iloc[forcer_id].name
            list_of_MCTP_for_all_forcers[forcer_id].append(MCTP[f"MCTP {forcer_name}"].to_numpy()) # add last MCTP of the forcer computed to the list of MCTP of the forcer for all iterations
            if include_ozone:
                list_of_MCTP_for_all_forcers_with_ozone[forcer_id].append(MCTP[f"MCTP {forcer_name} with ozone"].to_numpy()) # idem with ozone
            
        if it%300==0:
            tac = time.time()
            print(f"end it. no. {it}, t.: {int(tac-tic)} sec., est. total t.: {int((tac-tic) * (n_iterations/(it+1)))} sec.")

    #last valid date
    max_date_valid = max([np.sum(~np.isnan(list_of_MCTP_for_all_forcers[0][it])) for it in range(n_iterations)])
    #cut to this max
    for forcer_id in range (n_forcers):
        for it in range(n_iterations):
            list_of_MCTP_for_all_forcers[forcer_id][it] = np.nan_to_num(list_of_MCTP_for_all_forcers[forcer_id][it][:max_date_valid])                
            if include_ozone:
                list_of_MCTP_for_all_forcers_with_ozone[forcer_id][it] = np.nan_to_num(list_of_MCTP_for_all_forcers_with_ozone[forcer_id][it][:max_date_valid])                
     

    # Compute mean, quantiles
    list_of_mean_for_all_forcers = np.mean(list_of_MCTP_for_all_forcers, axis=1)
    list_of_q5_for_all_forcers = np.quantile(list_of_MCTP_for_all_forcers, 0.05, axis=1)
    list_of_q95_for_all_forcers = np.quantile(list_of_MCTP_for_all_forcers, 0.95, axis=1)
    if include_ozone:
        list_of_mean_for_all_forcers_with_ozone = np.mean(list_of_MCTP_for_all_forcers_with_ozone, axis=1)
        list_of_q5_for_all_forcers_with_ozone = np.quantile(list_of_MCTP_for_all_forcers_with_ozone, 0.05, axis=1)
        list_of_q95_for_all_forcers_with_ozone = np.quantile(list_of_MCTP_for_all_forcers_with_ozone, 0.95, axis=1)

    # Create for each forcer a dataframe with mean and quantiles over time
    list_of_stats_for_all_forcers = []
    if include_ozone:
        list_of_stats_for_all_forcers_with_ozone = []
    for forcer_id in range(n_forcers):
        forcer_name = forcers_data.iloc[forcer_id].name
        stats_for_one_forcer = pd.DataFrame(
            [list_of_mean_for_all_forcers[forcer_id], list_of_q5_for_all_forcers[forcer_id], list_of_q95_for_all_forcers[forcer_id]],
            index=[f"Mean MCTP {forcer_name}", f"Q5 MCTP {forcer_name}", f"Q95 MCTP {forcer_name}"]
            ).transpose()
        stats_for_one_forcer = stats_for_one_forcer.reset_index().rename({"index":"Day index"}, axis=1)
        stats_for_one_forcer = stats_for_one_forcer.set_index(date_table.Year.drop_duplicates().values[:max_date_valid])
        list_of_stats_for_all_forcers.append(stats_for_one_forcer)
        if include_ozone:
            stats_for_one_forcer_with_ozone = pd.DataFrame(
                [list_of_mean_for_all_forcers_with_ozone[forcer_id], list_of_q5_for_all_forcers_with_ozone[forcer_id], list_of_q95_for_all_forcers_with_ozone[forcer_id]],
                index=[f"Mean MCTP {forcer_name} with ozone", f"Q5 MCTP {forcer_name} with ozone", f"Q95 MCTP {forcer_name} with ozone"]
                ).transpose()
            stats_for_one_forcer_with_ozone = stats_for_one_forcer_with_ozone.reset_index().rename({"index":"Day index"}, axis=1)
            stats_for_one_forcer_with_ozone = stats_for_one_forcer_with_ozone.set_index(date_table.Year.drop_duplicates().values[:max_date_valid])
            list_of_stats_for_all_forcers_with_ozone.append(stats_for_one_forcer_with_ozone)

    # Time
    print(f"Total time: {time.time()-tic} sec.")


    # Plot
    for forcer_id in range (n_forcers):
        forcer_name = forcers_data.iloc[forcer_id].name
        plt.figure(forcer_name)
        plt.plot(list_of_stats_for_all_forcers[forcer_id].index, list_of_stats_for_all_forcers[forcer_id][f"Q95 MCTP {forcer_name}"], label="Q95", color="black", linestyle="-.")
        plt.plot(list_of_stats_for_all_forcers[forcer_id].index, list_of_stats_for_all_forcers[forcer_id][f"Mean MCTP {forcer_name}"], label="mean", color="red", linestyle="-")
        plt.plot(list_of_stats_for_all_forcers[forcer_id].index, list_of_stats_for_all_forcers[forcer_id][f"Q5 MCTP {forcer_name}"], label="Q5", color="black", linestyle=":")
        plt.xlabel("Time (years)")
        plt.ylabel("MCTP")
        plt.title(f"{scenario_name}, {forcer_name} without ozone, {epsilon} ppm, {n_iterations}it")
        plt.legend()
        if save_fig:
            plt.savefig(f'Results/{date}-{scenario_name}-noozone-instantaneous/{forcer_name}-noozone.png')
            plt.close()
        else:
            plt.show()

        if include_ozone:
            plt.figure(forcer_name)
            plt.plot(list_of_stats_for_all_forcers_with_ozone[forcer_id].index, list_of_stats_for_all_forcers_with_ozone[forcer_id][f"Q95 MCTP {forcer_name} with ozone"], label="Q95", color="black", linestyle="-.")
            plt.plot(list_of_stats_for_all_forcers_with_ozone[forcer_id].index, list_of_stats_for_all_forcers_with_ozone[forcer_id][f"Mean MCTP {forcer_name} with ozone"], label="mean", color="red", linestyle="-")
            plt.plot(list_of_stats_for_all_forcers_with_ozone[forcer_id].index, list_of_stats_for_all_forcers_with_ozone[forcer_id][f"Q5 MCTP {forcer_name} with ozone"], label="Q5", color="black", linestyle=":")
            plt.xlabel("Time (years)")
            plt.ylabel("MCTP")
            plt.title(f"{scenario_name}, {forcer_name} with ozone, {epsilon} ppm, {n_iterations}it")
            plt.legend()
            if save_fig:
                plt.savefig(f'Results/{date}-{scenario_name}-withozone-instantaneous/{forcer_name}-withozone.png')
                plt.close()
            else:
                plt.show()

    if save:
        pd.concat(list_of_stats_for_all_forcers,axis=1,keys=forcers_data.index).to_csv(f'Results/{date}-{scenario_name}-noozone-instantaneous/{scenario_name}-mean-mctp.csv')
        
        if include_ozone:
            pd.concat(list_of_stats_for_all_forcers_with_ozone,axis=1,keys=forcers_data.index).to_csv(f'Results/{date}-{scenario_name}-withozone-instantaneous/{scenario_name}-mean-mctp.csv')
           



# %%
