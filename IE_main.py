import pip
print("Start importing...")

try:
    import pandas as pd
    print("pandas has been installed!")
except ModuleNotFoundError:
    print("pandas is installing")
    pip.main(["install", "pandas"])
    import pandas as pd
    print("pandas has been installed!")

try:
    import xlsxwriter
    print("xlsxwriter has been installed!")
except ModuleNotFoundError:
    print("xlsxwriter is installing")
    pip.main(["install", "xlsxwriter"])
    import pandas as pd
    print("xlsxwriter has been installed!")

try:
    import numpy as np
    print("numpy has been installed!") 
except ModuleNotFoundError:
    print("numpy is installing")
    pip.main(["install", "numpy"])
    import numpy as np
    print("numpy has been installed!")  

try:
    import warnings
    print("warnings has been installed!") 
except ModuleNotFoundError:
    print("warnings is installing")
    pip.main(["install", "warnings"])
    import warnings
    print("sys has been installed!")

try:
    import sys
    print("sys has been installed!") 
except ModuleNotFoundError:
    print("sys is installing")
    pip.main(["install", "sys"])
    import sys
    print("sys has been installed!") 

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.mode.chained_assignment = None

start_date = sys.argv[1]
end_date = sys.argv[2]
pfe_sourse_path = sys.argv[3]
simple_sourse_path = sys.argv[4]
work_state_path = sys.argv[5]

interfaces = pd.read_csv(sys.argv[6])
datacenters = pd.read_csv(sys.argv[7])
customers = pd.read_csv(sys.argv[8],encoding='latin-1')

output_path = sys.argv[9]


pod_version = sys.argv[10].lower() == "true"
perc_version = sys.argv[11].lower() == "true"
perc_for_time = int(sys.argv[12]) if perc_version else 75

columns = ["pod", "date", "instance", "action", "status", "count", "sumRespTime", "sumRespSize"]

def get_work_state(work_state_path, gr_list):
    return pd.read_csv(work_state_path,encoding='latin-1',header=None, 
                         names = ["dc", "date", "Instance Name", "Logins", "Started", 
                                  "Completed", "Created", "Canceled", "Day_State", "Biweek_State"]).groupby(gr_list[:-1])[["Logins", "Started", 
                                  "Completed", "Created", "Canceled", "Day_State", "Biweek_State"]].any().reset_index()


def get_sourse_data(sourse_path):
    pfe = pd.read_csv(sourse_path, header=None, names=columns, parse_dates=["date"], encoding='latin-1')
    pfe["day"] = pfe['date'].dt.floor('d')

    # New pers ver:
    pfe = pfe.set_index("date")
    pfe = pfe.groupby([pd.Grouper(freq='5min'), "pod", "instance", "action", "status", "day"]).sum().reset_index()
    #
    pfe.rename(columns={"pod": "dc", "instance": "Instance Name", "action": "Request Type", "status": "Status", 
            "count": "Count", "sumRespTime": "Total Time", "sumRespSize": "Total Size"}, inplace=True)
    print("Sourse data has been downloaded.")
    return pfe


def get_active_RT_list(pfe, incident_day,base_gr_list):
    # pfe_hist = pfe[pfe["date"] < incident_day]
    pfe_hist = pfe[pfe["period"] != "00"]
    pfe_hist_ins_act_size = pfe_hist.groupby(base_gr_list)["date"].count().sort_values(ascending=False).reset_index(drop=False)
    pfe_hist_ins_act_size_max = pfe_hist_ins_act_size.groupby(base_gr_list[:-1])["date"].max().sort_values(ascending=False).reset_index(drop=False)
    
    pfe_hist_ins_act_size_max_pers = pd.merge(pfe_hist_ins_act_size, pfe_hist_ins_act_size_max, 
                                              on=base_gr_list[:-1],how='left',suffixes=('', '_max'))
    pfe_hist_ins_act_size_max_pers["pers"] = pfe_hist_ins_act_size_max_pers["date"] / pfe_hist_ins_act_size_max_pers["date_max"]
    
    active_RT = pfe_hist_ins_act_size_max_pers[(pfe_hist_ins_act_size_max_pers["pers"] >= 0.3) & 
     (pfe_hist_ins_act_size_max_pers["date"] >= pfe_hist_ins_act_size_max_pers["date"].max() * 0.1)].sort_values("date", ascending=False)
    
    if len(base_gr_list) == 2:  # without dc
        active_RT["inst_RT"] = active_RT["Instance Name"] + active_RT["Request Type"]
    else:
        active_RT["inst_RT"] = active_RT["dc"] + active_RT["Instance Name"] + active_RT["Request Type"]

    active_RT_list = active_RT["inst_RT"].drop_duplicates().tolist()
    print("Active Request Types list has been calculated.")
    return active_RT_list

def match_fast(df_main, df_source, index_col, data_col):
    join_df = pd.merge(df_main, df_source, on = [index_col], how= "left")
    print(all(df_main.loc[:,index_col].values == join_df.loc[:,index_col].values))
    return join_df.loc[:,data_col].values


def percentile(s):
    return np.percentile(s,perc_for_time)

def get_number_history_point(RD_df, gr_list):
    RD_count_hist = pd.DataFrame(RD_df[RD_df.period != "00"].groupby(gr_list)["period"].apply(lambda p: len(set(p))))
    RD_count_hist.reset_index(inplace=True)
    RD_count_hist.rename(columns={"period": "Count_point"}, inplace=True)
    print("Numbers of history point have been calculated.")
    return RD_count_hist

def data_preproc(pfe, base_gr_list, is_pfe=True):
    days = pfe["day"].sort_values().unique()
    incident_day = days[-1]
    date_shift = {0: "00", 1: "00", 7: "07", 8: "07", 14: "14", 15: "14", 21: "21", 22: "21",
                   28: "28", 29: "28", 35: "35", 36: "35", 42: "42", 43: "42", 
                   49: "49", 50: "49", 56: "56", 57: "56"}

    # print(incident_day)
    # print(pfe["day"].unique())
    pfe["period"] = pfe["day"].apply(lambda d: date_shift[(incident_day - d).days])
    # pfe = pfe.groupby(base_gr_list + ["date", "Status", "period", "day"]).sum().reset_index()
    pfe = pfe.groupby(base_gr_list + ["date", "Status", "period"]).sum().reset_index()

    if is_pfe:
        active_RT_list = get_active_RT_list(pfe, incident_day, base_gr_list)
        if len(base_gr_list) == 2: # without dc
            mask = (pfe["Instance Name"] + pfe["Request Type"]).isin(active_RT_list).tolist()
        else: #with dc
            mask = (pfe["dc"] + pfe["Instance Name"] + pfe["Request Type"]).isin(active_RT_list).tolist()
        
        RD = pfe.loc[mask,:]
    else:
        RD = pfe

    RD_df = RD.groupby(base_gr_list + ["Status", "period"]).agg({"Count": "sum", "Total Time": "sum", "Total Size":"sum"}).reset_index()
    RD_count_hist = get_number_history_point(RD_df, base_gr_list)
    RD_calc_merged = base_stat(RD, RD_df, RD_count_hist, base_gr_list)
    print("Data preprocessing has been completed.")
    return RD_calc_merged, RD, RD_count_hist

def time_stat(RD, RD_count_hist, base_gr_list):

    # RD_time = RD.drop(["Total Size", "day"], axis = 1)
    RD_time = RD.drop(["Total Size"], axis = 1)

    RD_time = RD_time[RD_time['Instance Name'].notnull()]

    RD_time = RD_time.merge(RD_count_hist, left_on = base_gr_list, 
                            right_on= base_gr_list, how = "left").fillna(0)

    RD_time['is_error'] = RD_time['Status'].apply(lambda x: "error" if x >= 400 else "normal")

    # RD_time = RD_time.drop(["Status", "dc"], axis = 1)

    def inner_time_stat(RD_time, base_gr_list):

        RD_time = RD_time.groupby(base_gr_list + ["date",  "period", "Count_point"]).sum().reset_index()

        RD_tot_time = RD_time.groupby(base_gr_list + ["date",  "period", "Count_point"]).sum().reset_index()

        RD_tot_time["Avg_Time"] = np.where(RD_tot_time["Count"] != 0, RD_tot_time["Total Time"] / RD_tot_time["Count"], 0)

        RD_tot_time_hist =  RD_tot_time[RD_tot_time.period != "00"]

        RD_tot_time_targ = RD_tot_time[RD_tot_time.period == "00"]

        RD_tot_time_hist_gr = RD_tot_time_hist.groupby(base_gr_list + ["Count_point"]).agg({"Avg_Time" : percentile}).reset_index()
        RD_tot_time_targ_gr = RD_tot_time_targ.groupby(base_gr_list + ["Count_point"]).agg({"Avg_Time" : percentile}).reset_index()

        RD_tot_time_gr = RD_tot_time_hist_gr.merge(RD_tot_time_targ_gr, left_on = base_gr_list + ["Count_point"], 
                                right_on= base_gr_list + ["Count_point"], suffixes=('_hist', '_targ'), how="outer").fillna(0)

        RD_tot_time_hist_per = RD_tot_time_hist.groupby(base_gr_list + ["period", "Count_point"]).agg({"Avg_Time" : percentile}).reset_index()
        RD_tot_time_hist_std = RD_tot_time_hist_per.groupby(base_gr_list + ["Count_point"]).agg({"Avg_Time" : np.std}).fillna(0).reset_index()
        RD_tot_time_gr = RD_tot_time_gr.merge(RD_tot_time_hist_std, left_on = base_gr_list + ["Count_point"], 
                                right_on= base_gr_list + ["Count_point"], suffixes=('_hist', '_targ'), how="outer").fillna(0).rename(columns={"Avg_Time":"Avg_Time_std"})


        RD_tot_time_gr["Min Perc Total Time"] = np.where((RD_tot_time_gr["Count_point"] <  RD_count_hist["Count_point"].max()) | 
                                                        (RD_tot_time_gr["Avg_Time_hist"] == 0), 0
                                                        ,np.where(RD_tot_time_gr["Avg_Time_hist"] - 2 * RD_tot_time_gr["Avg_Time_std"]> 0.000001,
                                                                    RD_tot_time_gr["Avg_Time_hist"] - 2 * RD_tot_time_gr["Avg_Time_std"], 0.000001))
        RD_tot_time_gr["Max Perc Total Time"] = RD_tot_time_gr["Avg_Time_hist"] + 2 * RD_tot_time_gr["Avg_Time_std"]

        return RD_tot_time_gr

    RD_tot_time_gr = inner_time_stat(RD_time, base_gr_list)
    RD_suc_time_gr = inner_time_stat(RD_time[RD_time["is_error"] == "normal"], base_gr_list)
    RD_time_stat = RD_tot_time_gr.merge(RD_suc_time_gr, left_on = base_gr_list, 
                            right_on= base_gr_list, how = "left", suffixes=("_tot", "_suc")).drop(["Count_point_tot", "Avg_Time_std_tot",
                                                                                                    "Count_point_suc", "Avg_Time_std_suc"], axis=1).fillna(0)
    RD_time_stat.rename(columns={"Avg_Time_hist_tot": "Perc Total Time history", "Avg_Time_targ_tot" : "Perc Total Time target",
                                "Min Perc Total Time_tot": "Min Perc Total Time", "Max Perc Total Time_tot": "Max Perc Total Time",
                                "Avg_Time_hist_suc" : "Perc Success Time history", "Avg_Time_targ_suc" : "Perc Success Time target",
                                "Min Perc Total Time_suc": "Min Perc Success Time", "Max Perc Total Time_suc": "Max Perc Success Time"}, inplace=True)

    print("Perc stat has been calculated.")
    return RD_time_stat


def base_stat(RD, RD_df, RD_count_hist, gr_list):
    RD_df = RD_df[RD_df['Instance Name'].notnull()]

    RD_df['is_error'] = RD_df['Status'].apply(lambda x: "error" if x >= 400 else "normal")

    RD_df['period'] = RD_df['period'].map({"00":"target", "07": "history", "14": "history", 
    "21": "history", "28":"history", "35": "history", "42": "history","49": "history", "56": "history"})
    RD_gr = RD_df.drop(['Status'], axis = 1).groupby(gr_list + ['period','is_error']).sum()
    RD_gr.reset_index(inplace = True)
    RD_gr = RD_gr.merge(RD_count_hist, left_on = gr_list, 
                        right_on= gr_list,how="left")
    RD_gr['Count'] = np.where(RD_gr['period'] == 'history', RD_gr['Count'] / RD_gr["Count_point"], RD_gr['Count'])
    RD_gr['Total Time'] = np.where(RD_gr['period'] == 'history', RD_gr['Total Time'] / RD_gr["Count_point"], RD_gr['Total Time'])
    RD_gr['Total Size'] = np.where(RD_gr['period'] == 'history', RD_gr['Total Size'] / RD_gr["Count_point"], RD_gr['Total Size'])
    RD_pivot = RD_gr.pivot_table(index=gr_list + ['period', 'Count_point'], 
                                  columns=['is_error'], 
                                  values=['Count', 'Total Time', 'Total Size'], fill_value = 0)

    RD_pivot.columns = [i+" "+j for i, j in RD_pivot.columns]
    RD_pivot.reset_index(inplace = True)
    RD_pivot['Total'] = RD_pivot['Count error'] + RD_pivot['Count normal']
    RD_pivot['Error Rate'] = np.where(RD_pivot['Total'] > 0, RD_pivot['Count error'] / RD_pivot['Total'], 0) 
    RD_pivot['Avg Total Time'] = np.where(RD_pivot['Total'] > 0,(RD_pivot['Total Time error'] + RD_pivot['Total Time normal']) / RD_pivot['Total'],0) 
    RD_pivot['Avg Success Time'] = np.where(RD_pivot['Count normal'] == 0, 0 ,RD_pivot['Total Time normal']  / RD_pivot['Count normal'])
    RD_pivot['Avg Total Size'] = np.where(RD_pivot['Total'] > 0,(RD_pivot['Total Size error'] + RD_pivot['Total Size normal']) / RD_pivot['Total'],0)

    RD_calc = RD_pivot.drop(['Count error', 'Count normal', 'Total Time error', 'Total Time normal','Total Size error','Total Size normal'], axis = 1)



    RD_calc_pivot = RD_calc.pivot_table(index=gr_list + ['Count_point'], 
                                    columns=['period'], 
                                    values=['Total', 'Error Rate', 'Avg Total Time', 'Avg Success Time','Avg Total Size'], fill_value = 0)

    RD_calc_pivot.columns = [i+" "+j for i, j in RD_calc_pivot.columns]
    RD_calc_pivot.reset_index(inplace = True)

    RD_hist = RD[RD["period"] != '00']
    RD_hist["Status"] = RD_hist["Status"].apply(lambda x: "error" if x >= 400 else "normal")

    RD_hist_error_pivot = RD_hist.pivot_table(index=gr_list + ['period', "date"], 
                                  columns=['Status'], aggfunc='sum',
                                  values=['Count', 'Total Time', 'Total Size'], fill_value = 0)
    RD_hist_error_pivot.columns = [i+" "+j for i, j in RD_hist_error_pivot.columns]
    RD_hist_error_pivot.reset_index(inplace = True)
    RD_hist_period_sum = RD_hist_error_pivot.groupby(gr_list + ["period"])[["Count error","Count normal","Total Size error",
                                                                                   "Total Size normal","Total Time error","Total Time normal"]].sum().reset_index()
    RD_hist_period_sum["Total"] = RD_hist_period_sum["Count error"] + RD_hist_period_sum["Count normal"]
    RD_hist_period_sum["Error Rate"] = np.where(RD_hist_period_sum["Total"] > 0, 
                                                    RD_hist_period_sum["Count error"] / RD_hist_period_sum["Total"], 0)
    RD_hist_period_sum["Avg Total Time"] = np.where(RD_hist_period_sum["Total"] > 0,
                                                        (RD_hist_period_sum["Total Time error"] + RD_hist_period_sum["Total Time normal"]) / RD_hist_period_sum["Total"], 0)
    RD_hist_period_sum["Avg Success Time"] = np.where(RD_hist_period_sum["Count normal"] > 0,
                                                        (RD_hist_period_sum["Total Time normal"]) / RD_hist_period_sum["Count normal"], 0)
    RD_hist_period_sum["Avg Total Size"] = np.where(RD_hist_period_sum["Total"] > 0,
                                                        (RD_hist_period_sum["Total Size error"] + RD_hist_period_sum["Total Size normal"]) / RD_hist_period_sum["Total"], 0)
    RD_hist_period_gr = RD_hist_period_sum.groupby(gr_list).agg({"Total": ["std", "mean"],
                                                                "Error Rate":["std", "mean"],
                                                                "Avg Total Time":["std", "mean"],
                                                                "Avg Success Time":["std","mean"],
                                                                "Avg Total Size": ["std","mean"]}).reset_index()
    RD_hist_period_gr.columns = [" ".join([i,j]) for i, j in RD_hist_period_gr.columns]
    RD_hist_period_gr.columns = [c.strip() for c in RD_hist_period_gr.columns]
    RD_hist_period_gr.reset_index(inplace = True, drop=True)
    RD_hist_period_gr.fillna(0, inplace=True)
    RD_hist_period_gr = pd.merge(RD_calc_pivot, RD_hist_period_gr, on = gr_list, 
                             how = "outer")[gr_list+['Count_point',
                                             'Total history', 'Total mean', 'Total std','Avg Success Time history', 'Avg Success Time std',
                                             'Avg Total Size history', 'Avg Total Size std', 'Avg Total Time history','Avg Total Time std',
                                             'Error Rate history', 'Error Rate std']]

    RD_hist_period_gr.fillna(0,inplace=True)
    print( "Check for Control Sum: ",  all(RD_hist_period_gr["Total history"] - RD_hist_period_gr["Total mean"] < 0.000001))

    RD_hist_period_gr.drop("Total mean", axis = 1, inplace=True)
    RD_hist_period_gr.rename(columns={"Total history": "Total mean", "Avg Success Time history": "Avg Success Time mean",
                                                       "Avg Total Size history": "Avg Total Size mean", "Avg Total Time history": "Avg Total Time mean",
                                                       "Error Rate history": "Error Rate mean"}, inplace=True)
    max_point = RD_hist_period_gr["Count_point"].max()

    RD_hist_period_gr["Min Total"] = np.where(RD_hist_period_gr["Total mean"] > 0, 
                                            np.where((RD_hist_period_gr["Total mean"] - 2 * RD_hist_period_gr["Total std"]> 0) &
                                                    (RD_hist_period_gr["Count_point"] == max_point), 
                                                    RD_hist_period_gr["Total mean"] - 2 * RD_hist_period_gr["Total std"],0),0)
    RD_hist_period_gr["Max Total"] = RD_hist_period_gr["Total mean"] + 2 * RD_hist_period_gr["Total std"]
    RD_hist_period_gr["Min Error Rate"] = np.where(RD_hist_period_gr["Error Rate mean"] > 0, 
                                            np.where((RD_hist_period_gr["Error Rate mean"] - 2 * RD_hist_period_gr["Error Rate std"]> 0.000001) &
                                                    (RD_hist_period_gr["Count_point"] == max_point), 
                                                    RD_hist_period_gr["Error Rate mean"] - 2 * RD_hist_period_gr["Error Rate std"],0.000001),0)
    RD_hist_period_gr["Max Error Rate"] = np.where(RD_hist_period_gr["Error Rate mean"] + 2 * RD_hist_period_gr["Error Rate std"] > 0.999999, 0.999999, 
                                                RD_hist_period_gr["Error Rate mean"] + 2 * RD_hist_period_gr["Error Rate std"]) 

    RD_hist_period_gr["Min Avg Total Time"] = np.where(RD_hist_period_gr["Avg Total Time mean"] > 0, 
                                          np.where((RD_hist_period_gr["Avg Total Time mean"] - 2 * RD_hist_period_gr["Avg Total Time std"]> 0.000001) &
                                                   (RD_hist_period_gr["Count_point"] == max_point), 
                                                   RD_hist_period_gr["Avg Total Time mean"] - 2 * RD_hist_period_gr["Avg Total Time std"],0.000001),0)
    RD_hist_period_gr["Max Avg Total Time"] = RD_hist_period_gr["Avg Total Time mean"] + 2 * RD_hist_period_gr["Avg Total Time std"]

    RD_hist_period_gr["Min Avg Success Time"] = np.where(RD_hist_period_gr["Avg Success Time mean"] > 0, 
                                            np.where((RD_hist_period_gr["Avg Success Time mean"] - 2 * RD_hist_period_gr["Avg Success Time std"]> 0.000001) &
                                                    (RD_hist_period_gr["Count_point"] == max_point), 
                                                    RD_hist_period_gr["Avg Success Time mean"] - 2 * RD_hist_period_gr["Avg Success Time std"],0.000001),0)
    RD_hist_period_gr["Max Avg Success Time"] = RD_hist_period_gr["Avg Success Time mean"] + 2 * RD_hist_period_gr["Avg Success Time std"]

    RD_hist_period_gr["Min Avg Total Size"] = np.where(RD_hist_period_gr["Avg Total Size mean"] > 0, 
                                            np.where((RD_hist_period_gr["Avg Total Size mean"] - 2 * RD_hist_period_gr["Avg Total Size std"]> 0) &
                                                    (RD_hist_period_gr["Count_point"] == max_point), 
                                                    RD_hist_period_gr["Avg Total Size mean"] - 2 * RD_hist_period_gr["Avg Total Size std"],0),0)
    RD_hist_period_gr["Max Avg Total Size"] = RD_hist_period_gr["Avg Total Size mean"] + 2 * RD_hist_period_gr["Avg Total Size std"]

    RD_hist_period_gr.drop(["Count_point"], axis=1, inplace=True)

    RD_calc_merged = pd.merge(RD_calc_pivot, RD_hist_period_gr, on = gr_list, how="left")
    RD_calc_merged.fillna(0, inplace=True)
    print("Base Stat has been calculated.")
    return RD_calc_merged

def base_time_merge(pfe_preproc, pfe_time_stat, base_gr_list):
    RD_calc_merged = pfe_preproc.merge(pfe_time_stat, on = base_gr_list)
    RD_calc_merged.drop(['Count_point',
       'Avg Success Time history', 'Avg Success Time target',       
       'Avg Total Time history', 'Avg Total Time target','Total mean',
       'Total std', 'Avg Success Time mean', 'Avg Success Time std',
       'Avg Total Size mean', 'Avg Total Size std', 'Avg Total Time mean',
       'Avg Total Time std', 'Error Rate mean', 'Error Rate std','Min Avg Total Time',
       'Max Avg Total Time', 'Min Avg Success Time', 'Max Avg Success Time',
       ], axis=1, inplace=True)


    # RD_calc_merged.rename(columns={'Avg Total Time history':'Total Time history', 'Avg Total Time target': 'Total Time target',
    #                                'Min Avg Total Time':'Min Total Time', 'Max Avg Total Time':'Max Total Time',
    #                                'Avg Success Time history':'Success Time history', 'Avg Success Time target':'Success Time target',
    #                                'Min Avg Success Time':'Min Success Time', 'Max Avg Success Time':'Max Success Time'}, inplace=True)

    RD_calc_merged.rename(columns={'Perc Total Time history':'Total Time history', 'Perc Total Time target': 'Total Time target',
                                'Min Perc Total Time':'Min Total Time', 'Max Perc Total Time':'Max Total Time',
                                'Perc Success Time history':'Success Time history', 'Perc Success Time target':'Success Time target',
                                'Min Perc Success Time':'Min Success Time', 'Max Perc Success Time':'Max Success Time'}, inplace=True)
    
    return RD_calc_merged

def losses_calculation(RD_calc_merged):

    RD_calc_merged['DELTA Total %'] = np.where(RD_calc_merged['Total history'] > 0,
                                            np.where(RD_calc_merged['Total target'] > RD_calc_merged['Max Total'],
                                                    (RD_calc_merged['Total target'] -
                                                    RD_calc_merged['Max Total']) / RD_calc_merged['Max Total'],
                                                    np.where(RD_calc_merged['Total target'] < RD_calc_merged['Min Total'],
                                                            (RD_calc_merged['Total target'] - RD_calc_merged['Min Total']) / RD_calc_merged['Min Total'], 0)), 0)*100

    RD_calc_merged['DELTA Total Time %'] = np.where(RD_calc_merged['Total Time history'] > 0,
                                                    np.where(RD_calc_merged['Total Time target'] > RD_calc_merged['Max Total Time'],
                                                    (RD_calc_merged['Total Time target'] -
                                                    RD_calc_merged['Max Total Time']) / RD_calc_merged['Max Total Time'],
                                                    np.where(RD_calc_merged['Total Time target'] < RD_calc_merged['Min Total Time'],
                                                            (RD_calc_merged['Total Time target'] - RD_calc_merged['Min Total Time']) / RD_calc_merged['Min Total Time'], 0)), 0)*100

    RD_calc_merged['DELTA Success Time %'] = np.where(RD_calc_merged['Success Time history'] > 0,
                                                        np.where(RD_calc_merged['Success Time target'] > RD_calc_merged['Max Success Time'],
                                                                (RD_calc_merged['Success Time target'] -
                                                                RD_calc_merged['Max Success Time']) / RD_calc_merged['Max Success Time'],
                                                                np.where(RD_calc_merged['Success Time target'] < RD_calc_merged['Min Success Time'],
                                                                        (RD_calc_merged['Success Time target'] - RD_calc_merged['Min Success Time']) / RD_calc_merged['Min Success Time'], 0)), 0)*100

    RD_calc_merged['DELTA Error Rate %'] = np.where(RD_calc_merged['Error Rate history'] > 0,
                                                np.where(RD_calc_merged['Error Rate target'] > RD_calc_merged['Max Error Rate'],
                                                        (RD_calc_merged['Error Rate target'] -
                                                            RD_calc_merged['Max Error Rate']) / RD_calc_merged['Max Error Rate'],
                                                        np.where(RD_calc_merged['Error Rate target'] < RD_calc_merged['Min Error Rate'],
                                                                    (RD_calc_merged['Error Rate target'] - RD_calc_merged['Min Error Rate']) / RD_calc_merged['Min Error Rate'], 0)), 
                                                                    np.where((RD_calc_merged['Total history'] > 0) & (RD_calc_merged['Error Rate target'] > 0),1,0))*100

    RD_calc_merged['DELTA Avg Total Size %'] = np.where(RD_calc_merged['Avg Total Size history'] > 0,
                                                    np.where(RD_calc_merged['Avg Total Size target'] > RD_calc_merged['Max Avg Total Size'],
                                                    (RD_calc_merged['Avg Total Size target'] -
                                                    RD_calc_merged['Max Avg Total Size']) / RD_calc_merged['Max Avg Total Size'],
                                                    np.where(RD_calc_merged['Avg Total Size target'] < RD_calc_merged['Min Avg Total Size'],
                                                            (RD_calc_merged['Avg Total Size target'] - RD_calc_merged['Min Avg Total Size']) / RD_calc_merged['Min Avg Total Size'], 0)), 0)*100
    print("Losses calculated!")

    return RD_calc_merged

def post_proc(pfe_losses, base_gr_list):
    pfe_proc = pfe_losses[base_gr_list + ['Total history', 'Total target', 'Min Total', 'Max Total', 
                 'DELTA Total %',
                'Total Time history', 'Total Time target', 'Min Total Time','Max Total Time',
                'DELTA Total Time %', 
                'Success Time history', 'Success Time target', 'Min Success Time','Max Success Time',
                'DELTA Success Time %', 
                'Error Rate history', 'Error Rate target', 'Min Error Rate', 'Max Error Rate',
                'DELTA Error Rate %',
                'Avg Total Size history','Avg Total Size target','Min Avg Total Size','Max Avg Total Size',
                'DELTA Avg Total Size %']]
    print("Post processing has been completed.")
    
    return pfe_proc

def get_anomalies(RD_simple, gr_list):
    filter_total = 0.1
    filter_time = 1
    filter_error = 0.01
    RD_simple_total_gr_max = RD_simple.groupby(gr_list[:-1])["Total history"].max().reset_index().rename(columns={"Total history":"Total max"})
    RD_simple = pd.merge(RD_simple, RD_simple_total_gr_max, on = gr_list[:-1], how="left")
    RD_simple["Total_in_stat"] = (RD_simple["Total history"] >= RD_simple["Total max"] * filter_total)

    RD_simple["anomaly"] = np.round(RD_simple[["DELTA Total %", "DELTA Total Time %",
                                   "DELTA Success Time %", "DELTA Avg Total Size %",
                                   "DELTA Error Rate %" ]].apply(lambda row: int(np.any(np.array([(i <= -50) or (i >= 100) for i in row[:-1]] + [row[-1]> 0]))), axis=1),2)
    RD_simple["anomaly_adj"] = RD_simple[["DELTA Total %", 
                                        "Total Time target", "DELTA Total Time %",
                                            "Success Time target","DELTA Success Time %", 
                                            "DELTA Avg Total Size %",
                                            "Error Rate target" ,"DELTA Error Rate %" ]].apply(lambda row: 
                                                                    int(np.any(np.array([((row[0] <= -50) or (row[0] >= 100)),
                                                                                            ((row[1] >= filter_time) & ((row[2] <= -50) or (row[2] >= 100))),
                                                                                            ((row[3] >= filter_time) & ((row[4] <= -50) or (row[4] >= 100))),
                                                                                            ((row[5] <= -50) or (row[5] >= 100)),
                                                                                            (((row[6] >= filter_error)  & (row[7] >= 100)) | (row[6] == 1))]))), axis=1)
    RD_simple["anomaly_adj"] = np.round(np.where(RD_simple["Total_in_stat"], RD_simple["anomaly_adj"], np.NAN),2)

    RD_simple = RD_simple[gr_list + ["anomaly", "anomaly_adj"] + ['Total history', 'Total target', 'Min Total', 'Max Total', 
                 'DELTA Total %',
                'Total Time history', 'Total Time target', 'Min Total Time','Max Total Time',
                'DELTA Total Time %', 
                'Success Time history', 'Success Time target', 'Min Success Time','Max Success Time',
                'DELTA Success Time %', 
                'Error Rate history', 'Error Rate target', 'Min Error Rate', 'Max Error Rate',
                'DELTA Error Rate %',
                'Avg Total Size history','Avg Total Size target','Min Avg Total Size','Max Avg Total Size',
                'DELTA Avg Total Size %', "Total max", "Total_in_stat"]]

    print("Anomalies have been calculated.")
    return RD_simple

def merge_and_write(RD_simple, RD_pfe,customers, work_state,output_path, gr_list):
    # RD_simple_gr = RD_simple.groupby(['Instance Name', 'dc_origin'])[["anomaly", "anomaly_adj"]].mean().reset_index()
    # RD_pfe_gr = RD_pfe.groupby(['Instance Name', 'dc_origin'])[["anomaly", "anomaly_adj"]].mean().reset_index()
    RD_simple_gr = RD_simple.groupby(gr_list[:-1])[["anomaly_adj"]].mean().reset_index()
    RD_pfe_gr = RD_pfe.groupby(gr_list[:-1])[["anomaly_adj"]].mean().reset_index()
    RD_join = pd.merge(RD_simple_gr, RD_pfe_gr, on=gr_list[:-1],how='outer',suffixes=("_Simple", "_PFE"))

    RD_join = pd.merge(RD_join, work_state[gr_list[:-1] + ["Day_State", "Biweek_State"]], how="left")

    RD_join["dc_origin"] = match_fast(RD_join, customers, "Instance Name", "Datacenter Label")    


    RD_join = RD_join[gr_list[:-1] + ["dc_origin", "Day_State", "Biweek_State", "anomaly_adj_Simple", "anomaly_adj_PFE"]]
    writer = pd.ExcelWriter(output_path, engine = 'xlsxwriter')
    RD_join.to_excel(writer, sheet_name='Statistic', index=False)
    RD_simple.to_excel(writer, sheet_name='Simple', index=False)
    RD_pfe.to_excel(writer, sheet_name='PFE', index=False)

    # writer.save()
    writer.close()
    print()
    print("All calculation have been completed. File has been created!")

print()
print("Start pfe Calculating...")
print()

base_gr_list = ["Instance Name","Request Type"]
pod_gr_list = ["dc","Instance Name","Request Type"]

gr_list = pod_gr_list if pod_version else base_gr_list

print(pod_version, perc_version, perc_for_time)

print(gr_list)

work_state = get_work_state(work_state_path, gr_list)

pfe = get_sourse_data(pfe_sourse_path)
pfe_preproc, pfe_RD, pfe_RD_count_hist = data_preproc(pfe, gr_list)
pfe_time_stat = time_stat(pfe_RD, pfe_RD_count_hist, gr_list)
pfe_merged = base_time_merge(pfe_preproc, pfe_time_stat, gr_list)
pfe_losses = losses_calculation(pfe_merged)
pfe_fin = post_proc(pfe_losses,gr_list)
pfe_anom = get_anomalies(pfe_fin, gr_list)

print()
print("Start simple Calculating...")
print()

simple = get_sourse_data(simple_sourse_path)
simple_preproc, simple_RD, simple_RD_count_hist = data_preproc(simple, gr_list, is_pfe=False)
simple_time_stat = time_stat(simple_RD, simple_RD_count_hist, gr_list)
simple_merged = base_time_merge(simple_preproc, simple_time_stat, gr_list)
simple_losses = losses_calculation(simple_merged)
simple_fin = post_proc(simple_losses,gr_list)
simple_anom = get_anomalies(simple_fin,gr_list)

merge_and_write(simple_anom, pfe_anom,customers,work_state, output_path, gr_list)
