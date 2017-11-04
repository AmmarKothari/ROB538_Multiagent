import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import pdb as pdb





file_names =[
'HW_3_Best_K_epsilon_Global_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Local_Nightly_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Difference_Local_Nightly_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Difference_Global_No_Replace_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Global_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Local_Nightly_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Difference_Local_Nightly_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Difference_Global_No_Replace_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=40,k=5,b=5.pickle']

nights_distr_stat_mean = list(range(len(file_names)))
nights_distr_stat_std = list(nights_distr_stat_mean)

bar_colors = ['blue','orange','green','yellow','burlywood']*2
labels = ['G', 'L', 'DL', 'DG_NR','DG_R']*2
markings = ['ro:', 'bo:', 'ko:','go:','yo:']
save_names = ['Scenario1_hist.png',
              'Scenario1_reward.png',
              'Scenario2_hist.png',
              'Scenario2_reward.png']
width = 1
opacity = 0.7
plot_number = 1
plots_per_graph = 5
rows_per_figure = 5
cols_per_figure = 1
subplot_loc_base = rows_per_figure*100 + cols_per_figure*10 + 1*1 
for i in range(len(file_names)):
    with open(file_names[i], 'rb') as f:
        [nights_distr_stat, reward_stat] = pickle.load(f)
    nights_distr_stat_mean[i] = np.mean(np.array(nights_distr_stat), axis = 0)
    nights_distr_stat_std[i] = np.std(np.array(nights_distr_stat), axis = 0)
    pdb.set_trace()
    plt.figure(i//plots_per_graph*2 + 2)
    plt.plot(reward_stat[i1], markings[i1])
