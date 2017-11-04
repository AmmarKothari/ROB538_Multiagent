import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import pdb as pdb


file_names =[
'HW_4_Best_K_epsilon_Global_N=50,k=5,b=5.pickle',
'HW_4_Best_K_epsilon_Local_Nightly_N=50,k=5,b=5.pickle',
'HW_4_Best_K_epsilon_Difference_Local_Nightly_N=50,k=5,b=5.pickle',
#'HW_4_Best_K_epsilon_Difference_Gloal_Replace_N=50,k=5,b=5.pickle'
]

nights_distr_stat_mean = list(range(len(file_names)))
nights_distr_stat_std = list(nights_distr_stat_mean)
reward_stat_mean = list(nights_distr_stat_mean)
reward_stat_std = list(nights_distr_stat_mean)


bar_colors = ['blue','orange','green','red']*3
##labels = ['G', 'L', 'DL', 'DG_NR','DG_R']*3
labels = ['G', 'L','D','DG_R']*3
marker_color = ['blue','orange','green','red']*3
markings = ['x', 'v', '*','+']*3
linestyle = [':',':',':',':']*3
save_names = ['Scenario1_hist.png',
              'Scenario1_reward.png',
              'Scenario2_hist.png',
              'Scenario2_reward.png',
              'Scenario3_hist.png',
              'Scenario3_reward.png']

scenario_count = 1
width = 1
opacity = 0.7
attendance_optimal = [5]


reward_names = ['Global', 'Local', 'Difference', 'Difference\nReplace']

plots_per_graph = 3
rows_per_figure = 1
cols_per_figure = 3
subplot_loc_base = rows_per_figure*100 + cols_per_figure*10 + 0*1


#Load the information into an array
#Calculate the things we want to plot

for i in range(len(file_names)):
    with open(file_names[i], 'rb') as f:
        [nights_distr_stat, reward_stat] = pickle.load(f)
##    pdb.set_trace()
    
    nights_distr_stat_mean[i] = np.mean(np.sort(np.array(nights_distr_stat)), axis = 0)
    nights_distr_stat_std[i] = np.std(  np.sort(np.array(nights_distr_stat)), axis = 0)

    reward_stat_mean[i] = np.mean(np.array(reward_stat), axis = 0)
    reward_stat_std[i]  = np.std( np.array(reward_stat), axis = 0)

fig_count = 0
bar_plot = 1
for i in range(len(file_names)):
    if i%plots_per_graph == 0:
        fig_count = fig_count + 1
        bar_plot = 1
    plt.figure(fig_count)
    subplot_loc = subplot_loc_base + bar_plot
    bar_plot +=1
    plt.subplot(subplot_loc)
    ind = np.arange(len(nights_distr_stat_std[i]))
    ind_width = ind + width
    ind_ticks = ind + width/2
    
    plt.bar(ind, nights_distr_stat_mean[i],
            width = width, yerr = nights_distr_stat_std[i],
            alpha=opacity, facecolor=bar_colors[i],
            label = labels[i])
####    plt.bar(ind_width, np.array(nights_distr_stat)[i2],
####            width = width,
####            alpha=opacity, facecolor=bar_colors[i],
####            label = labels[i])
    if i//plots_per_graph == 2:
        plt.ylim(0,100)
    elif i//plots_per_graph == 1:
        plt.ylim(0,16)
    elif i //plots_per_graph == 0:
        plt.ylim(0,30)
    plt.axhline(attendance_optimal[i//plots_per_graph], color='b', linestyle='dashed', linewidth=2)
    plt.xticks(ind_ticks, ('1', '2', '3', '4', '5', '6', '7'))
    plt.title(reward_names[i%plots_per_graph])
    plt.xlabel('Day of Week')
    frame1 = plt.gca()
    frame1.axes.get_yaxis().set_visible(False)
    if i%plots_per_graph == 0:
        plt.ylabel('Agent Attendance')
        frame1.axes.get_yaxis().set_visible(True)
    if i%plots_per_graph == plots_per_graph-1:
        plt.savefig('Histograms' + str(i//plots_per_graph), bbox_inches='tight')



##fig_count += 1
plot_ever_n_point = 5
##plot_upto = len(reward_stat_mean[0])
plot_upto = 20
##point_arr = [:20}
for i in list(range(len(file_names))):
    
    if i%plots_per_graph == 0:
##        plt.subplot(311 + i //3)
        fig_count += 1
    plt.figure(fig_count)
    ##plt.plt(reward_stat_mean[i], markings[i], label = labels[i])
    ind = list(range(1,len(reward_stat_mean[i][:plot_upto])+1))
    if i == 1:
        plt.xlabel('1,000s of weeks')
    plt.ylabel('Weekly System Reward')
    plt.errorbar(ind, reward_stat_mean[i][:plot_upto],
                 yerr=reward_stat_std[i][:plot_upto],
                 marker = markings[i],
                 color = marker_color[i],
                 linestyle = linestyle[i],
                 label = labels[i])
    plt.legend(loc = 'best', fancybox=True, framealpha=0.5)
    if i < plots_per_graph:
        Title = 'Reward Learning Rate for N=50, k=5, b=5'
    elif i < 2*plots_per_graph:
        Title = 'Reward Learning Rate for N=40, k=5, b=5'
    elif i < 3*plots_per_graph:
        Title = 'Reward Learning Rate for N=100, k=5, b=5'
    plt.title(Title)
    plt.savefig('Scatter' + str(i//plots_per_graph), bbox_inches='tight')
##    pdb.set_trace()
    

plt.show()
    ##        pdb.set_trace()


