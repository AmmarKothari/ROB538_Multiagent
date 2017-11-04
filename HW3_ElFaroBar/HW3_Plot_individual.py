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
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=100,k=5,b=5.pickle']

file_names_wanted = [
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=30,k=7,b=4.pickle',
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=40,k=5,b=5.pickle',
'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=100,k=5,b=5.pickle']

nights_distr_stat_mean = list(range(len(file_names_wanted)))
nights_distr_stat_std = list(nights_distr_stat_mean)

bar_colors = ['blue','orange','green','yellow','burlywood']*2
labels = ['G', 'L', 'DL', 'DG_NR','DG_R']*2
markings = ['ro:', 'bo:', 'ko:','go:','yo:']*2
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
if __name__ == '__main__':
    for i in range(len(file_names_wanted)):
        with open(file_names_wanted[i], 'rb') as f:
            [nights_distr_stat, reward_stat] = pickle.load(f)
        nights_distr_stat_mean[i] = np.mean(np.array(nights_distr_stat), axis = 0)
        nights_distr_stat_std[i] = np.std(np.array(nights_distr_stat), axis = 0)


        
        ind = np.arange(len(nights_distr_stat[i]))
        ind_width = ind
        ind_ticks = ind + 0.5
        for i2 in range(len(nights_distr_stat[i])):
        ##    pdb.set_trace()
        ##    plt.figure(i//plots_per_graph*2 + 1)
        ##    subplot_loc = subplot_loc_base + i
        ##        plt.bar(ind_width, nights_distr_stat_mean[i],
        ##                width = width, yerr = nights_distr_stat_std[i],
        ##                alpha=opacity, facecolor=bar_colors[i],
        ##                label = labels[i])
        ##        pdb.set_trace()
            plt.bar(ind_width, np.array(nights_distr_stat)[i2],
                    width = width,
                    alpha=opacity, facecolor=bar_colors[i2],
                    label = labels[i2])
            plt.axhline(4, color='b', linestyle='dashed', linewidth=2)
            plt.xticks(ind_ticks, ('M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'))
            plt.title(' Reward')
            plt.xlabel('Day of Week')
            plt.ylabel('Agent Attendance')
            plt.legend()
            if i == plots_per_graph-1:
                plt.savefig(save_names[0], bbox_inches='tight')





            plt.figure(i//plots_per_graph*2 + 2)
            plt.title(file_names_wanted[i])
            plt.plot(reward_stat[i2], markings[i2])
            plt.xlabel('Iterations')
            plt.ylabel('Reward')

            
            if i >= 0 :
                plt.show()
            
    ##        pdb.set_trace()


