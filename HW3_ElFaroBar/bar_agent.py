import numpy as np
import random as random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb as pdb
import pickle as pickle
import shelve as shelve
import time as time



######NEXT STEP: Figure out how to graph all the data easily and cleanly!!!!#########

def choose_random(K):
    rand_arr = np.random.rand(K.shape[0], K.shape[1])
    night_arr = np.argmax(rand_arr, 1)
##    print(night_arr)
    return night_arr

def choose_K(K):
    night_arr = np.argmax(K, 1)
##    print(night_arr)
    return night_arr

def choose_K_epsilon_greedy(K, epsilon,nights_in_week):
    epsilon_arr = np.random.rand(K.shape[0])
    x = [indx for i1,indx in zip(epsilon_arr,np.arange(0,K.shape[0])) if i1 < epsilon ]
    random_choices = np.random.randint(0,nights_in_week, len(x))
##    print(x)
    night_arr = np.argmax(K, 1)
    #replace e-greedy agents with random choices
    night_arr[x] = random_choices
##    pdb.set_trace()
##    print(night_arr)
    return night_arr

def attendance_reward_one_night(z):
    #z i s the attendance on a given night
    r = np.multiply(z,np.exp(-z/attendance_optimal))
    return r

def attendance_all_nights_arr(nights_distribution):
    #reward that comes from each night
    r_each_night = [attendance_reward_one_night(i) for i in nights_distribution]
    return np.array(r_each_night)

def attendance_all_nights(night_arr_dist):
    r_total = 0
    r_total = sum(attendance_all_nights_arr(night_arr_dist))
    return r_total

def reward_global(nights_arr_week, nights_distribution_week):
    r_total = 0
    r_total = attendance_all_nights(nights_distribution_week)
    r_each_agent = [r_total for i in night_arr_week]
    return r_each_agent

def reward_local_one_night(night_arr_week, nights_distribution_week):
    #each agent gets the reward for that night
    r_each_night = attendance_all_nights_arr(nights_distribution_week)
    r_each_agent = [r_each_night[i] for i in night_arr_week]
    return np.array(r_each_agent)

def reward_diff_one_night_no_replace(nights_chosen, nights_distribution):
    #reward difference if one less person shows up
    #NOT replaced on standard night
    #agents reward is difference between showing up and not but only on one night
    r_diff = np.arange(0,float(nights_in_week))*0
    r_current = attendance_all_nights_arr(nights_distribution)
    r_less_one = attendance_all_nights_arr(nights_distribution-1)
    r_diff_nightly = r_current - r_less_one
    r_each_agent = [r_diff_nightly[i] for i in nights_chosen]
    return r_each_agent

def reward_diff_global_no_replace(nights_chosen, nights_distribution):
    #reward difference if one less person shows up
    #NOT replaced on standard night
    #summed up over entire week
    r_current = attendance_all_nights_arr(nights_distribution)
    r_less_one = attendance_all_nights_arr(nights_distribution-1)

##    r_diff_global = [sum(r_current) - sum(r_current) - r_current[i] + r_less_one[i] for i in list(range(nights_in_week))]
    r_diff_global = r_current - r_less_one #difference in global reward is difference between the days with and without agent
    r_each_agent = [r_diff_global[i] for i in nights_chosen]
##    pdb.set_trace()
    return r_each_agent

def reward_diff_global_replace(nights_chosen, nights_distribution):
    #reward difference if one less person shows up
    #REPLACED on standard night
    r_current = attendance_all_nights_arr(nights_distribution)
    r_less_one = attendance_all_nights_arr(nights_distribution-1)
    r_plus_one = attendance_reward_one_night(nights_distribution[cf_night] + 1)
    r_diff_nightly = r_current - r_less_one
    r_diff_cf = r_current[cf_night] - r_plus_one
    r_diff_global = [(r_current[i] + r_current[cf_night] )- (r_less_one[i] + r_plus_one) for i in list(range(nights_in_week))]
    r_each_agent = [r_diff_global[i] for i in nights_chosen]
##    pdb.set_trace()
    return r_each_agent

def plot_data_test(test_name):
##    filename = 'HW_3_' + test_name + '.pickle'
    filename_g = 'HW_3_Best_K_epsilon_Global_N=30,k=7,b=4.pickle'
    filename_ln = 'HW_3_Best_K_epsilon_Local_Nightly_N=30,k=7,b=4.pickle'
    filename_dln = 'HW_3_Best_K_epsilon_Difference_Local_Nightly_N=30,k=7,b=4.pickle'
    filename_dgnr = 'HW_3_Best_K_epsilon_Difference_Global_No_Replace_N=30,k=7,b=4.pickle'
    filename_dgr = 'HW_3_Best_K_epsilon_Difference_Gloal_Replace_N=30,k=7,b=4.pickle'
    for i_s in range(scenario_count):
        for i in range(len(reward_names)):
            nights_in_week = sc_nights_in_week[i_s] #k
            attendance_optimal = sc_attendance_optimal[i_s] #b
            scenario_name ='N=' + str(agent_number) + ',k=' + str(nights_in_week )+ ',b=' + str(attendance_optimal)
            filename = 'HW_3_Best_K_epsilon_' + reward_names[i] + scenario_name
            with open(filename_dgr, 'rb') as f:
                [nights_distr_stat, reward_stat] = pickle.load(f)
        nights_distr_stat_mean = np.mean(np.array(nights_distr_stat), axis = 0)
        nights_distr_stat_std = np.std(np.array(nights_distr_stat), axis = 0)
        pdb.set_trace()
        ind = np.arange(len(nights_distr_stat_std))
        plt.bar(ind, nights_distr_stat_mean, width = width, yerr = nights_distr_stat_std, alpha=opacity)
        plt.xticks(ind + width/2, ('M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'))
        plt.title(' Reward')
        plt.xlabel('Day of Week')
        plt.ylabel('Agent Attendance')
        plt.savefig('HW_3_' + test_name + '.png', bbox_inches='tight')
        plt.show()
        reward_total_all_2 = np.array(reward_total_all)
        reward_total_average = np.mean(reward_total_all_2.reshape(10,-1), axis=1)
        print(reward_total_average)
        plt.plot(reward_total_average, marker = 'o')
        plt.show()

def attendance_animation(nights_distribution_all):
    plt.ion()
    every = 100
    opacity_animation = 0.2
    for i in list(range(len(nights_distribution_all)/every)):
        plt.bar(ind, nights_distribution_all[i*every], width = width, alpha=opacity_animation)
        plt.show()
        plt.pause(0.1)
    

epsilon_aneeling = 0.99
weeks_test_limit = 100000
scenario_count = 3
sc_agent_number = [30, 40, 100]
sc_nights_in_week = [7, 5, 5] #k
sc_attendance_optimal = [4, 5, 5] #b
learning_rate = 0.01
cf_night = 0
opacity = 0.7
width = 1
stat_runs = 10
choice_type = 2
choice_names = ['Random', 'Best_K', 'Best_K_epsilon']
reward_names = ['Global', 'Local_Nightly', 'Difference_Local_Nightly', 'Difference_Global_No_Replace', 'Difference_Gloal_Replace']
if __name__ == '__main__':
    for reward_type in [4]: #list(range(len(reward_names))):
        nights_distr_stat = list(range(0,stat_runs))
        reward_stat = list(range(0,stat_runs))
        for i_scenario in list(range(scenario_count)):
            agent_number = sc_agent_number[i_scenario]
            nights_in_week = sc_nights_in_week[i_scenario] #k
            attendance_optimal = sc_attendance_optimal[i_scenario] #b
            scenario_name ='N=' + str(agent_number) + ',k=' + str(nights_in_week )+ ',b=' + str(attendance_optimal)
            for i_stat in list(range(stat_runs)):
                epsilon = 0.2
                nights_in_week_bins = np.arange(0,nights_in_week+1,1)
                K_agents = np.ones([agent_number, nights_in_week])*20
                nights_error = np.arange(nights_in_week)*0.0
                nights_distribution_all = list(range(0,weeks_test_limit))
                nights_arr_all = list(range(0,weeks_test_limit))
                reward_total_all = list(range(0,weeks_test_limit))
                reward_individual_all = list(range(0,weeks_test_limit))
                for i in range(0,weeks_test_limit):
                    epsilon = epsilon * epsilon_aneeling
                    #nights chosen for all agents - "Each agent chooses an action"
                    if choice_type == 0:
                        night_arr_week = choose_random(K_agents)
                    elif choice_type == 1:
                        night_arr_week = choose_K(K_agents)
                    elif choice_type == 2:
                        night_arr_week = choose_K_epsilon_greedy(K_agents,epsilon,nights_in_week)
                    nights_arr_all[i] = night_arr_week
                    #distribution of all agents choices - "Those actions lead to a system state"
                ##    print("Nights Picked: %s" %night_arr_week)
                    [nights_distribution_week , edges] = np.histogram(night_arr_week,nights_in_week_bins)    
                ##    print("Nights Distribution: %s" %nights_distribution_week)
                    nights_distribution_all[i] = nights_distribution_week
                    
                    #calculate reward for system - "The system state leads to a system reward"
                    reward_total = attendance_all_nights(nights_distribution_week)
                    reward_total_all[i] = reward_total
                ##    print('Reward for the Week: %s' %reward_total)

                    #each agent receives a reward - "Each agent receives a reward (i.e. agent reward)"
                    if reward_type == 0:
                        reward_individual = reward_global(night_arr_week, nights_distribution_week)# np.ones(agent_number)* reward_total/agent_number
                    elif reward_type == 1:
                        reward_individual = reward_local_one_night(night_arr_week, nights_distribution_week)
                    elif reward_type ==2:
                        reward_individual = reward_diff_one_night_no_replace(night_arr_week, nights_distribution_week)
                    elif reward_type == 3:
                        reward_individual = reward_diff_global_no_replace(night_arr_week, nights_distribution_week)
                    elif reward_type == 4:
                        reward_individual = reward_diff_global_replace(night_arr_week, nights_distribution_week)
                        

                    reward_individual_all[i] = reward_individual

                    #Update K values based on results
                    for (agent_index, night) in zip(np.arange(agent_number), night_arr_week):
                        K_agents[agent_index,night] = K_agents[agent_index,night] + learning_rate*(reward_individual[agent_index] - K_agents[agent_index,night])
                        
                   
                #statistical data update
                nights_distr_stat[i_stat] = np.mean(np.array(nights_distribution_all[-10:-1]), axis = 0)
                reward_total_all_2 = np.array(reward_total_all)
                reward_total_average = np.mean(reward_total_all_2.reshape(100,-1), axis=1)
    ##            pdb.set_trace()
                reward_stat[i_stat] = reward_total_average

            fname = 'HW_3_'
            fname = fname + choice_names[choice_type] +'_'
            fname = fname + reward_names[reward_type] +'_'
            fname = fname + scenario_name
            fname_shelve = fname + '.out'
            fname_pickle = fname + '.pickle'

            with open(fname_pickle, 'wb') as f:
                pickle.dump([nights_distr_stat,
                             reward_stat],
                            f)
            print(fname_pickle)

    ##plot_data_test('b')









