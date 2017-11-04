import numpy as np
import pdb, random
import matplotlib.pyplot as plt


################PARAMETERS#############
scenario = 2
seeding = 1
test_profile = 1
#Scenario 1
if scenario == 1:
    p1_Clods_prob = 0
    p1_Mcm_prob = 1
    fname = 'Scenario1'
###Scenario 2
elif scenario == 2:
    p1_Clods_prob = 0.5
    p1_Mcm_prob = 0.5
    fname = 'Scenario2'
    if test_profile == 1:
        fname = fname + '_optimal'
###Scenario 3
elif scenario == 3:
    p1_Clods_prob = 1/3
    p1_Mcm_prob = 2/3
    fname = 'Scenario3'
##Scenario 4
elif scenario == 4:
    p1_Clods_prob = random.random()
    p1_Mcm_prob = random.random()
    fname = 'Scenario4'
    if seeding == 1:
        p1_Clods_prob = 0.9
        p1_Mcm_prob = 0.1
        fname = fname + '_seeded'
        

n_runs = 10000
epsilon = 0.01
learn_rate = 0.01
payoff = np.zeros([2,2,2])
payoff[0][0] = [1,2]
payoff[1][0] = [0,0]
payoff[0][1] = [0,0]
payoff[1][1] = [2,1]

P2_stat_reward=list()
P2_stat_Q     =list()
P1_stat_reward=list()
P1_stat_Q     =list()

numofpoints = 100
trials = 10


class Player:

    def __init__(self, player_number, ):
        self.player_number = player_number
        self.payoff = payoff
        self.choice = 10
        self.reward = 0
        self.payoff_history =list()
        self.Q_history = list()
        if player_number == 1:
            self.Q = np.sort([p1_Clods_prob, p1_Mcm_prob])
            self.epsilon = 0
        else:
            self.Q = np.random.rand(2)
            if test_profile == 1:
                self.Q = np.array([0,1])
            self.epsilon = epsilon
            if scenario == 4 and seeding:
                #seed values
                self.Q = np.sort([p1_Clods_prob, p1_Mcm_prob])

    def set_reward(self, p1_choice, p2_choice):
        reward = self.payoff[p1_choice][p2_choice][self.player_number-1]
        self.reward = reward
        self.payoff_history.append(reward)

    def update_Q(self):
        self.Q[self.choice] = self.Q[self.choice] + learn_rate * (self.reward - self.Q[self.choice])
        if test_profile == 1:
            self.Q[0] = 1
            self.Q[1] = 0
        #normalize Q
        self.Q = self.Q/sum(self.Q)
        self.Q_history.append(self.Q[0])

    def reward_avg(self):
##        numofpoints = numofpoints
        totalpoints = len(self.Q_history)
        reshaped = np.reshape(self.payoff_history, [-1,numofpoints])
        self.avg_reward = np.mean(reshaped, axis=1)
        self.std_reward = np.std( reshaped, axis=1)
##        pdb.set_trace()
        if totalpoints > 0:
            self.Q_reduced = np.array(self.Q_history)[0:totalpoints:int(totalpoints/numofpoints)]

stat = list()

for i_stat in range(trials):

    p1 = Player(1)
    p2 = Player(2)
    player_list = list()
    player_list.append(p1)
    player_list.append(p2)
    for i in range(n_runs):
        for p in player_list:
            rand_action = np.random.rand(1)
            if rand_action < p.epsilon:
                #do random action
                p.choice = np.random.randint(2)
            else:
                #probabilistically do highest Q action
                val_choice = np.random.rand(1)
                if val_choice < p.Q[0]:
                    p.choice = 0
                else:
                    p.choice = 1

        #calculate reward
        try:
            p1.set_reward(p1.choice, p2.choice)
            p2.set_reward(p1.choice, p2.choice)
        except:
            pdb.set_trace()

        p2.update_Q()
        if scenario == 4:
            p1.update_Q()

    print("Q for round: %s, %s" %(p2.Q, p1.Q))
    stat.append([p2.Q,p1.Q])
    np.mean(np.array(stat), axis = 0)
    print("average Q: %s" %np.mean(np.array(stat), axis = 0))
    p1.reward_avg()
    p2.reward_avg()
    P2_stat_reward.append(p2.avg_reward)
    P2_stat_Q.append(p2.Q_reduced)
    if scenario == 4:
        P1_stat_reward.append(p1.avg_reward)
        P1_stat_Q.append(p1.Q_reduced)
    
convergence = 40
P2_stat_reward_avg = np.mean(np.array(P2_stat_reward),axis = 0)
P2_stat_reward_std = np.std( np.array(P2_stat_reward),axis = 0)
P2_stat_Q_avg      = np.mean(P2_stat_Q, axis=0)
P2_stat_Q_std      = np.std(P2_stat_Q, axis=0)

if scenario == 4:
    P1_stat_reward_avg = np.mean(np.array(P1_stat_reward),axis = 0)
    P1_stat_reward_std = np.std( np.array(P1_stat_reward),axis = 0)
    P1_stat_Q_avg      = np.mean(P1_stat_Q, axis=0)
    P1_stat_Q_std      = np.std(P1_stat_Q, axis=0)

plt.figure()
ind = list(range(len(P2_stat_reward_avg)))
plt.errorbar(ind, P2_stat_reward_avg, yerr=P2_stat_reward_std, fmt='o', color='b', label='Payoff-P2')
plt.errorbar(ind, P2_stat_Q_avg,yerr=P2_stat_Q_std,            fmt=':', color='r', label='Q-P2')
if scenario==4:
    plt.errorbar(ind, P1_stat_reward_avg, yerr=P1_stat_reward_std, fmt='o', color='g', label='Payoff-P1')
    plt.errorbar(ind, P1_stat_Q_avg,yerr=P1_stat_Q_std,            fmt=':', color='k', label='Q-P1')

plt.ylim([0,2.1])
plt.legend(loc='best', fancybox=True)
plt.title('Player 2 Overall Payoff and Q Value for Clod\'s')
if scenario == 4:
    plt.title('Individual Payoff and Q Value for Clod\'s')
plt.xlabel(str(int(n_runs/numofpoints)) + 's of Iterations')
plt.ylabel('Payoff or Action Value')
plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None)
plt.show()

