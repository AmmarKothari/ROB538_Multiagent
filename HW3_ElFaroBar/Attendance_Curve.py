import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math as math




def attendance(z,b):
    #z i s the attendance on a given night
    r = np.multiply(z,np.exp(-z/b))
    return r


    
total_possible = np.arange(0,100,1)
b1 = 5
for b1 in [5, 10, 20, 40]:
    reward = attendance(total_possible,b1)
    plt.plot(total_possible,reward,label='OA = %s' %b1)


plt.ylabel('reward')
plt.xlabel('attendance')
plt.title('Reward for Attending on Night with Optimal Attendance')
plt.legend()
plt.show()

