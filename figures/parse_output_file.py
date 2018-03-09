import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: ")
    print("    python {} <log file>".format(sys.argv[0]))
    sys.exit()

logfile = sys.argv[1]

steps = []
loss_d = []
loss_g = []

with open(logfile, 'r') as flog:
    for line in flog:
        line = line.strip().split()
        if len(line) == 10 and line[6] == 'generator':
            steps.append( int(line[1][:line[1].find(':')]) )
            loss_d.append( float(line[5][:line[5].find(',')]) )
            loss_g.append( float(line[9]) )


# Plot the discriminant and generator losses

# fig, ax1 = plt.subplots()
# 
# ax1.plot(steps, loss_d, 'b-')
# ax1.set_xlabel('iteration')
# # Make the y-axis label, ticks and tick labels match the line color.
# ax1.set_ylabel('Discriminant loss', color='b')
# ax1.tick_params('y', colors='b')
# 
# ax2 = ax1.twinx()
# ax2.plot(steps, loss_g, 'r-')
# ax2.set_ylabel('Generator loss', color='r')
# ax2.tick_params('y', colors='r')

fig = plt.figure()

plt.semilogy(steps, loss_d, 'b-', label='Discriminator')
plt.semilogy(steps, loss_g, 'r-', label='Generator')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.legend()

fig.tight_layout()
plt.show()
