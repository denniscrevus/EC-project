import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

generations = 50
nframes = generations
plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

def animate(i):
    png_name = "Experiment_max_power_129_max_parts10/Run0/Plots directory/" + "Fronts plot" + str(i) + ".png"
    im = plt.imread(png_name)
    plt.imshow(im)

anim = FuncAnimation(plt.gcf(), animate, frames=nframes, interval=2000)
anim.save('pareto_evolution.gif', writer='imagemagick')
