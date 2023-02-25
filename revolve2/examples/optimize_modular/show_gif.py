import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

generations = 20
nframes = generations
plt.subplots_adjust(top=1, bottom=0, left=0, right=1)

def animate(i):
    png_name = "Plots directory/" + "Fronts plot" + str(i) + ".png"
    im = plt.imread(png_name)
    plt.imshow(im)

anim = FuncAnimation(plt.gcf(), animate, frames=nframes, interval=2000)
anim.save('pareto_evolution.gif', writer='imagemagick')
