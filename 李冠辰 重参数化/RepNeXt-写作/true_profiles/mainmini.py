import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
plt.style.use('ggplot')
plt.rc('font',family='Times New Roman')
matplotlib.rcParams.update({'font.size': 7.8})

reped_scale = 0.5124828536
rep_scale = 0.41755851
conv_scale = 0.40248771


labels = ["add", "gelu", "bn", "ln", "lc", "pwconv", "conv"]
colors = ["#ed5736", "#9ed900", "#007b43", "#69b076", "#4d5aaf", "#f9906f", "#83ccd2"]
x = ["ConvNeXt", "RepNeXt\nbefore rep.", "RepNeXt\nafter rep."]
barx = np.arange(len(x)) 

# others = np.asarray([conv_scale * 0.2948, rep_scale * 0.277,  reped_scale * 0.083])
add    = np.asarray([conv_scale * 1.1686, rep_scale * 5.613,  reped_scale * 0])
gelu   = np.asarray([conv_scale * 1.213,  rep_scale * 0.958,  reped_scale * 0.482])
bn     = np.asarray([conv_scale * 0,      rep_scale * 6.394,  reped_scale * 0])
ln     = np.asarray([conv_scale * 9.2396, rep_scale * 0,      reped_scale * 0])
lc     = np.asarray([conv_scale * 9.962 , rep_scale * 0,      reped_scale * 0])
pwconv = np.asarray([conv_scale * 0,      rep_scale * 9.057,  reped_scale * 8.445])
dwconv = np.asarray([conv_scale * 5.384,  rep_scale * 6.3,    reped_scale * 3.834])
data = [add, gelu, bn, ln, lc, pwconv, dwconv]


bottom_y = [0] * len(x)
bottom_y_small = [0] * len(x)
bottom_y_base = [0] * len(x)

fig, ax1 = plt.subplots(figsize=(2.7, 2.5))
ax1.set_xticks(barx, x)

ax1.set_ylim(ymin = 0, ymax = 13)
for y, label,color in zip(data, labels, colors):
    ax1.bar(x, y, 0.5, color=color, bottom=bottom_y, label=label, edgecolor='black')
    bottom_y = [a+b for a, b in zip(y, bottom_y)]

ax1.set_ylabel('Latency (ms)',color='black')

ax2 = ax1.twinx()
ax2.set_ylim(ymin = 0, ymax = 65)
ax2.plot(x, [18, 31, 61], color="yellow", marker='*', linewidth=0.5, markersize=9, ls='--', label="GPU util", path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.set_ylabel("GPU util %",color='black')


num1 = 0.5
num2 = -0.1
num3 = 0.5
num4 = 0
fig.legend(bbox_to_anchor=(num1, num2, num3, num4), prop = {'size':7}, ncol=5)
plt.grid(False)
plt.gcf().subplots_adjust(left=0.15) 
ax1.tick_params(axis='x',colors='black')
ax1.tick_params(axis='y',colors='black')
ax2.tick_params(axis='x',colors='black')
ax2.tick_params(axis='y',colors='black')
# plt.show()
# plt.show()
fig.savefig('./speed_compose_mini.pdf', bbox_inches='tight')