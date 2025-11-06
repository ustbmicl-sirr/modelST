import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
plt.style.use('ggplot')
plt.rc('font',family='Times New Roman')
matplotlib.rcParams.update({'font.size': 9})

reped_scale = 0.4544828536
rep_scale = 0.41755851
conv_scale = 0.39248771


labels = ["others", "add", "gelu", "bn", "ln", "lc", "pwconv", "conv"]
colors = ["#3d3b4f", "#ed5736", "#9ed900", "#007b43", "#69b076", "#4d5aaf", "#f9906f", "#83ccd2"]
x = ["ConvNeXt", "RepneXt\nbefore rep.", "RepneXt\nafter rep."]
barx = np.arange(len(x)) 

others = np.asarray([conv_scale * 0.2948, rep_scale * 0.277,  reped_scale * 0.083])
add    = np.asarray([conv_scale * 1.1686, rep_scale * 5.613,  reped_scale * 0])
gelu   = np.asarray([conv_scale * 1.213,  rep_scale * 0.958,  reped_scale * 0.482])
bn     = np.asarray([conv_scale * 0,      rep_scale * 6.394,  reped_scale * 0])
ln     = np.asarray([conv_scale * 9.2396, rep_scale * 0,      reped_scale * 0])
lc     = np.asarray([conv_scale * 9.962 , rep_scale * 0,      reped_scale * 0])
pwconv = np.asarray([conv_scale * 0,      rep_scale * 9.057,  reped_scale * 8.445])
dwconv = np.asarray([conv_scale * 5.384,  rep_scale * 6.3,    reped_scale * 3.834])
data = [others, add, gelu, bn, ln, lc, pwconv, dwconv]

reped_scale_small = 0.41683878
rep_scale_small = 0.3645514
conv_scale_small = 0.75
others_small = np.asarray([conv_scale_small * 0.069,  rep_scale_small * 0.19,  reped_scale_small * 0.083])
add_small    = np.asarray([conv_scale_small * 2.173,  rep_scale_small * 9.565,  reped_scale_small * 0])
gelu_small   = np.asarray([conv_scale_small * 1.372,  rep_scale_small * 2.78,  reped_scale_small * 0.882])
bn_small     = np.asarray([conv_scale_small * 0,      rep_scale_small * 13.679,  reped_scale_small * 0])
ln_small     = np.asarray([conv_scale_small * 9.726, rep_scale_small * 0,      reped_scale_small * 0])
lc_small     = np.asarray([conv_scale_small * 8.087 , rep_scale_small * 0,      reped_scale_small * 0])
pwconv_small = np.asarray([conv_scale_small * 0,      rep_scale_small * 19.057,  reped_scale_small * 17.359])
dwconv_small = np.asarray([conv_scale_small * 4.174,  rep_scale_small * 10.692,    reped_scale_small * 8.063])
data_small = [others_small, add_small, gelu_small, bn_small, ln_small, lc_small, pwconv_small, dwconv_small]


reped_scale_base = 0.521939455
rep_scale_base = 0.341000776
conv_scale_base = 0.8575
others_base = np.asarray([conv_scale_base * 0.091, rep_scale_base * 0.163,  reped_scale_base * 0.066])
add_base    = np.asarray([conv_scale_base * 2.125, rep_scale_base * 11.395,  reped_scale_base * 0])
gelu_base   = np.asarray([conv_scale_base * 1.325,  rep_scale_base * 2.579,  reped_scale_base * 0.736])
bn_base     = np.asarray([conv_scale_base * 0,      rep_scale_base * 14.763,  reped_scale_base * 0])
ln_base     = np.asarray([conv_scale_base * 9.13, rep_scale_base * 0,      reped_scale_base * 0])
lc_base     = np.asarray([conv_scale_base * 7.943 , rep_scale_base * 0,      reped_scale_base * 0])
pwconv_base = np.asarray([conv_scale_base * 0,      rep_scale_base * 20.897,  reped_scale_base * 19.26])
dwconv_base = np.asarray([conv_scale_base * 4.109,  rep_scale_base * 16.0921,    reped_scale_base * 7.718])
data_base = [others_base, add_base, gelu_base, bn_base, ln_base, lc_base, pwconv_base, dwconv_base]


bottom_y = [0] * len(x)
bottom_y_small = [0] * len(x)
bottom_y_base = [0] * len(x)

fig, ax1 = plt.subplots(figsize=(2.7, 2.5))
ax1.set_xticks(barx, x)

ax1.set_ylim(ymin = 0, ymax = 24)
for y, y_small, y_base, label,color in zip(data, data_small, data_base, labels, colors):
    ax1.bar(barx-0.21, y, 0.2, color=color, bottom=bottom_y, label=label)
    ax1.bar(barx, y_small, 0.2, color=color, bottom=bottom_y_small)
    ax1.bar(barx+0.21, y_base, 0.2, color=color, bottom=bottom_y_base)
    bottom_y = [a+b for a, b in zip(y, bottom_y)]
    bottom_y_small = [a+b for a, b in zip(y_small, bottom_y_small)]
    bottom_y_base = [a+b for a, b in zip(y_base, bottom_y_base)]

ax1.set_ylabel('speed (ms)',color='black')

ax2 = ax1.twinx()
ax2.set_ylim(ymin = 0, ymax = 95)
ax2.plot([0-0.21, 0, 0+0.21], [18, 19, 21], color="yellow", marker='*', linewidth=0.5, markersize=6, label="GPU util", path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.plot([1-0.21, 1, 1+0.21], [31, 33, 44], color="yellow", marker='*', linewidth=0.5, markersize=6, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.plot([2-0.21, 2, 2+0.21], [61, 66, 89], color="yellow", marker='*', linewidth=0.5, markersize=6, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.set_ylabel("GPU util %")


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
fig.savefig('./speed_compose_test.pdf', bbox_inches='tight')