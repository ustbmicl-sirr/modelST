import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patheffects as path_effects
plt.style.use('ggplot')
plt.rc('font',family='Times New Roman')
matplotlib.rcParams.update({'font.size': 9})


labels = ["others", "add", "gelu", "bn", "ln", "lc", "pwconv", "conv"]
colors = ["#3d3b4f", "#ed5736", "#9ed900", "#007b43", "#69b076", "#4d5aaf", "#f9906f", "#83ccd2"]
x = ["Tiny", "Small", "Base"]
barx = np.arange(len(x)) 


reped_scale = 0.4544828536
rep_scale = 0.41755851
conv_scale = 0.39248771
reped_scale_small = 0.41683878
rep_scale_small = 0.3645514
conv_scale_small = 0.75
reped_scale_base = 0.521939455
rep_scale_base = 0.341000776
conv_scale_base = 0.8575

# convnext
others = np.asarray([conv_scale * 0.2948, conv_scale_small * 0.069,      conv_scale_base * 0.091])
add    = np.asarray([conv_scale * 0.9686, conv_scale_small * 1.873,      conv_scale_base * 1.825])
gelu   = np.asarray([conv_scale * 1.213,  conv_scale_small * 1.372,      conv_scale_base * 1.325])
bn     = np.asarray([conv_scale * 0,      conv_scale_small * 0,          conv_scale_base * 0])
ln     = np.asarray([conv_scale * 9.2396, conv_scale_small * 9.726,      conv_scale_base * 9.13])
lc     = np.asarray([conv_scale * 10.162 , conv_scale_small * 8.387,      conv_scale_base * 8.243])
pwconv = np.asarray([conv_scale * 0,      conv_scale_small * 0,          conv_scale_base * 0])
dwconv = np.asarray([conv_scale * 5.384,  conv_scale_small * 4.174,      conv_scale_base * 4.109])
data = [others, add, gelu, bn, ln, lc, pwconv, dwconv]

# repnext
others_small = np.asarray([rep_scale * 0.277,  rep_scale_small * 0.19,   rep_scale_base * 0.163])
add_small    = np.asarray([rep_scale * 5.813,  rep_scale_small * 9.865,  rep_scale_base * 11.795])
gelu_small   = np.asarray([rep_scale * 0.958,  rep_scale_small * 2.78,   rep_scale_base * 2.579])
bn_small     = np.asarray([rep_scale * 6.394,  rep_scale_small * 13.679,  rep_scale_base * 14.763])
ln_small     = np.asarray([rep_scale * 0,      rep_scale_small * 0,      rep_scale_base * 0])
lc_small     = np.asarray([rep_scale * 0 ,     rep_scale_small * 0,      rep_scale_base * 0])
pwconv_small = np.asarray([rep_scale * 8.857,  rep_scale_small * 16.757,  rep_scale_base * 20.497])
dwconv_small = np.asarray([rep_scale * 6.3,    rep_scale_small * 12.692,    rep_scale_base * 16.0921])
data_small = [others_small, add_small, gelu_small, bn_small, ln_small, lc_small, pwconv_small, dwconv_small]

# repnext_reped
add_base    = np.asarray([reped_scale * 0,      reped_scale_small * 0,      reped_scale_base * 0])
others_base = np.asarray([reped_scale * 0.083,  reped_scale_small * 0.083,  reped_scale_base * 0.066])
gelu_base   = np.asarray([reped_scale * 0.482,  reped_scale_small * 0.882,  reped_scale_base * 0.736])
bn_base     = np.asarray([reped_scale * 0,      reped_scale_small * 0,      reped_scale_base * 0])
ln_base     = np.asarray([reped_scale * 0,      reped_scale_small * 0,      reped_scale_base * 0])
lc_base     = np.asarray([reped_scale * 0,      reped_scale_small * 0,      reped_scale_base * 0])
pwconv_base = np.asarray([reped_scale * 8.445,  reped_scale_small * 16.359, reped_scale_base * 18.26])
dwconv_base = np.asarray([reped_scale * 3.834,  reped_scale_small * 9.063,  reped_scale_base * 8.718])
data_base = [others_base, add_base, gelu_base, bn_base, ln_base, lc_base, pwconv_base, dwconv_base]


bottom_y = [0] * len(x)
bottom_y_small = [0] * len(x)
bottom_y_base = [0] * len(x)

fig, ax1 = plt.subplots(figsize=(2.7, 2.5))
ax1.set_xticks(barx, x)

ax1.set_ylim(ymin = 0, ymax = 24)
for y, y_small, y_base, label,color in zip(data, data_small, data_base, labels, colors):
    ax1.bar(barx-0.23, y, 0.2, color=color, bottom=bottom_y, label=label, edgecolor='black')
    ax1.bar(barx, y_small, 0.2, color=color, bottom=bottom_y_small, edgecolor='black')
    ax1.bar(barx+0.23, y_base, 0.2, color=color, bottom=bottom_y_base, edgecolor='black')
    bottom_y = [a+b for a, b in zip(y, bottom_y)]
    bottom_y_small = [a+b for a, b in zip(y_small, bottom_y_small)]
    bottom_y_base = [a+b for a, b in zip(y_base, bottom_y_base)]

ax1.set_ylabel('Latency (ms)',color='black')

ax2 = ax1.twinx()
ax2.set_ylim(ymin = 0, ymax = 95)
ax2.plot([0-0.23, 0, 0+0.23], [18, 31, 61], color="yellow", ls='--', marker='*', linewidth=0.5, markersize=6, label="GPU util", path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.plot([1-0.23, 1, 1+0.23], [19, 33, 66], color="yellow", ls='--', marker='*', linewidth=0.5, markersize=6, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
ax2.plot([2-0.23, 2, 2+0.23], [21, 44, 89], color="yellow", ls='--', marker='*', linewidth=0.5, markersize=6, path_effects=[path_effects.SimpleLineShadow(), path_effects.Normal()])
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
fig.savefig('./speed_compose_test3.pdf', bbox_inches='tight')