import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages


def darken_color(color, factor=0.7):
    """Darkens a given color by multiplying the RGB values by the given factor."""
    c = mcolors.to_rgb(color)  # Convert to RGB
    darker = tuple(factor * x for x in c)  # Darken color
    return darker


data = {
    "DeiT": [np.asarray([79.8, 81.8]), np.asarray([8.2, 8.3]), '.', '#1B6299', ["small", "base"]],
    "Swin": [np.asarray([81.3, 83.0, 83.5]), np.asarray([11.2, 22.2, 24.1]), '.', '#E2E72B', ["tiny", "small", "base"]],
    "EfficientNet": [np.asarray([81.6, 82.9, 83.6]), np.asarray([19.7, 24.6, 31.1]), '.', '#83ccd2', ["B3", "B4", "B5"]],
    "ResNet": [np.asarray([77.2, 78.3, 78.6]), np.asarray([9.3, 19.2, 27.9]), '.', '#d8b1d4', ["50", "101", "152"]],
    "ConvNeXt": [np.asarray([82.1, 83.1, 83.8]), np.asarray([10.7, 19.2, 21.2]), '.', '#A36AAA', ["tiny", "small", "base"]],
    "MobileNetV3": [np.asarray([67.5, 75.2]), np.asarray([8.8, 10.8]), '.', '#007CD3', ["small", "large"]],
    "RepVGG": [np.asarray([78.4, 78.8]), np.asarray([5.2, 11.0]), '.', '#FF9671', ["B1", "B2"]],
    "RepLKNet": [np.asarray([83.5]), np.asarray([28]), '.', '#D26F9D', ["31B"]],
    "RepNeXt": [np.asarray([82.0524, 83.13, 83.688]), np.asarray([6.4, 11.1, 14.5]), '*', '#C25E5E', ["tiny", "small", "base"]],
}

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 12})

fig = plt.figure(figsize=(4.4,4), dpi=300)
ax = fig.add_subplot(111)

for key, value in data.items():
    ax.plot(
        value[1], value[0], marker=value[2], color=value[3],
        markersize=12, markeredgecolor=value[3], label=key
    )
    # 为每个点添加文本描述
    for x, y, text in zip(value[1], value[0], value[4]):
        darker_color = darken_color(value[3], factor=0.6)
        ax.text(x, y, text, fontsize=7, ha='right', va='bottom', color=darker_color, weight='bold')

label_fond_dict = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 14,
}
ax.set_xlabel("Latency (ms)", color='black', fontdict=label_fond_dict)
ax.set_ylabel("Top-1 accuracy (%)", color='black', fontdict=label_fond_dict)
plt.legend(prop={'size':10.2}, loc='lower right')
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.grid(ls='--', alpha=0.4) 
plt.tick_params(axis='x', colors='black')
plt.tick_params(axis='y', colors='black')
plt.gcf().subplots_adjust(top=0.9, bottom=0.15)

plt.savefig('./top2024.pdf')
exit(0)



import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages

data = {
    "DeiT": [np.asarray([79.8, 81.8]), np.asarray([8.2, 8.3]), '.', '#1B6299', ["small", "base"]], #yes
    "Swin": [np.asarray([81.3, 83.0, 83.5]), np.asarray([11.2, 22.2, 24.1]), '.', '#E2E72B', ["tiny", "small", "base"]], 
    "EfficientNet": [np.asarray([81.6, 82.9, 83.6]), np.asarray([19.7, 24.6, 31.1]), '.', '#83ccd2', ["B3", "B4", "B5"]],
    "ResNet": [np.asarray([77.2, 78.3, 78.6]), np.asarray([9.3, 19.2, 27.9]), '.', '#d8b1d4', ["50", "101", "152"]],  #yes
    "ConvNeXt": [np.asarray([82.1, 83.1, 83.8]), np.asarray([10.7, 19.2, 21.2]), '.', '#A36AAA', ["tiny", "small", "base"]],
    "MobileNetV3": [np.asarray([67.5, 75.2]), np.asarray([8.8, 10.8]), '.', '#007CD3', ["small", "large"]],
    "RepVGG": [np.asarray([78.4, 78.8]), np.asarray([5.2, 11.0]), '.', '#FF9671', ["B1", "B2"]],
    "RepLKNet": [np.asarray([83.5]), np.asarray([28]), '.', '#D26F9D', ["31B"]],
    "RepNeXt": [np.asarray([82.0524, 83.13, 83.688]), np.asarray([6.4, 11.1, 14.5]), '*', '#C25E5E', ["tiny", "small", "base"]],
}


plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})


fig = plt.figure(figsize=(4.4,4), dpi=300)
ax = fig.add_subplot(111)
for key, value in data.items():
    ax.plot(
        value[1], value[0], marker=value[2], color=value[3],
        markersize=12, markeredgecolor=value[3], label=key
    )

label_fond_dict={
    'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 14,
}
ax.set_xlabel("Latency (ms)", color='black', fontdict=label_fond_dict)
ax.set_ylabel("Top-1 accuracy (%)", color='black', fontdict=label_fond_dict)
plt.legend(prop = {'size':10.2})
ax.xaxis.set_major_locator(ticker.MultipleLocator(3))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.grid(ls='--', alpha=0.4) 
# plt.tick_params(color='black')
plt.tick_params(axis='x',colors='black')
plt.tick_params(axis='y',colors='black')
plt.gcf().subplots_adjust(top=0.9,bottom=0.15) 
# plt.show()
# plt.savefig('./top.pdf')
plt.savefig('./top.jpg')