import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
# plt.style.use('ggplot')

data = {
    "DeiT": [np.asarray([79.8, 81.8]), np.asarray([1000/8.2, 1000/8.3]), '.', '#4d82d8'], #yes
    "Swin": [np.asarray([81.3, 83.0, 83.5]), np.asarray([1000/11.2, 1000/22.2, 1000/24.1]), '.', '#800080'], 
    "EfficientNet": [np.asarray([81.6, 82.9, 83.6]), np.asarray([1000/19.7, 1000/24.6, 1000/31.1]), '.', '#83ccd2'],
    "ResNet": [np.asarray([77.2, 78.3, 78.6]), np.asarray([1000/9.3, 1000/19.2, 1000/27.9]), '.', '#d8b1d4'],  #yes
    "ConvNeXt": [np.asarray([82.1, 83.1, 83.8]), np.asarray([1000/10.7, 1000/19.2, 1000/21.2]), '.', 'gray'],
    "MobileNetV3": [np.asarray([67.5, 75.2]), np.asarray([1000/8.8, 1000/10.8]), '.', '#3d3b4f'],
    "RepNeXt": [np.asarray([82.0524, 83.13, 83.688]), np.asarray([1000/6.4, 1000/11.1, 1000/14.5]), '*', 'r'],
}


plt.rc('font',family='Times New Roman')
plt.rcParams.update({'font.size': 12})


fig = plt.figure(figsize=(4.4,4), dpi=300)
ax = fig.add_subplot(111)
for key, value in data.items():
    ax.plot(
        value[1], value[0], marker=value[2], color=value[3],
        markersize=10, markeredgecolor=value[3], label=key
    )

label_fond_dict={
    'family' : 'Times New Roman',
    'weight' : 'normal',
    'size' : 14,
}
ax.set_xlabel("Inference speed (images/sec.)", color='black', fontdict=label_fond_dict)
ax.set_ylabel("Top-1 accuracy", color='black', fontdict=label_fond_dict)
plt.legend()
ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
plt.grid(ls='--', alpha=0.4) 
# plt.tick_params(color='black')
plt.tick_params(axis='x',colors='black')
plt.tick_params(axis='y',colors='black')
plt.gcf().subplots_adjust(top=0.9,bottom=0.15) 
# plt.show()
plt.savefig('./top2.pdf')
# plt.savefig('./top.jpg')