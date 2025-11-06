import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.style.use('ggplot')
plt.rc('font',family='Times New Roman')
matplotlib.rcParams.update({'font.size': 12})

reped_scale = 0.4544828536
rep_scale = 0.41755851
conv_scale = 1.29851074756

x = ["convnext", "Training time\nrepnext", "Inference time\nrepnext"]

convnext = np.asarray([conv_scale * 1.1686, 
                       conv_scale * 1.213, 
                       conv_scale * 9.2396,
                       conv_scale * 9.962,
                       conv_scale * 5.384, 
                       conv_scale * 0.2948])
convnext_labels = ["add", "gelu", "ln", "lc", "dwconv", "others"]
convnext_colors = ["#ed5736", "#9966CC", "#69b076", "#4d5aaf", "#83ccd2", "#3d3b4f"]
convnext_explode=(0, 0, 0, 0, 0, 0.26)


repnext = np.asarray([rep_scale * 4.613,  
                      rep_scale * 0.958,  
                      rep_scale * 6.394,   
                      rep_scale * 11.757, 
                      rep_scale * 4.6, 
                      rep_scale * 0.277])
repnext_labels = ["add", "gelu", "bn", "pwconv", "dwconv", "others"]
repnext_colors = ["#ed5736", "#9966CC", "#007b43", "#f9906f", "#83ccd2", "#3d3b4f"]
repnext_explode=(0, 0, 0, 0, 0, 0.26)

reped_repnext = np.asarray([reped_scale * 0.482,
                            reped_scale * 11.745,
                            reped_scale * 1.734, 
                            reped_scale * 0.083])
reped_repnext_labels = ["gelu", "pwconv", "gwconv", "others"]
reped_repnext_colors = ["#9966CC", "#f9906f", "#4c8dae", "#3d3b4f"]
reped_repnext_explode=(0, 0, 0, 0.26)

fig, axj = plt.subplots(nrows=2, ncols=2)
axj[0, 1].axis('off')
axes = axj.flatten()
axes[0].pie(x=convnext, labels=convnext_labels, textprops=dict(color='w'), shadow=True, colors=convnext_colors, radius=1.3, explode=convnext_explode, autopct='%3.1f%%')
axes[0].set_xlabel('ConvNeXt 35.4ms', fontsize = 12)
axes[0].xaxis.set_label_coords(0.5, -0.07)
axes[2].pie(x=repnext, labels=repnext_labels, textprops=dict(color='w'),shadow=True, colors=repnext_colors, radius=1.3, explode=repnext_explode, autopct='%3.1f%%')
axes[2].set_xlabel('RepNeXt before rep. 11.9ms', fontsize = 12)
axes[2].xaxis.set_label_coords(0.5, -0.07)
axes[3].pie(x=reped_repnext, labels=reped_repnext_labels, textprops=dict(color='w'),shadow=True, colors=reped_repnext_colors,radius=1.3, explode=reped_repnext_explode, autopct='%3.1f%%')
axes[3].set_xlabel('RepNeXt after rep. 6.6ms', fontsize = 12)
axes[3].xaxis.set_label_coords(0.5, -0.07)
all_lines = []
all_labels = []
use_lines = []
use_labels = []
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    all_lines.extend(axLine)
    all_labels.extend(axLabel)
for i, (li, la) in enumerate(zip(all_lines, all_labels)):
    if la not in use_labels:
        use_lines.append(li)
        use_labels.append(la)

fig.legend(use_lines, use_labels, ncol=2, bbox_to_anchor=(0.92,0.85), loc="upper right", prop={'size': 12})
# plt.show()
plt.savefig('./pie.pdf')