import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ipdb
nfi_B_total = 24.34
nfi_C_total = 16.84


nfi_B_above = nfi_B_total*0.8208
nfi_B_below = nfi_B_total*0.1792

nfi_C_above = nfi_C_total*0.8208
nfi_C_below = nfi_C_total*0.1792

pred_B_above = 23.69
pred_C_above = 11.77

# total above ours compared to nfi
pred_total = pred_B_above + pred_C_above
nfi_total = nfi_B_above + nfi_C_above
print('total bias: ', ((pred_total - nfi_total)/nfi_total)*100)
print('proportion of different forest type in nfi: B and c ', nfi_B_above/nfi_total, nfi_C_above/nfi_total)
print('proportion of different forest type in ours: B and c ', (pred_B_above)/pred_total, (pred_C_above)/pred_total)

ipdb.set_trace()



# plot stacked bar plot, three groups, NFI total, NFI above, pred above, each stacked with b, c
fig, ax = plt.subplots(figsize=(10, 8))
# NFI
nfi = [nfi_B_total, nfi_C_total]
nfi_above = [nfi_B_above, nfi_C_above]
pred_above = [pred_B_above, pred_C_above]
values = {
    "Broadleaved": np.array([nfi[0], nfi_above[0], pred_above[0]]),
    "Coniferous": np.array([nfi[1], nfi_above[1], pred_above[1]])
}
# plot
# bar width
bw = 0.3
# x
x = [1, 2, 3
        ]
# colors
colors = ['steelblue', 'chocolate', 'darkorange']
# labels
bottom = [0, 0, 0]
labels = ['NFI above+below', 'NFI above proxy', 'Predicted above']
# grid
ax.grid(axis='y', linestyle='--', alpha=0.6)
for i, key in enumerate(values.keys()):
    ax.bar(x, values[key], width=bw, color=colors[i], label=key, bottom=bottom, alpha=0.9)
    bottom = np.array(values[key]) + bottom


# set xtick
ax.set_xticks(x)
ax.set_xticklabels([], fontsize=30)
# set ytick
yticks = np.arange(0, 50, 5)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=26)
plt.tight_layout()

# upper axis off
ax.spines['top'].set_visible(False)
# right axis off
ax.spines['right'].set_visible(False)
plt.legend(fontsize=26)
plt.show()
