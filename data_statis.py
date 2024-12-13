import glob

import numpy as np
import pandas as pd
import ipdb
import matplotlib.pyplot as plt
from docutils.nodes import header
from matplotlib.font_manager import font_family_aliases
from scipy.ndimage import label
import rasterio
path1 = '/home/sizhuo/Desktop/AGB_project/train_plots_1897.csv'
df1 = pd.read_csv(path1)
path2 = '/home/sizhuo/Desktop/AGB_project/valid_plots268.csv'
df2 = pd.read_csv(path2)
path3 = '/home/sizhuo/Desktop/AGB_project/test_plots258.csv'
df3 = pd.read_csv(path3)

# concat
df = pd.concat([df1, df2, df3])
# ipdb.set_trace()
# column names
# randomly sample 3 examples with low agb from each group
# low agb and coniferous
low_agb = df1[(df1['BMag_ha'] < 100) & (df1['forest_type'] == 'B')]
low_agb = low_agb.sample(3, random_state=0)
# high agb
high_agb = df1[(df1['BMag_ha'] > 200) & (df1['forest_type'] == 'B')]
high_agb = high_agb.sample(3, random_state=0)

# plot
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
for i in range(3):
    # low agb
    filename = glob.glob('/home/sizhuo/Desktop/code_repository/CNN_AGB/dataset/train/' + low_agb.iloc[i]['psussu'] + '*.tif')[0]
    with rasterio.open(filename) as src:
        ax[0, i].imshow(np.transpose(src.read((1, 2, 3)), (1, 2, 0))
                        )
        ax[0, i].set_title(f'AGBD: {low_agb.iloc[i]["BMag_ha"]:.1f} t/ha')
    # high agb
    with rasterio.open(glob.glob('/home/sizhuo/Desktop/code_repository/CNN_AGB/dataset/train/' + high_agb.iloc[i]['psussu'] + '*.tif')[0]) as src:
        ax[1, i].imshow(np.transpose(src.read((1, 2, 3)), (1, 2, 0))
                        )
        ax[1, i].set_title(f'AGBD: {high_agb.iloc[i]["BMag_ha"]:.1f} t/ha')
# # circle plot proportion of samples based on forest type
# # nan values as unknown
# df['forest_type'] = df['forest_type'].fillna('Unknown')
# df['forest_type'] = df['forest_type'].replace('B', 'Deciduous')
# df['forest_type'] = df['forest_type'].replace('C', 'Coniferous')
# df['forest_type'] = df['forest_type'].replace('M', 'Mixed')
# df['forest_type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=0, legend=False, colors=['skyblue', 'yellowgreen', 'tomato', 'gray'], fontsize=13, label = '')
# plt.show()



# split into three groups based on forest type
agb_b = df[df['forest_type'] == 'B']['BMag_ha']
agb_c = df[df['forest_type'] == 'C']['BMag_ha']
agb_m = df[df['forest_type'] == 'M']['BMag_ha']

# plot boxplot
plt.figure(figsize=(10, 6))
plt.boxplot([agb_b, agb_c, agb_m], labels=['Deciduous', 'Coniferous', 'Mixed'], showmeans=True, meanline=True, meanprops={'color': 'red', 'linewidth': 2}, medianprops={'color': 'green', 'linewidth': 2},
            boxprops={'color': 'gray', 'linewidth': 2}, whiskerprops={'color': 'gray', 'linewidth': 2}, capprops={'color': 'gray', 'linewidth': 2}, flierprops={'markeredgecolor': 'gray', 'markersize': 8})

# add text showing mean value
mean_b = agb_b.mean()
mean_c = agb_c.mean()
mean_m = agb_m.mean()
plt.text(1.28, mean_b, f'{mean_b:.1f}', fontsize=12, ha='center', va='center')
plt.text(2.28, mean_c, f'{mean_c:.1f}', fontsize=12, ha='center', va='center')
plt.text(3.28, mean_m, f'{mean_m:.1f}', fontsize=12, ha='center', va='center')
# add x ticknames
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.ylabel('AGBD (t/ha)', fontsize=13)
plt.show()












# # process unique training plots
# path = '/home/sizhuo/Desktop/AGB_project/train_plots.csv'
# df = pd.read_csv(path)
# print(df.head())
# print(len(df))
#
# # ipdb.set_trace()
# # unique values
# print(len(df['psussu'].unique()))
#
# # if multiple rows with same psussu, keep mean agb
# unique_ids = df['psussu'].unique()
# df2 = pd.DataFrame(columns=df.columns)
# for uid in unique_ids:
#     sub = df[df['psussu'] == uid]
#     if len(sub) > 1:
#         # ipdb.set_trace()
#         # keep first row and update agb
#         avg = sub['BMag_ha'].mean()
#         sub2 = sub.head(1)
#         # udate agb
#         sub2.iloc[0, sub2.columns.get_loc('BMag_ha')] = avg
#     else:
#         sub2 = sub
#     df2 = pd.concat([df2, sub2])
#
# print(len(df2))
