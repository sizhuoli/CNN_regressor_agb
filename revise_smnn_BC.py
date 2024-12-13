
import geopandas as gpd
import numpy as np
from IPython.testing.ipunittest import ip2py
import ipdb
from matplotlib import cm
from matplotlib.cm import cmap_d
from networkx.algorithms.bipartite.basic import color
from rasterio.rio.helpers import coords
from stack_data import markers_from_ranges
import matplotlib.pyplot as plt
from core.predictor import alpha_loss

file = '/home/sizhuo/Desktop/AGB_project/allometric_smnn_test/test_area_smnn_pred_nonnegative.gpkg'

# load into a df
gdf = gpd.read_file(file)

# split into two groups based on the column 'smnn'
gdf_dc = gdf[gdf['_ForestTypeCopernicusmajority'] == 1]
gdf_cf = gdf[gdf['_ForestTypeCopernicusmajority'] == 2]
gdf_non = gdf[gdf['_ForestTypeCopernicusmajority'] == 0]

print(len(gdf_dc))
print(len(gdf_cf))
print(len(gdf_non))

# sample some from each group

gdf_dc = gdf_dc.dropna(subset=['_updated_lidarH_maxmax', 'crownDiameter', 'SMNN_AGB_t'])
gdf_cf = gdf_cf.dropna(subset=['_updated_lidarH_maxmax', 'crownDiameter', 'SMNN_AGB_t'])
gdf_dc_sample = gdf_dc.sample(2000, random_state=0)
gdf_cf_sample = gdf_cf.sample(2000, random_state=0)
#
# make 2d histogram, x=crown, y = treeheight, color = smnn
# plot
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 2, figsize=(15, 7))
# # use same color normalization for two plots
# # all agb
agb = np.concatenate([gdf_dc_sample['SMNN_AGB_t'], gdf_cf_sample['SMNN_AGB_t']])
max_agb = agb.max()
cmap = cm.get_cmap('Greens', 5)
dc_trans = cmap(gdf_dc_sample['SMNN_AGB_t'])
cf_trans = cmap(gdf_cf_sample['SMNN_AGB_t'])
#
# ax[0].scatter(gdf_dc_sample['crownDiameter'], gdf_dc_sample['_updated_lidarH_maxmax'], c = dc_trans, norm = plt.Normalize(0, 100))
# # ax[0].set_title('DC')
# # same axis limit for both plots
# # tick size
# ax[0].tick_params(axis='both', which='major', labelsize=20)
# ax[0].set_xlim([0, 20])
# ax[0].set_ylim([0, 50])
# ax[1].scatter(gdf_cf_sample['crownDiameter'], gdf_cf_sample['_updated_lidarH_maxmax'], c = cf_trans, norm = plt.Normalize(0, 100))
# # ax[1].set_title('CF')
# # tick size
# ax[1].tick_params(axis='both', which='major', labelsize=20)
# ax[1].set_xlim([0, 20])
# ax[1].set_ylim([0, 50])
# # colorbar
# # add colorbar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_agb))
# sm.set_array([])
# # make colobar smaller, set fontsize
# cb = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
# cb.ax.tick_params(labelsize=20)
#
#
#
# # plt.colorbar()
# plt.savefig('/home/sizhuo/Pictures/spe.png')
#

#
# # overlay the two plots
# fig, ax = plt.subplots()
# cmap2 = cm.get_cmap('Reds', 5)
# cf_trans2 = cmap2(gdf_cf_sample['SMNN_AGB_t'])
# ax.scatter(gdf_dc_sample['crownDiameter'], gdf_dc_sample['_updated_lidarH_maxmax'], c=dc_trans, label='DC', alpha=0.2, s=10)
# ax.scatter(gdf_cf_sample['crownDiameter'], gdf_cf_sample['_updated_lidarH_maxmax'], c=cf_trans2, label='CF', alpha=0.2, s=10)
# ax.set_title('Crown diameter vs Tree height')
# ax.set_xlabel('Crown Diameter')
# ax.set_ylabel('Tree height')
# ax.legend()
# plt.savefig('/home/sizhuo/Pictures/spe1.png')


#
# # plot crown diamter vs biomass, split forest type
# fig, ax = plt.subplots(1, 3)
# ax[0].scatter(gdf_dc_sample['crownDiameter'], gdf_dc_sample['SMNN_AGB_t'], c='blue', label='DC', alpha=0.2, s=2)
# ax[0].scatter(gdf_cf_sample['crownDiameter'], gdf_cf_sample['SMNN_AGB_t'], c='green', label='CF', alpha=0.2, s = 2)
# ax[0].set_title('Crown diameter vs Biomass')
# ax[0].set_xlabel('Crown Diameter')
# ax[0].set_ylabel('Biomass')
# ax[0].legend()
# ax[0].grid()
#
# # plot tree height vs biomass, split forest type
# ax[1].scatter(gdf_dc_sample['_updated_lidarH_maxmax'], gdf_dc_sample['SMNN_AGB_t'], c='blue', label='DC', alpha=0.2, s=2)
# ax[1].scatter(gdf_cf_sample['_updated_lidarH_maxmax'], gdf_cf_sample['SMNN_AGB_t'], c='green', label='CF', alpha=0.2, s = 2)
# ax[1].set_title('Tree height vs Biomass')
# ax[1].set_xlabel('Tree height')
# ax[1].set_ylabel('Biomass')
# ax[1].legend()
# ax[1].grid()
#
# # plot height*diameter vs biomass, split forest type
# ax[2].scatter(gdf_dc_sample['crownDiameter']*gdf_dc_sample['_updated_lidarH_maxmax'], gdf_dc_sample['SMNN_AGB_t'], c='blue', label='DC', alpha=0.2, s=2)
# ax[2].scatter(gdf_cf_sample['crownDiameter']*gdf_cf_sample['_updated_lidarH_maxmax'], gdf_cf_sample['SMNN_AGB_t'], c='green', label='CF', alpha=0.2, s = 2)
# ax[2].set_title('Height*Diameter vs Biomass')
# ax[2].set_xlabel('Height*Diameter')
# ax[2].set_ylabel('Biomass')
# # log scale
# # ax[2].set_yscale('log')
# # ax[2].set_xscale('log')
# ax[2].legend()
# ax[2].grid()
# plt.savefig('/home/sizhuo/Pictures/spe2.png')
#


