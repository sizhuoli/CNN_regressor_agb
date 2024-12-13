import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb



# bar plot comparing all models
def plot_model_bars(values, color='gray',
                    offset=0.25,
                    ylim=(-1, 1), xs=[1, 2, 3], xlabel=['Deciduous', 'Coniferous', 'Mixed'],
                    color2='firebrick',
                    bw=0.05,
                    linewidth=3,
                    skip=False,
                    uneven=False, plot_descend=False):
    spinexy = True

    sns.set(style="ticks", font_scale=2)
    # sns.set_style({'font.family':'serif', 'font.serif':'Helvetica', 'font.weight':'normal'})
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams['axes.axisbelow'] = True
    nn = len(values)
    mm = len(values[0])
    if not uneven:
        for i in range(1, nn + 1):
            xx = np.linspace(i - offset, i + offset, num=mm)
            # import ipdb; ipdb.set_trace()
            ax.bar(xx, values[i - 1],
                   width=bw, color=color)
    else:

        # import ipdb; ipdb.set_trace()
        ax.bar([1],
               values[0],
               width=bw, color=color)
        ax.bar([2],
               values[1],
               width=bw, color=color)
        ax.bar(np.linspace(3 - offset, 3 + offset, num=2),
               values[2],
               width=bw, color=color)
    if plot_descend:
        ylines1 = [cal_mean_std(values[0])[0], cal_mean_std(values[1])[0]]
        ylines2 = [cal_mean_std(values[1])[0], cal_mean_std(values[2])[0]]

        if not skip:
            plt.plot([1, 2], ylines1, '--', color=color2, linewidth=linewidth, label='Descend')
            if ylines2[1] > ylines2[0]:
                color2 = 'firebrick'

                plt.plot([2, 3], ylines2, '--', color=color2, linewidth=linewidth, label='Ascend')
            else:
                plt.plot([2, 3], ylines2, '--', color=color2, linewidth=linewidth)
        else:
            plt.plot([1, 3], [cal_mean_std(values[0])[0], cal_mean_std(values[2])[0]],
                     '--',
                     color=color2, linewidth=linewidth
                     )
    ax.locator_params(axis='y', nbins=7)
    # plt.legend()
    plt.ylim(ylim)
    plt.grid(True, axis='y', zorder=-1)
    fig.tight_layout()
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.legend(fontsize = 28)
    ax.set_xticks(xs)
    ax.set_xticklabels(xlabel)
    ax.yaxis.set_tick_params(labelsize=20)

    if spinexy:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    else:
        for key, spine in ax.spines.items():
            spine.set_visible(False)

    plt.show()


# only for cnn models with rgb bands

# broadleaf, in order: b0 dense, b1 map, b1 dense etc
bd = [0.67, 0.75, 0.68, 0.66, 0.66, 0.69, 0.74, 0.66]
# coniferous
cf = [0.54, 0.58, 0.54, 0.48, 0.52, 0.54, 0.39, 0.54]
# mixed
mx = [0.65, 0.71, 0.59, 0.50, 0.55, 0.61, 0.69, 0.67]


# plot_model_bars([bd, cf, mx], color = 'gray', color2 = 'slateblue',
#                 xlabel = ['', '', ''],
#                ylim = (0, 1), bw = 0.06, linewidth = 5)

# compute performance is how much lower is for coniferous than broadleaf
low = [(bd[i] - cf[i])/bd[i] for i in range(len(bd))]
# average in percentage
print(np.mean(low)*100)
print(low)