import pandas as pd
from os import path
import matplotlib.pyplot as plt
import matplotlib as mpl

# Visualizes the distribution of the re-rank distance scores for global model
algoName = 'UISplitting-BPR'
test_item_amount = 25  # 25, 50, 100 or 200
d_set = 'nprs'

all_init = []
all_dist = []

ind = 1
folds = []
while ind <= 10:
    folds.append(f'{algoName}-{ind}')
    ind = ind + 1


# Extract distance tuples of (item, initial prediction, distance)
def parse_distance(inp):
    if type(inp) is str:
        inp = inp.replace("(", "")
        inp = inp.replace(")", "")
        resu = tuple(inp.split(','))
        all_init.append(float(resu[1]))
        all_dist.append(float(resu[2]))


input_rec_path = f'input\\{d_set}_{test_item_amount}\\'
fold_item_init_dist = {}
for f in folds:
    if path.isfile(f'{input_rec_path}{f}-global-distance.csv'):
        distances = pd.read_csv(f'{input_rec_path}{f}-global-distance.csv', delimiter=',')
        fold_item_init_dist[f] = {}
        for i, col in enumerate(distances):
            for x in distances[col].tolist():
                parse_distance(x)

max_dist = max(all_dist)
min_dist = min(all_dist)
max_init = max(all_init)
min_init = min(all_init)

dist_norm = [(x-min_dist)/(max_dist-min_dist) for x in all_dist]
init_norm = [(x-min_init)/(max_init-min_init) for x in all_init]

isDist = True
res = {0: 0, 0.2: 0, 0.4: 0, 0.6: 0, 0.8: 0}
if isDist:
    for item in dist_norm:
        if 0 <= item < 0.2:
            res[0] = res[0] + 1
        elif 0.2 <= item < 0.4:
            res[0.2] = res[0.2] + 1
        elif 0.4 <= item < 0.6:
            res[0.4] = res[0.4] + 1
        elif 0.6 <= item < 0.8:
            res[0.6] = res[0.6] + 1
        elif 0.8 <= item <= 1:
            res[0.8] = res[0.8] + 1
else:
    for item in init_norm:
        if 0 <= item < 0.2:
            res[0] = res[0] + 1
        elif 0.2 <= item < 0.4:
            res[0.2] = res[0.2] + 1
        elif 0.4 <= item < 0.6:
            res[0.4] = res[0.4] + 1
        elif 0.6 <= item < 0.8:
            res[0.6] = res[0.6] + 1
        elif 0.8 <= item <= 1:
            res[0.8] = res[0.8] + 1

fig = plt.figure()
mpl.rcParams.update(mpl.rcParamsDefault)
print(res)
print(len(list(res.keys())))
print(len(list(res.values())))

bars = plt.bar(list(res.keys()), list(res.values()), width=0.1, zorder=3)
for i, b in enumerate(bars):
    bars[i].set_color('#00a6d6')

plt.xticks(list(res.keys()), ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"])

if isDist:
    plt.title(f'Distribution of the normalized distance scores for the {algoName} algorithm')
else:
    plt.title(f'Distribution of the normalized initial scores for the {algoName} algorithm')
plt.xlabel('Range of score')
plt.ylabel('#Occurrences')
plt.grid(zorder=0)
plt.plot()

plt.savefig(f'res\\{d_set}\\viz\\Distr-{algoName}-{test_item_amount}-{isDist}.png', dpi=fig.dpi)
