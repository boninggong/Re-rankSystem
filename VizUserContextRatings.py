import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import copy

# Visualizes the amount of ratings per contextual condition
dataset = 'nprs'
user_context_ratings = pd.read_csv(f'input\\{dataset}\\user_context_items.csv', delimiter=',', keep_default_na=False)


def count(x):
    if x == "":
        return 0
    else:
        return len(list(x.split(';')))


user_context_ratings['Frequency'] = user_context_ratings['Item_Rating_Pair'].apply(lambda x: count(x))

res = {}
if dataset == 'nprs':
    contextual_values = ["morning", "afternoon", "evening", "night"]
elif dataset == 'car':
    contextual_values = ["sunny", "snowing", "rainy", "cloudy", "traffic jam", "lots of cars", "sleepy", "awake", "city",
                         "night", "morning", "sad", "lazy", "active", "urban", "mountains", "country side", "coast line",
                         "relaxed driving", "free road", "serpentine", "highway", "day time", "afternoon", "happy",
                         "sport driving"]

buckets = {0:0, 100:0, 200:0, 300:0, 400:0, 500:0, 600:0}

res = {}

for i, row in user_context_ratings.iterrows():
    if row['Context'] not in res:
        res[row['Context']] = copy.deepcopy(buckets)

    if 0 <= row['Frequency'] <= 100:
        res[row['Context']][0] = res[row['Context']][0] + 1
    elif 100 < row['Frequency'] <= 200:
        res[row['Context']][100] = res[row['Context']][100] + 1
    elif 200 < row['Frequency'] <= 300:
        res[row['Context']][200] = res[row['Context']][200] + 1
    elif 300 < row['Frequency'] <= 400:
        res[row['Context']][300] = res[row['Context']][300] + 1
    elif 400 < row['Frequency'] <= 500:
        res[row['Context']][400] = res[row['Context']][400] + 1
    elif 500 < row['Frequency'] <= 600:
        res[row['Context']][500] = res[row['Context']][500] + 1
    else:
        res[row['Context']][600] = res[row['Context']][600] + 1

    # if row['Frequency'] not in res[row['Context']]:
    #     res[row['Context']][row['Frequency']] = 1
    # else:
    #     res[row['Context']][row['Frequency']] = res[row['Context']][row['Frequency']] + 1

for c in contextual_values:
    fig = plt.figure()
    mpl.rcParams.update(mpl.rcParamsDefault)

    # x = []
    # y = []
    #
    # for k in res[c]:
    #     x.append(k)
    #     y.append(res[c][k])
    #
    bars = plt.bar(list(res[c].keys()), list(res[c].values()), width=30, zorder=3)
    for i, b in enumerate(bars):
        bars[i].set_color('#00a6d6')

    plt.xticks(list(res[c].keys()), ["0-100", "101-200", "201-300", "301-400", "401-500", "501-600", "600+"])

    # plt.xticks([x * 2 for x in range(0, math.ceil((max(x) + 1) / 2))])
    # plt.yticks([y * 2 for y in range(0, math.ceil((max(y) + 1) / 2))])
    plt.title(f'Frequency of user ratings for the {c} context')
    plt.xlabel('#ratings given by a specific user in this context')
    plt.ylabel('#users giving these amount of ratings')
    plt.grid(zorder=0)
    plt.plot()

    plt.savefig(f'res\\{dataset}\\viz\\{c}-user-rating-frequency.png', dpi=fig.dpi)
