import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Calculates and visualizes a variety of statistics on relevant songs in the initial recommendation lists
# GLOBAL PARAMETERS
d_set = 'nprs'  # nprs or car
algos = ["CAMF_ICS", "BPR", "UserSplitting-BPR"]
test_item_amount = 50  # 25, 50, 100 or 200
k = 5
input_rec_path = f'input\\{d_set}_{test_item_amount}\\'


def sum_indices(algoName):
    folds = []
    ind = 1
    while ind <= k:
        folds.append(f'{algoName}-{ind}')
        ind = ind + 1
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


    def parse_tuple(inp):
        if type(inp) is str:
            resu = tuple(inp.split(';'))
            return int(resu[0]), float(resu[1])
        return 0, 0


    def crt_items(items):
        if type(items) is int:
            return [items]
        else:
            l = list(map(int, items.split(';')))
            return {i: 0 for i in l}


    sums = {}
    rel = {10:0, 25:0, 'total_songs':0}
    for i in range(1, test_item_amount + 1):
        sums[i] = 0

    totalPos = 0
    for f in folds:
        initial_pred = pd.read_csv(f'{input_rec_path}{f}.csv', delimiter=',', skiprows=1)
        initial_pred['correctItems'] = initial_pred['correctItems'].apply(lambda x: crt_items(x))
        recommendations = []
        for i in range(1, test_item_amount + 1):
            recommendations.append(f'p{i}')
        for col in initial_pred[recommendations]:
            initial_pred[col] = initial_pred[col].apply(lambda x: parse_tuple(x))

        for index, row in initial_pred.iterrows():
            for i in range(1, test_item_amount + 1):
                rel['total_songs'] = rel['total_songs'] + 1
                if row[f'p{i}'][0] in row['correctItems']:
                    sums[i] = sums[i] + 1
                    if i <= 10:
                        rel[10] = rel[10] + 1
                    if i <= 25:
                        rel[25] = rel[25] + 1
                    totalPos = totalPos + 1

    for key in sums:
        sums[key] = round(sums[key] / totalPos * 100, 2)

    print(f'{algoName}: {sums}')
    print(f'{algoName}: {rel}')
    print(f'Ratio top 10: {round(rel[10]/totalPos*100,2)} | Ratio top 25: '
          f'{round(rel[25]/totalPos*100,2)} | Total ratio: {round(totalPos/rel["total_songs"]*100,2)}')
    print()

    fig = plt.figure(figsize=(12.0, 6.0))

    plt.bar(range(len(sums)), list(sums.values()), align='center')
    plt.xticks(range(len(sums)), list(sums.keys()))

    plt.title(f'All song ranks and occurrences for the {algoName} initial recommendation algorithm')
    plt.xlabel('Song rank (#)')
    plt.ylabel(f'Relative occurrence in all recommendations (%)')

    plt.savefig(f'res\\{d_set}\\viz\\{algoName}_Rank_Occurrence', dpi=fig.dpi)

    # plt.show()



for algoName in algos:
    sum_indices(algoName)
