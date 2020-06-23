from __future__ import division
import time
import pandas as pd
import numpy as np
from statistics import mean
from Ranking import ndcg_at, mapk, precision_at

# Calculates the initial recommendation list performance scores
test_item_amounts = [25]
d_set = 'nprs'  # nprs or car
algos = ['UserSplitting-BPR', 'BPR', 'CAMF_ICS']

for algoName in algos:
    folds = []
    ind = 1
    while ind <= 3:
        folds.append(f'{algoName}-{ind}')
        ind = ind + 1
    lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    final_df = {'test_amount': [], 'MAP_10': [], 'MAP_25': [], 'Prec_10': [], 'Prec_25': [],
                'NDCG_10': [], 'NDCG_25': []}
    for test_item_amount in test_item_amounts:
        print(f'{algoName}_{test_item_amount}')
        input_rec_path = f'input\\{d_set}_{test_item_amount}\\'

        map_10 = []
        map_25 = []
        prec_10 = []
        prec_25 = []
        ndcg_10 = []
        ndcg_25 = []
        for f in folds:
            initial_pred = pd.read_csv(f'{input_rec_path}{f}.csv', delimiter=',', skiprows=1)

            correct_items = []
            for xx in initial_pred['correctItems']:
                correct_items.append([int(x) for x in xx.split(';')])

            predictions = []
            for index, row in initial_pred.iterrows():
                cntr = 1
                pred = []
                while cntr <= test_item_amount:
                    if type(row[f'p{cntr}']) is str:
                        pred.append(int(row[f'p{cntr}'].split(';')[0]))
                    cntr = cntr + 1

                predictions.append(pred)

            map_10.append(mapk(predictions, correct_items, 10))
            map_25.append(mapk(predictions, correct_items, 25))
            prec_10.append(precision_at(predictions, correct_items, 10))
            prec_25.append(precision_at(predictions, correct_items, 25))
            ndcg_10.append(ndcg_at(predictions, correct_items, 10))
            ndcg_25.append(ndcg_at(predictions, correct_items, 25))

        final_df['test_amount'].append(test_item_amount)
        final_df['MAP_10'].append(mean(map_10))
        final_df['MAP_25'].append(mean(map_25))
        final_df['Prec_10'].append(mean(prec_10))
        final_df['Prec_25'].append(mean(prec_25))
        final_df['NDCG_10'].append(mean(ndcg_10))
        final_df['NDCG_25'].append(mean(ndcg_25))

    pd.DataFrame.from_dict(final_df).to_csv(f'res\\{d_set}\\{algoName}-baseline.csv', index=False, float_format='%.5f')
