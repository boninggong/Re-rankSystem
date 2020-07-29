import pandas as pd
import csv

keepPositive = True  # Removes all recommendations where there are 0 positive songs recommended
createSubSet = True  # Creates recommendation lists of 25, 50 and 100 songs, respectively
convertRankingScores = True  # Only use with ranking based initial recommendation algorithms

test_item_amounts = [200]
d_set = 'nprs'
algos = ['BPR', 'UserSplitting-BPR', 'CAMF_ICS']
k = 5

if keepPositive:
    for algoName in algos:
        folds = []
        ind = 1
        while ind <= k:
            folds.append(f'{algoName}-{ind}')
            ind = ind + 1
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        for test_item_amount in test_item_amounts:
            input_rec_path = f'input\\{d_set}_{test_item_amount}\\'

            for f in folds:
                print(f'Removing non positive for {algoName} | fold {f}')
                initial_pred = pd.read_csv(f'{input_rec_path}{f}.csv', delimiter=',', skiprows=1)

                valid_rows = []
                invalid_rows = []
                for index, row in initial_pred.iterrows():
                    correct_items = [int(x) for x in row['correctItems'].split(';')]
                    cntr = 1
                    pred = []
                    while cntr <= test_item_amount:
                        if type(row[f'p{cntr}']) is str:
                            pred.append(int(row[f'p{cntr}'].split(';')[0]))
                        cntr = cntr + 1

                    if not any(x in pred for x in correct_items):
                        initial_pred.drop(index, inplace=True)

                with open(f'{input_rec_path}{f}.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([test_item_amount])

                initial_pred.reset_index(drop=True, inplace=True)
                initial_pred.to_csv(f'{input_rec_path}{f}.csv', index=False, mode='a', float_format='%.5f')

print()
if createSubSet:
    target_amounts = [25, 50, 100]
    for algoName in algos:
        folds = []
        ind = 1
        while ind <= k:
            folds.append(f'{algoName}-{ind}')
            ind = ind + 1
        lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        for f in folds:
            print(f'Creating sub sets for {algoName} | fold {f}')
            initial_pred = pd.read_csv(f'input\\{d_set}_{test_item_amounts[0]}\\{f}.csv', delimiter=',', skiprows=1)

            for t_amount in target_amounts:
                new_pred = {'userId': initial_pred['userId'].values, 'correctItems': initial_pred['correctItems'].values,
                            'contexts': initial_pred['contexts'].values}
                cntr = 1
                while cntr <= t_amount:
                    new_pred[f'p{cntr}'] = initial_pred[f'p{cntr}'].values
                    cntr = cntr + 1

                full_path = f'input\\{d_set}_{t_amount}\\{f}.csv'
                with open(full_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([t_amount])

                df = pd.DataFrame.from_dict(new_pred)
                df.to_csv(full_path, index=False, mode='a', float_format='%.5f')

print()
if convertRankingScores:
    # Create scores based on rank, not the initial recommender algorithms scores
    amounts = [25, 50, 100, 200]
    for test_item_amount in amounts:
        input_rec_path = f'input\\{d_set}_{test_item_amount}\\'

        for algoName in algos:
            folds = []
            ind = 1
            while ind <= k:
                folds.append(f'{algoName}-{ind}')
                ind = ind + 1
            lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

            for f in folds:
                print(f'Converting score for {algoName}_{test_item_amount} | fold {f}')
                initial_pred = pd.read_csv(f'{input_rec_path}{f}.csv', delimiter=',', skiprows=1)
                new_pred = {'userId': [], 'correctItems': [], 'contexts': []}
                scores = []

                cntr = 1
                while cntr <= test_item_amount:
                    new_pred[f'p{cntr}'] = []
                    scores.append(test_item_amount - cntr + 1)
                    cntr = cntr + 1

                for index, row in initial_pred.iterrows():
                    new_pred['userId'].append(row['userId'])
                    new_pred['correctItems'].append(row['correctItems'])
                    new_pred['contexts'].append(row['contexts'])

                    cntr = 1
                    while cntr <= test_item_amount:
                        tple = row[f'p{cntr}']
                        if isinstance(tple, str):
                            item = tple.split(';')[0]
                            new_pred[f'p{cntr}'].append(f'{item};{scores[cntr-1]}')
                        else:
                            new_pred[f'p{cntr}'].append('')
                        cntr = cntr + 1

                full_path = f'input\\{d_set}_{test_item_amount}\\{f}.csv'
                with open(full_path, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([test_item_amount])

                df = pd.DataFrame.from_dict(new_pred)
                df.to_csv(full_path, index=False, mode='a', float_format='%.5f')
