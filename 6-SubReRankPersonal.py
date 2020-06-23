from __future__ import division
from statistics import mean
from os import path
from Ranking import ndcg_at, mapk, precision_at
import time
import copy
import pandas as pd

# Re-ranks + evaluates sub initial recommendation lists based on the re-rank distance scores of the biggest list
# Re-rank algorithm uses personal mapping in this class
start_time = time.time()

d_set = 'nprs'  # nprs or car
algoName = "BPR"
test_item_amount = 200
k = 5
sub_rerank_sizes = [25, 50, 100]

dist_metric = 'euclidean' # cosine, euclidean
all_metrics = ["MAP", "Prec"] # Prec, MAP
metrics_sizes = [10, 25, 'all']

ind = 1
folds = []
while ind <= k:
    folds.append(f'{algoName}-{ind}')
    ind = ind + 1

lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
afs = ["acousticness", "danceability", "energy", "instrumentalness", "loudness", "speechiness", "tempo", "valence"]
input_rec_path = f'input\\{d_set}_{test_item_amount}\\'

global_models = {}
personal_models = {}
fold_item_init_dist = {}
initial_pred_list = {}
# Keeps track of all calculated distances and used initial ratings for normalization later
all_old = []
all_new = {0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1: []}
all_correct = {}

# Read initial recommendation as input
def read_java_output(file):
    def tple(inp):
        if type(inp) is str:
            resu = tuple(inp.split(';'))
            return int(resu[0]), float(resu[1])
        return 0, 0

    def crt_items(items):
        if type(items) is int:
            return [items]
        else:
            return list(map(int, items.split(';')))

    def context_map(ctx):
        return list(ctx.split(';'))

    def extract_rec_amount(file):
        with open(f'{input_rec_path}{file}.csv', newline='') as f:
            reader = csv.reader(f)
            amount = next(reader)  # gets the first line
            return int(amount[0])

    initial_pred = pd.read_csv(f'{input_rec_path}{file}.csv', delimiter=',', skiprows=1)
    cols = ['userId', 'correctItems', 'contexts']
    recommendations = []
    for i in range(1, extract_rec_amount(file) + 1):
        recommendations.append(f'p{i}')
    cols = cols + recommendations
    initial_pred['correctItems'] = initial_pred['correctItems'].apply(lambda x: crt_items(x))
    initial_pred['contexts'] = initial_pred['contexts'].apply(lambda x: context_map(x))
    for col in initial_pred[recommendations]:
        initial_pred[col] = initial_pred[col].apply(lambda x: tple(x))

    res = []
    for index, row in initial_pred.iterrows():
        r = []
        for c in cols:
            if c == 'contexts':
                for cxt in row[c]:
                    r.append(cxt)
            else:
                r.append(row[c])
        res.append(r)

    return res


# Re-ranks a given list of initial recommendations using a specific lambda
def re_rank(initial_predictions, new_preds, rerank_size):
    # Calculates the final score on which the re-ranking is based using a mix of both the initial ranking and the
    # calculated distance based on audio features
    def calculate_scores(item_init_dist):
        old = []
        new_scores = {0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1: []}
        init_denom = max_init - min_init
        if init_denom == 0:
            init_denom = 1
        dist_denom = max_dist - min_dist
        if dist_denom == 0:
            dist_denom = 1
        for i, (item, initial_pred, distance) in enumerate(item_init_dist):
            if i < rerank_size:
                old.append(item)
                if distance != -99:
                    for l in lambdas:
                        norm_init = (initial_pred - min_init) / init_denom
                        if dist_metric == 'euclidean':
                            norm_dist = (1 - (distance - min_dist) / dist_denom)
                        elif dist_metric == 'cosine':
                            norm_dist = (distance - min_dist) / dist_denom
                        new_scores[l].append((item, ((1 - l) * norm_init + l * norm_dist)))
                else:
                    for l in lambdas:
                        new_scores[l].append((item,-99))
            else:
                break
        return new_scores, old

    def get_rating_of_tuple(t):
        return t[1]

    new_preds, old = calculate_scores(new_preds)
    if new_preds[0] != -99:
        for l in lambdas:
            new_preds[l] = sorted(new_preds[l], key=get_rating_of_tuple, reverse=True)

        new = {0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1: []}

        for l in lambdas:
            for t in new_preds[l]:
                new[l].append(t[0])

        all_old.append(old)
        for l in lambdas:
            all_new[l].append(new[l])


# Keeping track of all correct items for each fold and user-item case
def save_correct_items(fold, recommendation):
    for i, c in enumerate(recommendation):
        if isinstance(c, list):
            all_correct[fold].append(c)
            break


# Extracts the specific contexts in which the initial recommendation was made
def extract_contexts(recommendation):
    contexts = []
    for i, c in enumerate(recommendation[2:]):
        if type(c) is str:
            contexts.append(c.split(":", 1)[1])
        else:
            break
    return contexts


# Extracts the prediction part of the whole initial recommendation
def extract_predictions(recommendation):
    for i, c in enumerate(recommendation):
        if type(c) is tuple:
            return recommendation[i:]
    return []


# Extract distance tuples of (item, initial prediction, distance)
def parse_distance(inp):
    if type(inp) is str:
        inp = inp.replace("(", "")
        inp = inp.replace(")", "")
        resu = tuple(inp.split(','))
        return int(resu[0]), float(resu[1]), float(resu[2])
    return 0, 0, 0


# Creating global models based on global audio feature averages of certain contexts as back up if not enough personal
for f in folds:
    all_correct[f] = []

    num = [int(s) for s in f.split('-') if s.isdigit()]
    if path.isfile(f'input\\{d_set}_global_model\\{num[0]}-global-model.csv'):
        model = pd.read_csv(f'input\\{d_set}_global_model\\{num[0]}-global-model.csv', delimiter=',')
        model_final = {}
        for cntx in model:
            model_final[cntx] = {}
            for num, af in enumerate(afs):
                model_final[cntx][af] = model[cntx][num]
        global_models[f] = model_final

    if path.isfile(f'{input_rec_path}{algoName}-personal-general.csv'):
        distances = pd.read_csv(f'{input_rec_path}{f}-personal-distance.csv', delimiter=',')
        fold_item_init_dist[f] = {}
        for i, col in enumerate(distances):
            fold_item_init_dist[f][i] = [parse_distance(x) for x in distances[col].tolist()]

if path.isfile(f'{input_rec_path}{algoName}-personal-general.csv'):
    general_df = pd.read_csv(f'{input_rec_path}{algoName}-personal-general.csv', delimiter=',')
    max_dist = general_df['max_dist'].values[0]
    min_dist = general_df['min_dist'].values[0]
    max_init = general_df['max_init'].values[0]
    min_init = general_df['min_init'].values[0]
    div_factor_distances = general_df['div_factor_distances'].values[0]
    div_factor_all_ratings = general_df['div_factor_all_ratings'].values[0]

for metric_to_use in all_metrics:
    for rerank_size in sub_rerank_sizes:
        input_rec_path = f'input\\{d_set}_{rerank_size}\\'
        initial_pred_list = {}
        all_correct = {}
        for f in folds:
            all_correct[f] = []

        # Variables used to write the resulting output file
        if metric_to_use == 'Prec':
            final = {'lambda': [], f'{metric_to_use}_{metrics_sizes[0]}_initial': [],
                     f'{metric_to_use}_{metrics_sizes[0]}_rerank': [],
                     f'{metric_to_use}_{metrics_sizes[1]}_initial': [], f'{metric_to_use}_{metrics_sizes[1]}_rerank': []}
        else:
            final = {'lambda': [], f'{metric_to_use}_{metrics_sizes[0]}_initial': [],
                     f'{metric_to_use}_{metrics_sizes[0]}_rerank': [],
                     f'{metric_to_use}_{metrics_sizes[1]}_initial': [], f'{metric_to_use}_{metrics_sizes[1]}_rerank': [],
                     f'{metric_to_use}_{metrics_sizes[2]}_initial': [], f'{metric_to_use}_{metrics_sizes[2]}_rerank': []}

        # Re-rank for each lambda value
        if metric_to_use == 'Prec':
            measures = {f'{metric_to_use}_{metrics_sizes[0]}_initial': [], f'{metric_to_use}_{metrics_sizes[0]}_rerank': [],
                        f'{metric_to_use}_{metrics_sizes[1]}_initial': [], f'{metric_to_use}_{metrics_sizes[1]}_rerank': []}
        else:
            measures = {f'{metric_to_use}_{metrics_sizes[0]}_initial': [], f'{metric_to_use}_{metrics_sizes[0]}_rerank': [],
                        f'{metric_to_use}_{metrics_sizes[1]}_initial': [], f'{metric_to_use}_{metrics_sizes[1]}_rerank': [],
                        f'{metric_to_use}_{metrics_sizes[2]}_initial': [], f'{metric_to_use}_{metrics_sizes[2]}_rerank': []}

        measures_lambda = {0: copy.deepcopy(measures), 0.1: copy.deepcopy(measures), 0.2: copy.deepcopy(measures),
                           0.3: copy.deepcopy(measures), 0.4: copy.deepcopy(measures), 0.5: copy.deepcopy(measures),
                           0.6: copy.deepcopy(measures), 0.7: copy.deepcopy(measures), 0.8: copy.deepcopy(measures),
                           0.9: copy.deepcopy(measures), 1: copy.deepcopy(measures)}

        # Go through each fold
        for f in folds:
            initial_pred_list[f] = read_java_output(f)
            for i, r in enumerate(initial_pred_list[f]):
                save_correct_items(f, r)
                re_rank(extract_predictions(r), fold_item_init_dist[f][i], rerank_size)

            print(f'{input_rec_path}-{len(all_old)}-{len(all_new[1])}-{len(all_correct)}')

            # Calculating all metrics
            if len(all_correct) > 0:
                if metric_to_use == 'MAP':
                    for l in lambdas:
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_initial'].append(
                            mapk(all_old, all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_rerank'].append(
                            mapk(all_new[l], all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_initial'].append(
                            mapk(all_old, all_correct[f], metrics_sizes[1]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_rerank'].append(
                            mapk(all_new[l], all_correct[f], metrics_sizes[1]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[2]}_initial'].append(
                            mapk(all_old, all_correct[f], rerank_size))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[2]}_rerank'].append(
                            mapk(all_new[l], all_correct[f], rerank_size))
                elif metric_to_use == 'NDCG':
                    for l in lambdas:
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_initial'].append(
                            ndcg_at(all_old, all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_rerank'].append(
                            ndcg_at(all_new[l], all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_initial'].append(
                            ndcg_at(all_old, all_correct[f], metrics_sizes[1]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_rerank'].append(
                            ndcg_at(all_new[l], all_correct[f], metrics_sizes[1]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[2]}_initial'].append(
                            ndcg_at(all_old, all_correct[f], rerank_size))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[2]}_rerank'].append(
                            ndcg_at(all_new[l], all_correct[f], rerank_size))
                elif metric_to_use == 'Prec':
                    for l in lambdas:
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_initial'].append(
                            precision_at(all_old, all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[0]}_rerank'].append(
                            precision_at(all_new[l], all_correct[f], metrics_sizes[0]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_initial'].append(
                            precision_at(all_old, all_correct[f], metrics_sizes[1]))
                        measures_lambda[l][f'{metric_to_use}_{metrics_sizes[1]}_rerank'].append(
                            precision_at(all_new[l], all_correct[f], metrics_sizes[1]))

            all_old = []
            all_new = {0: [], 0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: [], 1: []}

        # Printing summary of all metrics after each lambda iteration
        def print_all_measures():
            for lmb in lambdas:
                print(f'Lambda {lmb}')
                final['lambda'].append(lmb)
                for key in measures_lambda[lmb]:
                    print(f'{key}: {mean(measures_lambda[lmb][key])}')
                    final[key].append(mean(measures_lambda[lmb][key]))
                print(f'Total runtime: {time.time() - start_time} s')
                print()


        print_all_measures()

        # Write resulting file
        df = pd.DataFrame.from_dict(final)
        res_name = f'personal-{algoName}-{metric_to_use}'
        if dist_metric == 'cosine':
            res_name += '-cos'

        res_name += '-daytime'
        df.to_csv(f'res\\{d_set}\\{rerank_size}\\{res_name}.csv', index=False, float_format='%.5f')