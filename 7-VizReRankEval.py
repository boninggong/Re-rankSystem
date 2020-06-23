import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
pd.set_option('display.max_columns', None)

# Visualizes all re-rank and initial recommendation list evaluation results
test_items = [25, 50, 100, 200]
ds = "nprs"
metric = "Prec"
algos = ["RankALS", "CAMF_ICS", "UserSplitting-BPR"]
dist = 'euclidean'

if metric == "Prec":
    metrics_sizes = [10, 25]
elif metric == "MAP":
    metrics_sizes = [10, 25, 'all']


def viz(m, t_item, metric):
    dataset = f"{ds}\\{t_item}\\"

    if dist == 'cosine':
        metric = metric + '-cos'

    bpr_global = pd.read_csv(f'res\\{dataset}\\global-{algos[0]}-{metric}-daytime.csv', delimiter=',')
    bpr_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[0]}-{metric}-daytime.csv', delimiter=',')
    camfics_global = pd.read_csv(f'res\\{dataset}\\global-{algos[1]}-{metric}-daytime.csv', delimiter=',')
    camfics_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[1]}-{metric}-daytime.csv', delimiter=',')
    us_bpr_global = pd.read_csv(f'res\\{dataset}\\global-{algos[2]}-{metric}-daytime.csv', delimiter=',')
    us_bpr_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[2]}-{metric}-daytime.csv', delimiter=',')
    colors = ['#a8fa23', '#548701', '#ff7c7c', '#c10101', '#7eacf2', '#154b9e', '#9a71f5', '#431e96']

    if dist == 'cosine':
        metric = metric[:-4]

    fig = plt.figure(figsize=(12.0, 6.0))
    dct = {'lambda': bpr_global['lambda'].tolist(), 'bpr_initial': bpr_global[f'{metric}_{m}_initial'],
           'bpr_global': bpr_global[f'{metric}_{m}_rerank'],
           'bpr_personal': bpr_personal[f'{metric}_{m}_rerank'].tolist(),
           'camfics_initial': camfics_global[f'{metric}_{m}_initial'].tolist(),
           'camfics_global': camfics_global[f'{metric}_{m}_rerank'].tolist(),
           'camfics_personal': camfics_personal[f'{metric}_{m}_rerank'].tolist(),
           'us-bpr_initial': us_bpr_global[f'{metric}_{m}_initial'].tolist(),
           'us-bpr_global': us_bpr_global[f'{metric}_{m}_rerank'].tolist(), 
           'us-bpr_personal': us_bpr_personal[f'{metric}_{m}_rerank'].tolist()}

    df = pd.DataFrame(dct)
    plt.plot('lambda', 'bpr_initial', data=df, c=colors[2], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'bpr_global', data=df, c=colors[2], linestyle='-.')
    plt.plot('lambda', 'bpr_personal', data=df, c=colors[3], alpha=0.6)
    plt.plot('lambda', 'us-bpr_initial', data=df, c=colors[4], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'us-bpr_global', data=df, c=colors[4], linestyle='-.')
    plt.plot('lambda', 'us-bpr_personal', data=df, c=colors[5], alpha=0.6)
    plt.plot('lambda', 'camfics_initial', data=df, c=colors[6], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'camfics_global', data=df, c=colors[6], linestyle='-.')
    plt.plot('lambda', 'camfics_personal', data=df, c=colors[7], alpha=0.6)

    if metric == 'MAP':
        ylabel = 'Mean Average Precision'
    elif metric == 'Prec':
        ylabel = 'Precision'
    elif metric == 'NDCG':
        ylabel = 'Normalized Discounted Cumulative Gain'

    plt.title(f'Re-rank performance evaluating the {ylabel}@{m} on {t_item} initial recommended songs for the {ds} dataset')
    plt.xlabel('Lambda')
    plt.ylabel(f'{ylabel}')
    fontp = FontProperties()
    fontp.set_size('small')
    plt.legend(loc=8, prop=fontp)
    plt.grid(zorder=0)
    plt.plot()

    if dist == 'cosine':
        metric = metric + '-cos'

    plt.savefig(f'res\\{dataset}\\viz\\{metric}_{m}', dpi=fig.dpi)


for m in metrics_sizes:
    for t_item in test_items:
        viz(m, t_item, metric)
