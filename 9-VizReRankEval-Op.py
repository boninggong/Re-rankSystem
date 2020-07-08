import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
pd.set_option('display.max_columns', None)

# Visualizes all opposite re-rank and initial recommendation list evaluation results
test_items = [25, 50, 100, 200]
ds = "nprs"
metric = "Prec"
algos = ["BPR", "CAMF_ICS", "UserSplitting-BPR"]
dist = 'euclidean'

if metric == "Prec":
    metrics_sizes = [10, 25]
elif metric == "MAP":
    metrics_sizes = [10, 25, 'all']


def viz(m, t_item, metric):
    dataset = f"{ds}\\{t_item}\\"

    bpr_global = pd.read_csv(f'res\\{dataset}\\global-{algos[0]}-{metric}-daytime-op.csv', delimiter=',')
    bpr_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[0]}-{metric}-daytime-op.csv', delimiter=',')
    camfics_global = pd.read_csv(f'res\\{dataset}\\global-{algos[1]}-{metric}-daytime-op.csv', delimiter=',')
    camfics_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[1]}-{metric}-daytime-op.csv', delimiter=',')
    us_bpr_global = pd.read_csv(f'res\\{dataset}\\global-{algos[2]}-{metric}-daytime-op.csv', delimiter=',')
    us_bpr_personal = pd.read_csv(f'res\\{dataset}\\personal-{algos[2]}-{metric}-daytime-op.csv', delimiter=',')
    colors = ['#a8fa23', '#548701', '#9eff59', '#c10101', '#0a00c2', '#154b9e', '#ba42ff', '#431e96']

    fig = plt.figure(figsize=(12.0, 6.0))
    dct = {'lambda': bpr_global['lambda'].tolist(), 'BPR_initial': bpr_global[f'{metric}_{m}_initial'],
           'BPR_global': bpr_global[f'{metric}_{m}_rerank'],
           'BPR_personal': bpr_personal[f'{metric}_{m}_rerank'].tolist(),
           'CAMF_ICS_initial': camfics_global[f'{metric}_{m}_initial'].tolist(),
           'CAMF_ICS_global': camfics_global[f'{metric}_{m}_rerank'].tolist(),
           'CAMF_ICS_personal': camfics_personal[f'{metric}_{m}_rerank'].tolist(),
           'US-BPR_initial': us_bpr_global[f'{metric}_{m}_initial'].tolist(),
           'US-BPR_global': us_bpr_global[f'{metric}_{m}_rerank'].tolist(),
           'US-BPR_personal': us_bpr_personal[f'{metric}_{m}_rerank'].tolist()}

    df = pd.DataFrame(dct)
    plt.plot('lambda', 'BPR_initial', data=df, c=colors[2], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'BPR_global', data=df, c=colors[2], linestyle='-.')
    plt.plot('lambda', 'BPR_personal', data=df, c=colors[2], alpha=0.8)
    plt.plot('lambda', 'US-BPR_initial', data=df, c=colors[4], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'US-BPR_global', data=df, c=colors[4], linestyle='-.')
    plt.plot('lambda', 'US-BPR_personal', data=df, c=colors[4], alpha=0.8)
    plt.plot('lambda', 'CAMF_ICS_initial', data=df, c=colors[6], linestyle='dashed', alpha=1)
    plt.plot('lambda', 'CAMF_ICS_global', data=df, c=colors[6], linestyle='-.')
    plt.plot('lambda', 'CAMF_ICS_personal', data=df, c=colors[6], alpha=0.8)

    if ds == 'nprs':
        dataset_label = "#NowPlaying-RS"

    plt.title(f'Opposite re-rank performance evaluating {metric}@{m} on {t_item} initial '
              f'recommended songs for the {dataset_label} dataset')
    plt.xlabel('Lambda')
    plt.ylabel(f'{metric}@{m}')
    fontp = FontProperties()
    fontp.set_size('small')
    plt.legend(loc=6, prop=fontp)
    plt.grid(zorder=0)
    plt.plot()

    plt.savefig(f'res\\{dataset}\\viz\\{metric}_{m}-op', dpi=fig.dpi)


for m in metrics_sizes:
    for t_item in test_items:
        viz(m, t_item, metric)
