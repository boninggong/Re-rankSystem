import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

# Visualizes the initial recommendation list evaluation results
ds = "nprs"
metrics = ['MAP', 'NDCG', 'Prec']
metrics_sizes = [10, 25]
algos = ["BPR", "UserSplitting-BPR", "CAMF_ICS"]

bpr = pd.read_csv(f'res\\{ds}\\{algos[0]}-baseline.csv', delimiter=',')
us_bpr = pd.read_csv(f'res\\{ds}\\{algos[1]}-baseline.csv', delimiter=',')
camf_ics = pd.read_csv(f'res\\{ds}\\{algos[2]}-baseline.csv', delimiter=',')

colors = ['#a8fa23', '#548701', '#53dee0', '#069496', '#7eacf2', '#154b9e', '#9a71f5', '#c10101']

for metric in metrics:
    for m in metrics_sizes:
        fig = plt.figure(figsize=(12.0, 6.0))
        dct = {'lambda': [0, 1], 'bpr': [bpr[f'{metric}_{m}'], bpr[f'{metric}_{m}']],
               'camfics': [camf_ics[f'{metric}_{m}'], camf_ics[f'{metric}_{m}']],
               'us_bpr': [us_bpr[f'{metric}_{m}'], us_bpr[f'{metric}_{m}']]}

        df = pd.DataFrame(dct)
        plt.plot('lambda', 'bpr', data=df, c=colors[0])
        plt.plot('lambda', 'us_bpr', data=df, c=colors[2])
        plt.plot('lambda', 'camfics', data=df, c=colors[4])

        plt.title(f'Baseline performance evaluating {metric}_{m} for the {ds} dataset')
        plt.ylabel('Lambda')
        plt.ylabel(f'{metric} score')
        fontp = FontProperties()
        fontp.set_size('small')
        plt.legend(loc=8, prop=fontp)
        plt.grid(zorder=0)
        plt.plot()

        plt.savefig(f'res\\{ds}\\viz\\{metric}_{m}', dpi=fig.dpi)

