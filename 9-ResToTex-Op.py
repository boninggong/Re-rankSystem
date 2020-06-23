import pandas as pd

# Converts all opposite re-rank and initial evaluation results to LaTeX tables
test_items = [25, 50, 100, 200]
ds = "nprs"
metrics = ["Prec", "MAP"]
metrics_sizes = [10, 25, 'all']
algos = ["BPR", "UserSplitting-BPR", "CAMF_ICS"]
dist = 'euclidean'
path = f"res\\{ds}\\"
dimensions = ['daytime']

for dimension in dimensions:
    print(f'\\FloatBarrier\n{{\\Large\\textbf{{{dimension} dimension based opposite re-rankings}}}}')
    for algo in algos:
        algo_d = algo
        if algo == 'CAMF_ICS':
            algo_d = "CAMF-ICS"
        for m in metrics_sizes:
            for t in test_items:
                if m != 'all':
                    algo_glob_p = pd.read_csv(f"{path}{t}\\global-{algo}-{metrics[0]}-{dimension}-op.csv", delimiter=',')
                    algo_glob_map = pd.read_csv(f"{path}{t}\\global-{algo}-{metrics[1]}-{dimension}-op.csv", delimiter=',')
                    algo_pers_p = pd.read_csv(f"{path}{t}\\personal-{algo}-{metrics[0]}-{dimension}-op.csv", delimiter=',')
                    algo_pers_map = pd.read_csv(f"{path}{t}\\personal-{algo}-{metrics[1]}-{dimension}-op.csv", delimiter=',')

                    output = f"\\begin{{table}}[ht]\n\\captionsetup{{justification=centering}} \\scalebox{{0.85}}{{" \
                             f"\\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|}}\n\\hline\nList size\t& \\" \
                             f"multicolumn{{10}}{{c|}}{{{t}}} \\\\ \\hline\n\\textbf{{}}\t& \\multicolumn{{2}}{{c|}}{{$\\" \
                             f"lambda$ = 0.2}}\t& \\multicolumn{{2}}{{c|}}{{$\\lambda$ = 0.4}}\t& \\multicolumn{{2}}{{c|}}{{" \
                             f"$\\lambda$ = 0.6}}\t& \\multicolumn{{2}}{{c|}}{{$\\lambda$ = 0.8}}\t& \\multicolumn{{2}}{{c|}}" \
                             f"{{$\\lambda$ = 1.0}} \\\\ \\hline\n\\textbf{{}}\t& P@{m}\t& MAP@{m}\t& P@{m}\t& MAP@{m}\t& P@{m}\t&" \
                             f" MAP@{m}\t& P@{m}\t& MAP@{m}\t& P@{m}\t& MAP@{m}\t \\\\ \\hline\n" \
                             f"Initial\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.2][f'Prec_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.4][f'Prec_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.6][f'Prec_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.8][f'Prec_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 1.0][f'Prec_{m}_initial'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}Re-ranked\\\\ Global\\end{{tabular}}\t" \
                             f"& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.2][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.4][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.6][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 0.8][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_p.loc[algo_glob_p['lambda'] == 1.0][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}Re-ranked\\\\ Personal\\end{{tabular}}\t" \
                             f"& {round(algo_pers_p.loc[algo_glob_p['lambda'] == 0.2][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_p.loc[algo_glob_p['lambda'] == 0.4][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_p.loc[algo_glob_p['lambda'] == 0.6][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_p.loc[algo_glob_p['lambda'] == 0.8][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_p.loc[algo_glob_p['lambda'] == 1.0][f'Prec_{m}_rerank'].item(), 5)}" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\end{{tabular}}}}\n\\caption*{{Opposite re-rank results for the \\textbf{{{algo_d}}} initial recommendation " \
                             f"for both global and personal model, evaluated over the \\textbf{{top {m}}} songs in " \
                             f"the recommendation list consisting of \\textbf{{{t} songs}} using the \\textbf{{{dimension}}} contextual dimension}}\\" \
                             f"label{{tab:res-{algo_d}-{t}-{m}-op}}\n\\end{{table}}"
                else:
                    algo_glob_map = pd.read_csv(f"{path}{t}\\global-{algo}-{metrics[1]}-{dimension}-op.csv",
                                                delimiter=',')
                    algo_pers_map = pd.read_csv(f"{path}{t}\\personal-{algo}-{metrics[1]}-{dimension}-op.csv",
                                                delimiter=',')

                    output = f"\\begin{{table}}[ht]\n\\captionsetup{{justification=centering}} \\scalebox{{0.85}}{{" \
                             f"\\begin{{tabular}}{{|l|l|l|l|l|l|l|l|l|l|l|}}\n\\hline\nList size\t& \\" \
                             f"multicolumn{{10}}{{c|}}{{{t}}} \\\\ \\hline\n\\textbf{{}}\t& \\multicolumn{{2}}{{c|}}{{$\\" \
                             f"lambda$ = 0.2}}\t& \\multicolumn{{2}}{{c|}}{{$\\lambda$ = 0.4}}\t& \\multicolumn{{2}}{{c|}}{{" \
                             f"$\\lambda$ = 0.6}}\t& \\multicolumn{{2}}{{c|}}{{$\\lambda$ = 0.8}}\t& \\multicolumn{{2}}{{c|}}" \
                             f"{{$\\lambda$ = 1.0}} \\\\ \\hline\n\\textbf{{}}\t& P@{m}\t& MAP@{m}\t& P@{m}\t& MAP@{m}\t& P@{m}\t&" \
                             f" MAP@{m}\t& P@{m}\t& MAP@{m}\t& P@{m}\t& MAP@{m}\t \\\\ \\hline\n" \
                             f"Initial\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_initial'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}Re-ranked\\\\ Global\\end{{tabular}}\t" \
                             f"& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_glob_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\begin{{tabular}}[c]{{@{{}}l@{{}}}}Re-ranked\\\\ Personal\\end{{tabular}}\t" \
                             f"& NA" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.2][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.4][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.6][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 0.8][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t& NA" \
                             f"\t& {round(algo_pers_map.loc[algo_glob_map['lambda'] == 1.0][f'MAP_{m}_rerank'].item(), 5)}" \
                             f"\t \\\\ \\hline\n" \
                             f"\\end{{tabular}}}}\n\\caption*{{Opposite re-rank results for the \\textbf{{{algo_d}}} initial recommendation " \
                             f"for both global and personal model, evaluated over the \\textbf{{top {m}}} songs in " \
                             f"the recommendation list consisting of \\textbf{{{t} songs}} using the \\textbf{{{dimension}}} contextual dimension}}\\" \
                             f"label{{tab:res-{algo_d}-{t}-{m}-op}}\n\\end{{table}}"

                print(output)
                print("\n")
