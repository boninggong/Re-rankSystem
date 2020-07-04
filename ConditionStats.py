from statistics import mean
import pandas as pd

user_context_ratings = pd.read_csv(f'input\\nprs\\user_context_sums.csv', delimiter=',', keep_default_na=False)
conditions = ['morning', 'afternoon', 'evening', 'night']
sums = {}

for c in conditions:
    sums[c] = []

for index, row in user_context_ratings.iterrows():
    sums[row['Context']].append(row['Amount'])

print(sums)

for c in sums:
    print(f'{c} {mean(sums[c])}')
