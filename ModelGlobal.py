import pandas as pd

# Uses the global audio feature averages for each contextual value as model for re-ranking
def global_model(fold, d_set, songs, personal_models, num, invalid_songs):
    AUDIO_FEATURES = ["acousticness", "danceability", "energy", "instrumentalness", "loudness",
                      "speechiness", "tempo", "valence"]

    test_set = pd.read_csv(f'input\\{d_set}_test\\{num}-test.csv', delimiter=',')

    if 'nprs' in d_set:
        relevant_contexts = ['morning', 'afternoon', 'evening', 'night']
    else:
        relevant_contexts = ['sunny', 'snowing', 'rainy', 'cloudy', 'traffic jam', 'lots of cars', 'sleepy', 'awake',
                             'city', 'night', 'morning', 'sad', 'lazy', 'active', 'urban', 'mountains', 'country side',
                             'coast line', 'relaxed driving', 'free road', 'serpentine', 'highway', 'day time',
                             'afternoon', 'happy', 'sport driving']

    global_avgs = {}
    for index, row in test_set.iterrows():
        if row['Item'] not in invalid_songs:
            song_afs = songs.loc[songs['id'] == row['Item']]
            for af in AUDIO_FEATURES:
                personal_models[row['User']][row['Context'].split(':')[1]][af] = \
                    personal_models[row['User']][row['Context'].split(':')[1]][af] - song_afs[af].item()
            personal_models[row['User']][row['Context'].split(':')[1]]['Amount'] = \
                personal_models[row['User']][row['Context'].split(':')[1]]['Amount'] - 1

    for cntx in relevant_contexts:
        amount = 0
        total = {"acousticness": 0, "danceability": 0, "energy": 0, "instrumentalness": 0, "loudness": 0,
                 "speechiness": 0, "tempo": 0, "valence": 0}
        for user in personal_models:
            if personal_models[user][cntx]['Amount'] > 0:
                for af in AUDIO_FEATURES:
                    total[af] = total[af] + personal_models[user][cntx][af]
                amount = amount + personal_models[user][cntx]['Amount']

        for af in AUDIO_FEATURES:
            total[af] = total[af] / amount
        global_avgs[cntx] = total

    return global_avgs
