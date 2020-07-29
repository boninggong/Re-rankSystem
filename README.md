# Re-rankSystem

### Introduction
-------------------
This system re-ranks a given song recommendation list by calculating new scores for each song. These new scores are based on the initial recommendation score and the similarity between the audio features of the song and the given context in which a user is consuming it. The re-rank algorithm uses two models, a global and personalized one, which represent user preferences in certain contextual conditions. Initial results, tested on the #NowPlaying-RS dataset, show that the personalized model outperforms the global model and that such a re-ranking approach improves recommendation accuracy (measured using precision at k and mean average precision at k)

### Execution Pipeline
----------------------------

![Execution Pipeline](https://raw.githubusercontent.com/boninggong/Re-rankSystem/master/pipeline.png)

**Input data processing:** https://github.com/boninggong/DataPreProcess
**Initial recommender system:** https://github.com/boninggong/CARSKitModified
**Re-rank system & Accuracy evaluation:** this repository

### Running the system
-------------------
1. Pre-process data by executing the scripts within https://github.com/boninggong/DataPreProcess.
   * Move _nowplaying-rs-final.CSV_ to _\out\artifacts\CARSKitjar\data\\_ in https://github.com/boninggong/CARSKitModified. 
   * Move _nprs_ratings.CSV_, _nprs_audio_features.CSV_, _user_context_items.CSV_ and _user_context_sums.CSV_ to _\output\nprs\\_ in this repository.
2. Run https://github.com/boninggong/CARSKitModified as described by the original CARSKit guide: http://arxiv.org/abs/1511.03780. This will create the initial recommendations that will be used as input to the re-rank system.
   * Move the _k_ recommendation file outputs (e.g. _BPR-1.CSV_ to _BPR-5.CSV_ for _k_ = 5) within _\out\artifacts\CARSKitjar\data\CARSKit.Workspace\\_ to _\input\nprs_200\\_ in this repository.
   * Move the _k_ test set files (_1-test.csv_ to _k-test.csv_) to _\input\nprs_test\\_.
3. You are now ready to run the scripts within this repository to re-rank the given song recommendations and compare accuracy results. The scripts themselves contain extra instructions where necessary.

NOTE: Only the #NowPlaying-RS dataset is supported in the full execution. The InCarMusic dataset was supported, but has been dropped because of its scarcity. There are still some of the initial results of the InCarMusic dataset in the results folder of previous runs. 
