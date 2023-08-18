# Overview

This repo is to store the experiments done for hands-on learning with *dummy data*. It is never too late to learn and *showcase*!

- Each sub-directory in the root is named as the main topic of the experiment.
- All the contents are created solely by me with guidance from official resources and academic experts.

## Some highlighs

- **GraphNetwork**: 
    - Predict whether a user of LastFM would follow another user and serve as a recommendation.
    - Implemented multiple node embedding approaches for link prediction - *[Graph Factorization](https://static.googleusercontent.com/media/research.google.com/en/pubs/archive/40839.pdf), [DeepWalk](https://arxiv.org/pdf/1403.6652.pdf), [Node2Vec](https://arxiv.org/pdf/1607.00653.pdf), [Adamic-Adar index](http://www.cs.cornell.edu/home/kleinber/link-pred.pdf)* - and compared their performance for link prediction task.

- **MachineLearning**: 
    - Predict NYC taxi trip duration.
    - Implemented typical machine learning models from *[scikit-learn](https://scikit-learn.org/stable/)* ([GammaRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.GammaRegressor.html), [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html), [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)) with intelligently derived features, viz., traffic information in an area at a given time window based on average active number of trips originating or ending.

- **ReinforcementLearning**: 
    - Simulated multiple UCB (Upper Confidence Bound) policies for MAB (Multi Armed Bandit) problems and compared their performance.
    - Learned to do simulation of multiple states Markov Chain and calculate average reward, expected present value, estimate steady state probabilities, etc.
    - Most of the research papers referred for simulation exercises are authored by [Dr. Michael Katehakis](http://en.wikipedia.org/wiki/Michael_N._Katehakis).

- **TimeSeries**: 
    - Forecasted 2 weeks ahead grocery store sales of 33 product groups across 54 stores, approx. 1.8K time series.
    - Engineered multiple sensible features, viz., cross-store, cross-product elements, algorithmically short-listed important events for a given store-product, etc.
    - Some Seasonal ARIMA models were built manually, and then scaled it using ARIMA where seasonal components were extracted beforehand for faster execution.
    - Experimented with DeepAR on AWS Sagemaker to build a single global model instead of 1.8K ARIMA models.
