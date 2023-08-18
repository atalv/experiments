## This folder is for experiments done on Time Series data

### Grocery Store Sales

* **[Grocery Store Sales][kaggle] forecast** done during Spring'23 and the data is sourced from Kaggle.  

* `train` and `test` datasets can be downloaded directly from the Kaggle link above. They should be saved in the project repository's root, so that they can be read via R and Python notebooks directly. Those files will not be tracked via git.  

* [Kaggle implementation done by me](https://www.kaggle.com/code/vivekatal/store-sales-forecasting-r-markdown).

* **Check the `grocery-store-sales-ARIMA.md`** (*.md version*) to see the final output of the R Markdown script (`.Rmd`) **nicely rendered on GitHub**.

#### Codes:
- *`grocery-store-sales-ARIMA.Rmd`*: This includes all the R codes for this problem where EDA, data pre-processing, feature creation, and model build are present.
- *`grocery-store-sales-DeepAR-aws.ipynb`*: This includes an experimental implementation of the same problem using [DeepAR](https://arxiv.org/pdf/1704.04110.pdf) on AWS Sagemaker. Hyperparameter tuning is not done; that should have been done using time series cross-validation.  
    - **`grocery-store-sales-DeepAR-aws-TrimmedOutput.ipynb`**: This is the light weight version of above notebook so that it can be rendered easily on GitHub. Some verbose outputs are truncated, no other changes.

#### Notes:
1. Initial idea is to create some descriptive of the data, e.g., how many stores are there, how many products are there, distribution of sales by store, distribution of sales by product, etc.
2. Create some high level time series plots for selected products' sales, note the irregularities observed in the data - impact of earthquake, promotion, etc.
3. Analyze one of the most sold product-store combination and tune SARIMA model manually using the time series plots, model diagnostics and time-series specific cross-validation.
4. Some of the **highlights** are:
    - The events data had ~100 unique events, however, all of them are not useful in all stores/products. A *simple algorithmic approach is defined to identify some of most impactful events separately for each of the store-product combination*. Only those ones are tested while model building.
    - *Cross-store* and *cross-product* information is used while modeling. E.g., if a somewhat related product has more promotion in a week then that may impact sales of a given product. Similarly, if a nearby store has significantly high promotions for a product then the sales may be lower of that product in a given store.
    - ARIMA modeling approach needs tuning for each individual time series, and there are ~1,800 time series in this dataset to be modeled. Experimented with various approaches and finally used *automated version of `stl` decomposition where season window is tuned algorithmically, and a non-seasonal ARIMA model is fit using `auto.arima`*. Objective is to speed up the model fitting process without significant loss in model quality.  


[kaggle]: https://www.kaggle.com/competitions/store-sales-time-series-forecasting
