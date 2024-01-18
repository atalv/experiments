## This folder is for experiments done for Link Prediction/ Community Detection in Graph & Network Analysis

### LastFM User Analysis

* **[LastFM][snap1] user analysis** done during Spring'23 and the data is sourced from SNAP (Stanford).  

* No need to download the datasets from SNAP, the input files are already accessed from the direct link to the dataset as hosted on SNAP.  

* Use Jupyter Notebook (prefereably on Google Colab) to run the analysis. 
    * If intend to use Google Colab then [authorize Colab to talk to your GitHub account][colab].

#### Notes
1. The LastFM dataset is relatively simpler, it is an undirected graph with a twist of node classification.
2. This [NetworkX example notebook][nxnb1] is a good reference for finding graph data attributes.
3. Note that there are many different graph embeddings that can be defined, and in many ways it can be used to do link prediction. 
4. Initially, graph attributes are extracted to understand the high-level patterns present in the data.
5. One of the **highlights** is, here I implemented multiple node embedding approaches for link prediction - *[Graph Factorization](https://doi.org/10.1145/2488388.2488393), [DeepWalk](https://arxiv.org/pdf/1403.6652.pdf), [Node2Vec](https://arxiv.org/pdf/1607.00653.pdf), [Adamic-Adar index](http://www.cs.cornell.edu/home/kleinber/link-pred.pdf)* - and compared their performance for link prediction task. 

[snap1]: https://snap.stanford.edu/data/feather-lastfm-social.html
[nxnb1]: https://networkx.org/nx-guides/content/exploratory_notebooks/facebook_notebook.html
[colab]: https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb
