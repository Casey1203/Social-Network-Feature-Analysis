# Social-Network-Feature-Analysis
This project aims to find the most importance features towards the detection of Microblog's credibility on Sina Weibo, the largest social network platform in China.

## Dataset
DataSet:
4815 samples(2473 true news & 2342 rumors)
The dataset is prepared along with the extracted features in "dataset.csv". In this project, we use 22 features.


## Package
In this project, you need to install several open-source third-party package we list here.

1. Pandas. https://github.com/pydata/pandas
2. Numpy. https://github.com/numpy/numpy
3. Scipy. https://github.com/scipy/scipy
4. sklearn https://github.com/scikit-learn/scikit-learn
5. Matplotlib https://github.com/matplotlib/matplotlib
6. plotly [opt]https://github.com/plotly

In this project, we also use [Weka](https://weka.wikispaces.com/Subversion)
## Detailed description
We use two different models to evaluate the importance of features, which is logistic regression and random forest. All the method are written in the single file. 


Usage:

Open terminal, locate the current address to where the new_all.py file is.
Then enter "python new_all.py -f function_name"(without quotation mark). The parameter "function_name" is the name of one function used in the .py file. For example, "python new_all.py -f univChi2" will calculate chi-square value and p-value of one feature with the class feature.

