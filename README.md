# Social-Network-Feature-Analysis
This project aims to find the most importance features towards the detection of Microblog's credibility on Sina Weibo, the largest social network platform in China.

## Dataset
DataSet:
4815 samples(2473 true news & 2342 rumors)
The dataset is prepared along with the extracted features in "dataset.csv". In this project, we use 22 features.
The original dataset is provided by thesis
[False Rumors Detection on Sina Weibo by Propagation Structures](http://www.cs.sjtu.edu.cn/~kzhu/papers/kzhu-rumor.pdf)


## Package
In this project, you need to install Python 2.7. Besides, you need to install several open-source third-party packages we list here.

1. Pandas. https://github.com/pydata/pandas
2. Numpy. https://github.com/numpy/numpy
3. Scipy. https://github.com/scipy/scipy
4. sklearn https://github.com/scikit-learn/scikit-learn
5. Matplotlib https://github.com/matplotlib/matplotlib
6. plotly [opt]https://github.com/plotly

In this project, we also use [Weka](https://weka.wikispaces.com/Subversion). U can download the GUI version of Weka [here](http://www.cs.waikato.ac.nz/ml/weka/downloading.html).

## Detailed description
We use two different models to evaluate the importance of features, which is logistic regression and random forest. All the method are written in the single file. 

Usage:

Open terminal, locate the current address to where the featEva.py file is.
Then enter 
<pre>python featEva.py -f function_name</pre>
 The parameter "function_name" is the name of one function used in the .py file. For example, <pre>python featEva.py -f logisticR</pre> 
 it will round the logistic regression to train model and return the performance of it along with the feature importance rank. Same as <pre>python featEva.py -f meanDecreaseImpurityOnWeibo</pre> that will run the random forest algorithm to train model and use Gini-index to evaluate the importance of features.


