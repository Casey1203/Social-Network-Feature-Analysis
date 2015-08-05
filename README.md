# Social-Network-Feature-Analysis
This project aims to find the most importance features towards the detection of Microblog's credibility on Sina Weibo, the largest social network platform in China.

The dataset is prepared along with the extracted features in "dataset.csv". In this project, we use 22 features.

We use two different model to evaluate the importance of features, which is logistic regression and random forest. All the method are written in the single file.

DataSet:
4815 samples(2473 true news & 2342 rumors)

The descriptions of functions in new_all.py are written in that file.py

Usage:

Open terminal, locate the current address to where the new_all.py file is.
Then enter "python new_all.py -f function_name"(without quotation mark). The parameter "function_name" is the name of one function used in the .py file. For example, "python new_all.py -f univChi2" will calculate chi-square value and p-value of one feature with the class feature.

