import pandas as pd
from sklearn import feature_extraction, svm
import numpy as np
from optparse import OptionParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from scipy.stats import pearsonr
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.metrics import r2_score
from collections import defaultdict
py.sign_in('Casey1203','yco27825pi')


USAGE = "usage:    python new_all.py -f [function_name]"
parser = OptionParser(USAGE)
parser.add_option("-f", dest="function")

opt, args = parser.parse_args()

'''
For nominal feature with several catagories, data_encode function transform those catagories into several individual features.
For example, feature verified_type is a nominal feature with "celebrity, normal user, media..." catagories. Thus, several new features
like verified_type=celebrity or verified_type=normal user are created. The original feature verified_type is replaced. 
'''
def data_encode(data, feature_name, replace=False):
	vec = feature_extraction.DictVectorizer()
	multi_dict = []
	for i in range(len(data)):
		multi_dict.append({feature_name: str(np.asarray(data[feature_name])[i])})
	vecData = pd.DataFrame(vec.fit_transform(multi_dict).toarray())
	vecData.columns = vec.get_feature_names()
	vecData.index = data.index
	if replace is True:
		data = data.drop(feature_name, axis=1)
		data = data.join(vecData)
	return data

'''
Load dataset. If one algorithm can be used when inputs the nominal type features, like decision tree, then the data_encode on
"verified_type" and "province" doesn't need. Thus, 51 & 52 lines can be commented. This function return X, Y and feature list.
'''

def loadData():
	data = pd.read_csv('./dataset.csv')
	
	# data = data_encode(data, 'verified_type', replace=True)
	# data = data_encode(data, 'province', replace=True)
	data = data_encode(data, 'verified_kind', replace=True)
	Y = data['is_rumor'].tolist()

	feature_list = data.columns.tolist()
	feature_list.remove('is_rumor')
	feature_list.remove('province')
	feature_list.remove('verified_type')

	
	X = data[feature_list]

	return np.asarray(X.values), Y, feature_list

'''
The subset of dataset is selected by CFS feature selection for removing the inner-correlation between features.
This process is necessary if we want to use logistic regression to train models and use it to represent the feature importance.
'''
def loadDataSubset():
	data = pd.read_csv('./dataset.csv') # the path of dataset.

	Y = data['is_rumor'].tolist()
	feature_list = ['followers_count', 'bi_followers_count', 'share_ct', 'emotion', 'imgwd', 'engwd', 'verified_kind']
	X = data[feature_list]
	X = data_encode(X, 'verified_kind', replace=True)
	feature_list = X.columns
	return np.asarray(X.values), Y, feature_list


def drawScatter():
	data = pd.read_csv('./dataset.csv')
	X = data['statuses_count'].values
	Y = data['friends_count'].values
	fit = np.polyfit(X, Y, 1)
	plt.plot(X, fit[0] * X + fit[1], '-', color='red')
	plt.xlabel("statuses_count")
	plt.ylabel("friends_count")
	plt.title("Feature-Feature inter-correlation")
	pearson_coef = pearsonr(X, Y)
	plt.text(5000, 2500, "pearson coefficient: %f" %pearson_coef[0])
	plt.plot(X, Y, '.')
	plt.show()

'''
calculate the feature-feature correlation. Only numerical data can be used.
'''
def pearson(a,b):
	data = pd.read_csv('./dataset.csv')
	Y = data['is_rumor'].tolist()
	feature_list = data.columns.tolist()
	# we need to remove the class and not numerical feature
	feature_list.remove('is_rumor')
	feature_list.remove('verified_type')
	feature_list.remove('verified_kind')
	feature_list.remove('province')
	# print feature_list
	X = data[feature_list]
	m, n = np.shape(X)

	return pearsonr(X[a], X[b])
'''
Linear kernel of SVM classifier. It is similar to L2-norm logistic regression. Both of them
can reduce the impact of overfitting. Linear kernel of SVM is count on large margin while
L2-norm logistic regression limits the length of inner product of feature vector.
'''
def linearSVC():
	X, Y, names = loadDataSubset()
	X = X.astype(float)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	scaler_test = preprocessing.StandardScaler().fit(X_test)
	X_test = scaler_test.transform(X_test)

	clf = svm.SVC(kernel="linear", degree=1, gamma=0.1, coef0=1)
	clf = clf.fit(X_train, Y_train)
	print clf.score(X_test, Y_test)
	print clf.support_
	featureRanking(abs(clf.coef_[0]), names)


def meanDecreaseAccuracyOnWeibo():
	X, y, names = loadData()
	X = [dict(enumerate(sample)) for sample in X]
	vect = feature_extraction.DictVectorizer(sparse=False)
	X = vect.fit_transform(X)

	rf = RandomForestClassifier(n_estimators=500)
	scores = defaultdict(list)
	iter = 0
	while(iter <= 100):
		iter += 1
		X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
		rf = rf.fit(X_train, Y_train)
		acc = r2_score(Y_test, rf.predict(X_test))
		for i in range(X.shape[1]):
			X_t = X_test.copy()
			np.random.shuffle(X_t[:, i])
			shuff_acc = r2_score(Y_test, rf.predict(X_t))
			scores[names[i]].append((acc-shuff_acc)/acc)

	print "Features sorted by their score:"
	result = sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True)
	importance = []
	featurename = []
	for score, name in result:
		importance.append(score)
		featurename.append(name)
	featureRanking(importance, featurename)


'''
This method trains a model using random forest. During this procedure, each feature is likely to be chosen to split the dataset
into two so that similar values can be splited into the same set as many as possible (largest impurity decrease).
The importance of features is calculated by the average of gini impurity decrease among all trees in the forest.
'''

def meanDecreaseImpurityOnWeibo():
	X, Y, names = loadData()
	X = [dict(enumerate(sample)) for sample in X]
	vect = feature_extraction.DictVectorizer(sparse=False)
	X = vect.fit_transform(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	rf = RandomForestClassifier(n_estimators=500)
	rf.fit(X_train, Y_train)
	print "Features sorted by their score:"
	print sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True)
	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_train,rf.predict(X_train)))
	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test,rf.predict(X_test)))
	print "Classification report"
	print metrics.classification_report(Y_test, rf.predict(X_test))
	print "Confusion matrix"
	print metrics.confusion_matrix(Y_test, rf.predict(X_test))
	print rf.feature_importances_
	featureRanking(rf.feature_importances_, names)

	return rf.score(X_test, Y_test)

'''
This method is based on the idea that when all features are on the same scale, 
the most important features should have the highest coefficient in the model.
While uncorrelated features should have coefficient close to 0
The parameter penalty = l1 or l2 prevents overfitting and improve generalization. The cost function is E(X,Y)+alpha*||w||
where w is the vector of model coefficient. alpha can be adjusted, which means that gives how much penalty on the coefficient.
BTW, in this method, the feature verified_type and province is remove because nominal type feature cannot be used directly in
logistic regression algorithm unless they are encoded.
'''
def logisticR():
	X, Y, names = loadData()
	X = X.astype(float)
	# split the dataset by 7:3
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	# standardization
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)

	scaler_test = preprocessing.StandardScaler().fit(X_test)
	X_test = scaler_test.transform(X_test)

	lr = LogisticRegression(penalty='l2', C=1.5)# or 'l2'
	lr.fit(X_train, Y_train)
	# lr.intercept_: x0, lr.coef_:x1~xn
	# model evaluation
	print lr.score(X_test, Y_test)
	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test, lr.predict(X_test)))
	print "Classification report"
	print metrics.classification_report(Y_test, lr.predict(X_test))
	print "Confusion matrix"
	print metrics.confusion_matrix(Y_test, lr.predict(X_test))
	# feature ranking by lr coefficient
	featureRanking(abs(lr.coef_[0]), names)
	return lr.score(X_test, Y_test)

'''
This method will rank the importance and draw a histogram by their score.
'''
def featureRanking(importance, names):
	indices = np.argsort(importance)[::-1]
	print names
	bar_name = []
	sorted_importance = []
	for i in indices:
		bar_name.append(names[i])
		sorted_importance.append(importance[i])
	data = Data([Bar(x=bar_name, y=sorted_importance)])
	plot_url = py.plot(data, filename='basic-bar')
	print "Feature ranking:"
	for f in range(len(importance)):
		print("%d. feature %s (%f)" % (f + 1, names[indices[f]], importance[indices[f]]))
	# Plot the feature importances of the forest
	plt.figure()
	plt.title("Feature importances")
	plt.bar(range(len(importance)), importance[indices],
	       color="r", yerr=importance[indices], align="center")
	plt.xticks(range(len(importance)), bar_name)
	plt.xlim([-1, len(importance)])
	# plt.show()


if opt.function == 'loadData':
	loadData()
elif opt.function == 'meanDecreaseImpurityOnWeibo':
	meanDecreaseImpurityOnWeibo()
elif opt.function == 'logisticR':
	logisticR()
elif opt.function == 'pearson':
	pearson()
elif opt.function == 'drawScatter':
	drawScatter()
elif opt.function == 'linearSVC':
	linearSVC()
elif opt.function == 'meanDecreaseAccuracyOnWeibo':
	meanDecreaseAccuracyOnWeibo()