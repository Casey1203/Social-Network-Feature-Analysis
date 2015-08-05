import sklearn as sk
import pandas as pd
from sklearn import feature_extraction, tree, svm
import numpy as np
from optparse import OptionParser
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import RandomizedLasso, RandomizedLogisticRegression, LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.utils import atleast2d_or_csr, array2d
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from scipy import special, stats
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from sklearn.feature_selection import RFE
from sklearn.lda import LDA
from sklearn.decomposition import PCA
from sklearn.feature_selection import *
from sklearn.pipeline import Pipeline
from sklearn.covariance.outlier_detection import OutlierDetectionMixin, EllipticEnvelope
from scipy.stats import pearsonr
import plotly.plotly as py
from plotly.graph_objs import *
from sklearn.cross_validation import ShuffleSplit
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
	#print type(vec.fit_transform(multi_dict).toarray())
	#print vec.get_feature_names()
	vecData = pd.DataFrame(vec.fit_transform(multi_dict).toarray())
	vecData.columns = vec.get_feature_names()
	vecData.index = data.index
	if replace is True:
		data = data.drop(feature_name, axis=1)
		data = data.join(vecData)
	return data

'''
Load dataset. If one algorithm can be used when inputs the nominal type features, like decision tree, then the data_encode on
"verified_type" and "province" doesn't need. Thus, 64 & 65 lines can be commented. This function return X, Y and feature list.
'''


def loadData():
	data = pd.read_csv('/Users/Casey/Desktop/data07071.csv')
	
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


def drawScatter():
	data = pd.read_csv('/Users/Casey/Desktop/data07071.csv')
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


	# trace0 = Scatter(
	# 	x=X,
	# 	y=Y,
	# 	mode='markers'
	# )
	# data = Data([trace0])
	# plot_url = py.plot(data, filename='line-scatter')


def pearson(a,b):
	data = pd.read_csv('/Users/Casey/Desktop/data07071.csv')
	Y = data['is_rumor'].tolist()
	feature_list = data.columns.tolist()
	feature_list.remove('is_rumor')
	feature_list.remove('verified_type')
	feature_list.remove('province')
	# print feature_list
	X = data[feature_list]
	m, n = np.shape(X)

	return pearsonr(X[a], X[b])



'''
VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance
doesn't meet the threshold. By default, it removes all zero-variance features, i.e. features that have the same
value in all samples.
Function varianceFS is used in the boolean feature(0/1). We set the threshold as 0.85, meaning that we remove
feature that either 0 or 1 is more than 85%
'''
def varianceFS():
	data = pd.read_csv('/Users/Casey/Desktop/new.csv')
	
	#data = data_encode(data, 'verified_type', replace=True)
	#data = data_encode(data, 'province', replace=True)
	Y = data['is_rumor'].tolist()

	feature_list = ['verified_type', 'province']
	X = data[feature_list]
	X = data_encode(X, 'verified_type', replace=True)
	X = data_encode(X, 'province', replace=True)
	print 'before transform'
	print np.shape(X)
	names = X.columns
	#print np.shape(X)
	p=0.85
	threshold = (p*(1-p))
	sel = VarianceThreshold(threshold=threshold)
	
	print "after transform" 
	print np.shape(sel.fit_transform(X))
	index = []
	result = sel.variances_> threshold
	for i in range(len(result)):
		if result[i] == True:
			index.append(i)
	print sel.variances_[index]
	select = []
	for i in index:
		select.append(names[i])
	print select


'''
return contingencyTable in chi-squared test
'''
def count():
	X, Y, names = loadData()
	thatisFalse = 0
	thatisTrue = 0
	thatisFalselist = []
	thatisTruelist = []
	for j in range(35):
		for i in range(len(X)):
			if int(X[i][j+26])==1 and Y[i]==False:#26 is province, 14 is verified_type
				thatisFalse +=1
			elif int(X[i][j+26])==1 and Y[i]==True:
				thatisTrue +=1
		thatisFalselist.append(thatisFalse)
		thatisTruelist.append(thatisTrue)
		thatisFalse = 0
		thatisTrue = 0
	contingencyTable = []
	contingencyTable.append(thatisFalselist)
	contingencyTable.append(thatisTruelist)
	return contingencyTable


'''
calculate the chi2 value of nominal feature.
'''
def catagoryChi2():
	contingencyTable = count()
	print 'observed'
	print np.asarray(contingencyTable)
	columnnum = len(contingencyTable[0])
	sumFalse = sum(contingencyTable[0])
	sumTrue = sum(contingencyTable[1])
	total = sumFalse + sumTrue
	sumColumn = []
	for column in range(columnnum):
		sumColumn.append(sum(row[column] for row in contingencyTable))
	expectedConTable = []
	expectedConTable.append([round(1.0 * col * sumFalse/total, 2) for col in sumColumn])
	expectedConTable.append([round(1.0 * col * sumTrue/total, 2) for col in sumColumn])
	print 'expected'
	print expectedConTable
	print _chisquare(contingencyTable, expectedConTable)
	'''
	chi = 0
	for i in range(len(contingencyTable)):
		for j in range(len(contingencyTable[i])):
			chi += (contingencyTable[i][j]-expectedConTable[i][j]) ** 2 / expectedConTable[i][j]
	print chi
	'''
	
'''
calculate the chi2 value and p-value of one feature and the class feature.
user can change the name of feature in line 157.
'''
def univChi2():
	X, Y, names = loadData()
	print names
	isandF = 0
	isandT = 0
	notandF= 0
	notandT= 0
	index = names.index('verified_type=gov')
	for i in range(len(X)):
		if int(X[i][index])==1 and Y[i]==False:
			isandF += 1
		elif int(X[i][index])==1 and Y[i]==True:
			isandT += 1
		elif int(X[i][index])==0 and Y[i]==False:
			notandF += 1
		elif int(X[i][index])==0 and Y[i]==True:
			notandT += 1
	print isandF, isandT, notandF, notandT
	#print np.shape(isVerified), np.shape(np.asarray(np.mat(X)[:,8]))
	print calculatechi2(np.mat(X)[:,index], Y)
	'''
	for i in range(len(names)-14):
		print names[i+14]
		print calculatechi2(np.mat(X)[:,i+14], Y)
		print '\n'
	'''	
	#print chi2(np.asarray(np.mat(X)[:,8]), Y)
	#print SelectKBest(chi2, k=10).fit(X,Y).scores_
	
def calculatechi2(X,Y):
    X = atleast2d_or_csr(X)
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(Y)
    if X.shape[1] == 1:
    	X = np.append(1 - X, X, axis=1)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(X.T, Y)          # n_classes * n_features

    feature_count = array2d(X.sum(axis=0))
    class_prob = array2d(Y.mean(axis=0))
    expected = np.dot(feature_count.T, class_prob)
    print 'observed frequency'
    print observed
    print 'expected frequency'
    print expected
    return _chisquare(observed, expected)

def _chisquare(f_obs, f_exp):
    """Fast replacement for scipy.stats.chisquare.

    Version from https://github.com/scipy/scipy/pull/2525 with additional
    optimizations.
    """
    f_obs = np.asarray(f_obs, dtype=np.float64)

    k = len(f_obs)
    # Reuse f_obs for chi-squared statistics
    chisq = f_obs
    chisq -= f_exp
    chisq **= 2
    chisq /= f_exp
    chisq = chisq.sum()
    return chisq, special.chdtrc(k - 1, chisq)


def DTvisualization():
	X, Y, names = loadDataLogistic()
	X = X.astype(float)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
	clf = tree.DecisionTreeClassifier(max_depth=3)
	clf = clf.fit(X_train, Y_train)
	print clf.score(X_test, Y_test)
	tree.export_graphviz(clf, out_file='tree.dot', feature_names=names)

def gradientBoosting():
	X, Y, names = loadDataLogistic()
	X = X.astype(float)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=3).fit(X_train, Y_train)

	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_train, clf.predict(X_train)))
	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test, clf.predict(X_test)))
	print "Classification report"
	print metrics.classification_report(Y_test, clf.predict(X_test))
	print "Confusion matrix"
	print metrics.confusion_matrix(Y_test, clf.predict(X_test))
	featureRanking(clf.feature_importances_, names)

def linearSVC():
	X, Y, names = loadDataLogistic()
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
	#print np.shape(X), np.shape(y)
	rf = RandomForestClassifier(n_estimators=500)
	scores = defaultdict(list)

	#crossvalidate the scores on a number of different random splits of the data
	# for train_idx, test_idx in ShuffleSplit(len(X), n_iter=100, test_size=.3):# 0.3 proportion to be in testset
	iter = 0
	while(iter <= 100):
		iter += 1
		# X_train, X_test = X[train_idx], X[test_idx]
	    # Y_train, Y_test = y[train_idx], y[test_idx]
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

	print names
	#X = X.astype(float)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	# print X_train[0], Y_train[0]
	rf = RandomForestClassifier(n_estimators=500)
	rf.fit(X_train, Y_train)
	# print rf.feature_importances_
	# importance = np.append(rf.feature_importances_[0:6], sum(rf.feature_importances_[6:10]))
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



def loadDataLogistic():
	# data = pd.read_csv('/Users/Casey/Desktop/new.csv')
	data = pd.read_csv('/Users/Casey/Desktop/data07071.csv')
	#data = data_encode(data, 'verified_type', replace=True)
	#data = data_encode(data, 'province', replace=True)
	Y = data['is_rumor'].tolist()

	feature_list = ['followers_count', 'bi_followers_count', 'share_ct', 'emotion', 'imgwd', 'engwd', 'verified_kind']
	X = data[feature_list]
	X = data_encode(X, 'verified_kind', replace=True)
	feature_list = X.columns
	# feature_list = ['followers_count', 'bi_followers_count', 'statuses_count', 'share_ct', 'emotion', 'verified_kind=normal user', 'verified_kind=master', 'verified_kind=yellowV', 'verified_kind=blueV', 'imgwd', 'engwd']
	# X = X[feature_list]
	#X = X[feature_list]
	return np.asarray(X.values), Y, feature_list


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

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	print X_test[0]
	scaler = preprocessing.StandardScaler().fit(X_train)

	X_train = scaler.transform(X_train)



	scaler_test = preprocessing.StandardScaler().fit(X_test)
	X_test = scaler_test.transform(X_test)
	print X_test[0]
	print Y_test[0]

	# scaler_pre = preprocessing.StandardScaler().fit(X_pre)
	# X_pre = scaler_pre.transform(X_pre)

	lr = LogisticRegression(penalty='l2', C=1.5)# or 'l2'
	lr.fit(X_train, Y_train)
	print lr.intercept_
	print lr.coef_# print coefficient
	# coef = abs(lr.coef_)<0.1
	# coef = coef[0]
	# for i in range(len(coef)):
	# 	if coef[i]==True:
	# 		lr.coef_[0,i] = 0
	print lr.score(X_test, Y_test)
	print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test, lr.predict(X_test)))
	print "Classification report"
	print metrics.classification_report(Y_test, lr.predict(X_test))
	print "Confusion matrix"
	print metrics.confusion_matrix(Y_test, lr.predict(X_test))
	featureRanking(abs(lr.coef_[0]), names)
	return lr.score(X_test, Y_test)

def findC():
	Carr = np.linspace(0.01,5,500)
	result = []
	ma = -1
	mC = 0
	for i in Carr:
		tmp = logisticR(i)
		result.append(tmp)
		if tmp>ma:
			ma = tmp
			mC = i
	trace = Scatter(x=Carr, y=result)
	data = Data([trace])
	plot_url = py.plot(data, filename='basic-line')
	print mC, ma

def findMaxdepth():
	maxArr = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	result = []
	ma=-1
	mdepth = 0
	for i in maxArr:
		tmp = meanDecreaseImpurityOnWeibo(i)
		result.append(tmp)
		if tmp > ma:
			ma=tmp
			mdepth = i
	trace = Scatter(x=maxArr, y=result)
	data = Data([trace])
	plot_url = py.plot(data, filemname='basic-line')
	print mdepth, ma

# def stabilitySelectionOnWeibo():
# 	X, Y, names = loadData()
# 	X = X.astype(float)
# 	X = preprocessing.scale(X)
# 	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
# 	rlr = RandomizedLogisticRegression()
# 	rlr.fit(X_train, Y_train)
# 	print "Features sorted by their score:"
# 	print sorted(zip(map(lambda x: round(x, 4), rlr.scores_), 
# 	                 names), reverse=True)
# 	featureRanking(rlr.scores_, names)

# 	print rlr.get_params(deep=True)
# 	#print "Accuracy:{0:.3f}".format(metrics.accuracy_score(Y_test,rlr.predict(X_test)))


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

# def svmRFE():#not work because slow
# 	X, Y, names = loadData()
# 	svc = SVC(kernel="linear", C=1)
# 	selector = RFE(estimator=svc, n_features_to_select=1, step=1)
# 	selector.fit(X, Y)
# 	print selector.ranking_

# def linearWithL1():
# 	X, Y, names = loadData()
# 	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=10)
	
# 	linearSVC = LinearSVC(C=0.01, penalty="l1", dual=False)
# 	linearSVC.fit(X_train, Y_train)
# 	print np.shape(linearSVC.fit_transform(X_train, Y_train))
# 	print linearSVC.score(X_test, Y_test)
# 	print linearSVC.coef_
	


# def drawhist():
# 	X, Y, names = loadData()
# 	plt.hist(np.asarray(X)[:,4:5], bins=50)
# 	plt.title(names[4] + " histogram")
# 	plt.xlabel("Value")
# 	plt.ylabel("Frequency")
# 	plt.show()


# def outlierDetection():
# 	data = pd.read_csv('/Users/Casey/Desktop/new.csv')
# 	feature_name = np.asarray(data.columns)
# 	d = data[feature_name[2]]
# 	data['isAnomaly'] = d > d.quantile(.99)
# 	isAnomalylist = data['isAnomaly'].values
# 	for i in range(len(isAnomalylist)):
# 		if isAnomalylist[i] == True:
# 			print i
# 	fig = plt.figure()
# 	ax1 = fig.add_subplot(2,1,1)
# 	ax2 = fig.add_subplot(2,1,2)
# 	ax1.plot(data[feature_name[0]])
# 	ax2.plot(data['isAnomaly'])
# 	plt.show()




if opt.function == 'loadData':
	loadData()
elif opt.function == 'meanDecreaseImpurityOnWeibo':
	meanDecreaseImpurityOnWeibo()
# elif opt.function == 'stabilitySelectionOnWeibo':
# 	stabilitySelectionOnWeibo()
elif opt.function == 'univChi2':
	univChi2()
# elif opt.function == 'svmRFE':
# 	svmRFE()
elif opt.function == 'varianceFS':
	varianceFS()
# elif opt.function == 'univariateFS':
# 	univariateFS()
elif opt.function == 'count':
	count()
elif opt.function == 'catagoryChi2':
	catagoryChi2()
# elif opt.function == 'linearWithL1':
# 	linearWithL1()
# elif opt.function == 'drawhist':
# 	drawhist()
# elif opt.function == 'outlierDetection':
# 	outlierDetection()
elif opt.function == 'logisticR':
	logisticR()
elif opt.function == 'pearson':
	pearson()
elif opt.function == 'drawScatter':
	drawScatter()
elif opt.function == 'gradientBoosting':
	gradientBoosting()
elif opt.function == 'DTvisualization':
	DTvisualization()
elif opt.function == 'linearSVC':
	linearSVC()
elif opt.function == 'meanDecreaseAccuracyOnWeibo':
	meanDecreaseAccuracyOnWeibo()