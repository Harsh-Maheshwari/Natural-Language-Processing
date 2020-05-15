# Import Libraries
import numpy as np
import pandas as pd
# Data Vis
import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns
# Stat
from scipy import stats
from scipy.stats import norm

import glob
import nltk
import string
import pandas as pd
import numpy as np
import seaborn as sns
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit

import pickle


text  = 'Content'
use_regex = False
regex_used = r'\w+'
stemming = False
lemmatization = not stemming
category_codes = {'business': 0,'entertainment': 1,'politics': 2,'sport': 3,'tech': 4}
EDA = False
df_train_csv = glob.glob('./df_train.csv')
big_pickles = glob.glob('./df_train_processed_X_train_y_train_X_test_y_test_features_train_labels_train_features_test_labels_test.pickle')

if len(big_pickles) == 0:
	if len(df_train_csv) == 0:
		data = []
		for i in glob.glob('./data/*/*.txt'):
			content = []
			f = open(i,"r",encoding="utf8",errors="ignore")
			A = i.split('/')
			a = f.read()
			# a = a.split('\n')
			content.append(a)
			A.extend(content)
			data.append(A[2:])


		df_train = pd.DataFrame(data)
		df_train.columns = ['Category','File_Name','Content']
		df_train['Complete_Filename'] = list(map(lambda x: "-".join((x[0],x[1])),list(df_train.itertuples(index=False, name=None))))

		import csv

		with open('df_train.csv', 'w', newline='') as file:
		    writer = csv.writer(file)
		    writer.writerows(df_train.values.tolist())
	else:
		df_train = pd.read_csv("df_train.csv")
		df_train.columns = ['Category','File_Name','Content','Complete_Filename']

	# EDA

	if EDA ==True:
		plt.hist(df_train['Category'])
		plt.show(block=False)
		plt.pause(4)
		plt.close()

		print(df_train['Category'].value_counts())

		df_train['News_length'] = df_train['Content'].str.len()
		plt.figure(figsize=(12.8,6))
		sns.distplot(df_train['News_length']).set_title('News length distribution');
		plt.show(block=False)
		plt.pause(3)
		plt.close()


		quantile_95 = df_train['News_length'].quantile(0.95)
		df_95 = df_train[df_train['News_length'] < quantile_95]
		plt.figure(figsize=(12.8,6))
		sns.distplot(df_95['News_length']).set_title('News length distribution for 95% quantile data');
		plt.show(block=False)
		plt.pause(3)
		plt.close()


		plt.figure(figsize=(12.8,6))
		sns.boxplot(data=df_train, x='Category', y='News_length', width=.5).set_title('News length boxplot complete data');
		plt.show(block=False)
		plt.pause(3)
		plt.close()

		plt.figure(figsize=(12.8,6))
		sns.boxplot(data=df_95, x='Category', y='News_length').set_title('News length boxplot for 95% quantile data');
		plt.show(block=False)
		plt.pause(3)
		plt.close()

	# Text Cleaning
	df_train[text] = df_train[text].apply(lambda x:BeautifulSoup(x,'lxml').get_text()) 

	df_train[text] = df_train[text].apply(lambda x:"".join([c for c in x if c not in string.punctuation]))

	if use_regex == True:
		tokenizer = RegexpTokenizer(regex_used) 
		# RegEx Tokenizer and lower
		df_train[text] = df_train[text].apply(lambda x:tokenizer.tokenize(x.lower())) 
	else:
		# Word Tokenizer and lower
		df_train[text] = df_train[text].apply(lambda x:word_tokenize(x.lower())) 

	df_train[text] = df_train[text].apply(lambda x:[w for w in x if w not in stopwords.words('english')]) 

	if stemming == True:
		# Stemming: converting a word into its root form. prefixes or suffixes get removed that results in a word that may or may not be meaningful.
		stemmer = PorterStemmer()
		df_train[text] = df_train[text].apply(lambda x:" ".join([stemmer.stem(i) for i in x]))
	if lemmatization == True:
		# Lemmatization: converting a word into its root form but unlike stemming it always returns a proper word that can be found in a dictionary.
		lemmatizer = WordNetLemmatizer()
		df_train[text] = df_train[text].apply(lambda x:" ".join([lemmatizer.lemmatize(i) for i in x]))

	# Category mapping
	df_train['Category_Code'] = df_train['Category']
	df_train = df_train.replace({'Category_Code':category_codes})
	X_train, X_test, y_train, y_test = train_test_split(df_train['Content'], df_train['Category_Code'], test_size=0.15, random_state=8)

	'''
	We have various options for feature engineering:

	* Count Vectors as features
	* TF-IDF Vectors as features  [Chossing this easy one]
	* Word Embeddings as features
	* Text / NLP based features
	* Topic Models as features


	We have to define the different parameters:

	* `ngram_range`: We want to consider both unigrams and bigrams.
	* `max_df`: When building the vocabulary ignore terms that have a document
	    frequency strictly higher than the given threshold
	* `min_df`: When building the vocabulary ignore terms that have a document
	    frequency strictly lower than the given threshold.
	* `max_features`: If not None, build a vocabulary that only consider the top
	    max_features ordered by term frequency across the corpus.

	See `TfidfVectorizer?` for further detail.
	'''

	# Parameter selection
	ngram_range = (1,2)
	min_df = 10
	max_df = 1.
	max_features = 300

	tfidf = TfidfVectorizer(encoding='utf-8',
	                        ngram_range=ngram_range,
	                        stop_words=None,
	                        lowercase=False,
	                        max_df=max_df,
	                        min_df=min_df,
	                        max_features=max_features,
	                        norm='l2',
	                        sublinear_tf=True)
	                        
	features_train = tfidf.fit_transform(X_train).toarray()
	labels_train = y_train
	features_test = tfidf.transform(X_test).toarray()
	labels_test = y_test


	# We can use the Chi squared test in order to see what unigrams and bigrams are most correlated with each category:

	from sklearn.feature_selection import chi2

	for Product, category_id in sorted(category_codes.items()):
	    features_chi2 = chi2(features_train, labels_train == category_id)
	    indices = np.argsort(features_chi2[0])
	    feature_names = np.array(tfidf.get_feature_names())[indices]
	    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
	    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
	    print("Most correlated unigrams in {}:\n. {}".format(Product,'\n. '.join(unigrams[-5:])))
	    print("Most correlated bigrams in {}:\n. {}".format(Product,'\n. '.join(bigrams[-5:])))

	with open('df_train_processed_X_train_y_train_X_test_y_test_features_train_labels_train_features_test_labels_test.pickle', 'wb') as output:
	    pickle.dump([df_train,X_train,y_train,X_test,y_test,features_train,labels_train,features_test,labels_test], output)

	with open('tfidf.pickle', 'wb') as output:
	    pickle.dump(tfidf, output)


else:
	with open('df_train_processed_X_train_y_train_X_test_y_test_features_train_labels_train_features_test_labels_test.pickle', 'rb') as data:
	    [df_train,X_train,y_train,X_test,y_test,features_train,labels_train,features_test,labels_test] = pickle.load(data)


## Building the model

'''
Once we have our feature vectors built, we'll try several machine learning classification models
in order to find which one performs best on our data. We will try with the following models:

* Random Forest
* Support Vector Machine
* K Nearest Neighbors
* Multinomial Naïve Bayes
* Multinomial Logistic Regression
* Gradient Boosting

The methodology used to train each model is as follows:

1. First of all, we'll decide which hyperparameters we want to tune.

2. Secondly, we'll define the metric we'll get when measuring the performance of 
a model. In this case, we'll use the **accuracy**.

3. We'll perform a Randomized Search Cross Validation process in order to find 
the hyperparameter region in which we get higher values of accuracy.

4. Once we find that region, we'll use a Grid Search Cross Validation process to 
exhaustively find the best combination of hyperparameters.

5. Once we obtain the best combination of hyperparameters, we'll obtain the accuracy on 
the training data and the test data, the classification report and the confusion matrix.

6. Finally, we'll calculate the accuracy of a model with default hyperparameters, 
to see if we have achieved better results by hyperparameter tuning.

We need to be aware of the fact that our dataset only contains 5 categories:

* Business
* Politics
* Sports
* Tech
* Entertainment

So, when we get news articles that don't belong to any of that categories (for example, weather or 
terrorism news articles), we will surely get a wrong prediction. For this reason we will take into 
account the conditional probability of belonging to every class and set a lower threshold (i.e. if 
the 5 conditional probabilities are lower than 65% then the prediction will be 'other'). This 
probability vector can be obtained in a simple way in some models, but not in other ones. For this
reason we will take this into consideration when choosing the model to use.
'''

def build_model(model_to_use,df_train,X_train,y_train,X_test,y_test,features_train,labels_train,features_test,labels_test):

	select_model={
	'GradientBoostingClassifier':GradientBoostingClassifier(random_state = 8),
	'LogisticRegression':LogisticRegression(random_state = 8),
	'KNeighborsClassifier':KNeighborsClassifier(),
	'SVC':svm.SVC(random_state=8),
	'RandomForestClassifier':RandomForestClassifier(random_state = 8)
	}

	# Cross-Validation for Hyperparameter tuning

	clf = select_model[model_to_use]
	print('Parameters currently in use:\n')
	pprint(clf.get_params())
	print("")

	### Randomized Search Cross Validation

	if model_to_use == 'RandomForestClassifier':
		'''
		We'll tune the following ones:

		* `n_estimators` = number of trees in the forest.
		* `max_features` = max number of features considered for splitting a node
		* `max_depth` = max number of levels in each decision tree
		* `min_samples_split` = min number of data points placed in a node before the node is split
		* `min_samples_leaf` = min number of data points allowed in a leaf node
		* `bootstrap` = method for sampling data points (with or without replacement)

		'''

		n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]
		max_features = ['auto', 'sqrt']
		max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
		max_depth.append(None)
		min_samples_split = [2, 5, 10]
		min_samples_leaf = [1, 2, 4]
		bootstrap = [True, False]

		# Create the random grid
		random_grid = {'n_estimators': n_estimators,
		               'max_features': max_features,
		               'max_depth': max_depth,
		               'min_samples_split': min_samples_split,
		               'min_samples_leaf': min_samples_leaf,
		               'bootstrap': bootstrap}

	if model_to_use == 'SVC':

		'''
		We'll tune the following ones:

		* `C`: Penalty parameter C of the error term.
		* `kernel`: Specifies the kernel type to be used in the algorithm.
		* `gamma`: Kernel coefficient.
		* `degree`: Degree of the polynomial kernel function.
		'''

		C = [.0001, .001, .01]
		gamma = [.0001, .001, .01, .1, 1, 10, 100]
		degree = [1, 2, 3, 4, 5]
		kernel = ['linear', 'rbf', 'poly']
		probability = [True]
		
		# Create the random grid
		random_grid = {'C': C,
		              'kernel': kernel,
		              'gamma': gamma,
		              'degree': degree,
		              'probability': probability
		             }
	if model_to_use == 'GradientBoostingClassifier':

		'''
		We'll tune the following ones:

		Tree-related hyperparameters:
		* `n_estimators` = number of trees in the forest.
		* `max_features` = max number of features considered for splitting a node
		* `max_depth` = max number of levels in each decision tree
		* `min_samples_split` = min number of data points placed in a node before the node is split
		* `min_samples_leaf` = min number of data points allowed in a leaf node

		Boosting-related hyperparameters:
		* `learning_rate`= learning rate shrinks the contribution of each tree by learning_rate.
		* `subsample`= the fraction of samples to be used for fitting the individual base learners.

		'''
		n_estimators = [200, 800]
		max_features = ['auto', 'sqrt']
		max_depth = [10, 40]
		max_depth.append(None)
		min_samples_split = [10, 30, 50]
		min_samples_leaf = [1, 2, 4]
		learning_rate = [.1, .5]
		subsample = [.5, 1.]

		# Create the random grid
		random_grid = {'n_estimators': n_estimators,
		               'max_features': max_features,
		               'max_depth': max_depth,
		               'min_samples_split': min_samples_split,
		               'min_samples_leaf': min_samples_leaf,
		               'learning_rate': learning_rate,
		               'subsample': subsample}



	if model_to_use == 'LogisticRegression':
		
		'''

		We'll tune the following ones:

		* `C` = Inverse of regularization strength. Smaller values specify stronger regularization.
		* `multi_class` = We'll choose `multinomial` because this is a multi-class problem.
		* `solver` = Algorithm to use in the optimization problem. For multiclass problems, only `newton-cg`, `sag`, `saga` and `lbfgs` handle multinomial loss.
		* `class_weight`: Weights associated with classes. 
		* `penalty`: Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.

		'''

		C = [float(x) for x in np.linspace(start = 0.1, stop = 1, num = 10)]
		multi_class = ['multinomial']
		solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
		class_weight = ['balanced', None]
		penalty = ['l2']

		# Create the random grid
		random_grid = {'C': C,
		               'multi_class': multi_class,
		               'solver': solver,
		               'class_weight': class_weight,
		               'penalty': penalty}


	if model_to_use == 'KNeighborsClassifier':

		# Create the parameter grid 
		n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 500, num = 100)]

		random_grid = {'n_neighbors': n_neighbors}

	pprint(random_grid)
	print("")


	# First create the base model to tune
	clf = select_model[model_to_use]

	# Definition of the random search
	random_search = RandomizedSearchCV(estimator=clf,
	                                   param_distributions=random_grid,
	                                   n_iter=50,
	                                   n_jobs=7,
	                                   scoring='accuracy',
	                                   cv=3, 
	                                   verbose=1, 
	                                   random_state=8)

	# Fit the random search model
	random_search.fit(features_train, labels_train)

	print("The best hyperparameters from Random Search are:")
	print(random_search.best_params_)
	print("")
	print("The mean accuracy of a model with these hyperparameters is:")
	print(random_search.best_score_)
	print("")


	### Grid Search Cross Validation

	if model_to_use == 'RandomForestClassifier':
		# Create the parameter grid based on the results of random search 
		bootstrap = [False]
		max_depth = [30, 40, 50]
		max_features = ['sqrt']
		min_samples_leaf = [1, 2, 4]
		min_samples_split = [5, 10, 15]
		n_estimators = [800]

		param_grid = {
		    'bootstrap': bootstrap,
		    'max_depth': max_depth,
		    'max_features': max_features,
		    'min_samples_leaf': min_samples_leaf,
		    'min_samples_split': min_samples_split,
		    'n_estimators': n_estimators
		}

	if model_to_use == 'SVC':
	# Create the parameter grid based on the results of random search 
		C = [.0001, .001, .01, .1]
		degree = [3, 4, 5]
		gamma = [1, 10, 100]
		probability = [True]

		param_grid = [
		  {'C': C, 'kernel':['linear'], 'probability':probability},
		  {'C': C, 'kernel':['poly'], 'degree':degree, 'probability':probability},
		  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
		]

	if model_to_use == 'GradientBoostingClassifier':
		# Create the parameter grid based on the results of random search 
		max_depth = [5, 10, 15]
		max_features = ['sqrt']
		min_samples_leaf = [2]
		min_samples_split = [50, 100]
		n_estimators = [800]
		learning_rate = [.1, .5]
		subsample = [1.]

		param_grid = {
		    'max_depth': max_depth,
		    'max_features': max_features,
		    'min_samples_leaf': min_samples_leaf,
		    'min_samples_split': min_samples_split,
		    'n_estimators': n_estimators,
		    'learning_rate': learning_rate,
		    'subsample': subsample

		}

	if model_to_use == 'LogisticRegression':

		# Create the parameter grid based on the results of random search 
		C = [float(x) for x in np.linspace(start = 0.6, stop = 1, num = 10)]
		multi_class = ['multinomial']
		solver = ['sag']
		class_weight = ['balanced']
		penalty = ['l2']

		param_grid = {'C': C,
		               'multi_class': multi_class,
		               'solver': solver,
		               'class_weight': class_weight,
		               'penalty': penalty}




	if model_to_use == 'KNeighborsClassifier':

		# Create the parameter grid 
		n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 500, num = 100)]

		param_grid = {'n_neighbors': n_neighbors}

	# Create a base model
	clf = select_model[model_to_use]

	# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
	cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

	# Instantiate the grid search model
	grid_search = GridSearchCV(estimator=clf, 
	                           param_grid=param_grid,
	                           scoring='accuracy',
	                           n_jobs=7,
	                           cv=cv_sets,
	                           verbose=1)

	# Fit the grid search to the data
	grid_search.fit(features_train, labels_train)


	print("The best hyperparameters from Grid Search are:")
	print(grid_search.best_params_)
	print("")
	print("The mean accuracy of a model with these hyperparameters is:")
	print(grid_search.best_score_)
	print("")

	best_clf = grid_search.best_estimator_

	print(best_clf)
	print("")

	best_clf.fit(features_train, labels_train)
	clf_pred = best_clf.predict(features_test)

	# Training accuracy
	print("The training accuracy is: ")
	print(accuracy_score(labels_train, best_clf.predict(features_train)))
	print("")

	# Test accuracy
	print("The test accuracy is: ")
	print(accuracy_score(labels_test, clf_pred))
	print("")


	# Classification report
	print("Classification report")
	print(classification_report(labels_test,clf_pred))
	print("")


	aux_df = df_train[['Category', 'Category_Code']].drop_duplicates().sort_values('Category_Code')
	conf_matrix = confusion_matrix(labels_test, clf_pred)
	plt.figure(figsize=(12.8,6))
	sns.heatmap(conf_matrix, 
	            annot=True,
	            xticklabels=aux_df['Category'].values, 
	            yticklabels=aux_df['Category'].values,
	            cmap="Blues")
	plt.ylabel('Predicted')
	plt.xlabel('Actual')
	plt.title('Confusion matrix')
	plt.show(block=False)
	plt.pause(3)
	plt.close()


	# Let's see if the hyperparameter tuning process has returned a better model:

	base_model = select_model[model_to_use]
	base_model.fit(features_train, labels_train)
	accuracy_score(labels_test, base_model.predict(features_test))


	best_clf.fit(features_train, labels_train)
	accuracy_score(labels_test, best_clf.predict(features_test))

	d = {
	     'Model': model_to_use,
	     'Training Set Accuracy': accuracy_score(labels_train, best_clf.predict(features_train)),
	     'Test Set Accuracy': accuracy_score(labels_test, clf_pred)
	}

	df_models_clf = pd.DataFrame(d, index=[0])

	with open('best_'+model_to_use+'.pickle', 'wb') as output:
	    pickle.dump(best_clf, output)
	    
	with open('df_models_'+model_to_use+'.pickle', 'wb') as output:
	    pickle.dump(df_models_clf, output)

for i in  ['GradientBoostingClassifier','LogisticRegression','KNeighborsClassifier','SVC','RandomForestClassifier'] :
	model_to_use = i
	# build_model(model_to_use,df_train,X_train,y_train,X_test,y_test,features_train,labels_train,features_test,labels_test)

list_pickles = [
    "df_models_GradientBoostingClassifier.pickle",
    "df_models_LogisticRegression.pickle",
    "df_models_KNeighborsClassifier.pickle",
    "df_models_SVC.pickle",
    "df_models_RandomForestClassifier.pickle"
]

df_summary = pd.DataFrame()
for path in list_pickles:
    with open(path, 'rb') as data:
        df = pickle.load(data)
    df_summary = df_summary.append(df)

df_summary = df_summary.reset_index().drop('index', axis=1).sort_values('Test Set Accuracy', ascending=False)
df_summary['Test/Train'] = df_summary.apply(lambda x: x[-1]/x[-2],axis = 1,raw=True)
print(df_summary)


# Test a new random article

sample = """
Spain’s animal rights party PACMA posted a 38-second video on Twitter on Friday showing a man freeing a fox from a cage, before hunters immediately start shooting at it.

“Hunters shut what appears to be a fox in a cage and let it out only to pepper it with bullets,” says the accompanying text. “Another ‘isolated case’ as the hunting lobby refers to it. Every week, a trickle of ‘isolated cases.’ In fact, they are dangerous psychopaths with a rifle and a license to carry arms.”

 Video insertado

PACMA
✔
@PartidoPACMA
 Cazadores enjaulan a lo que parece ser un zorro y lo liberan solo para acribillarlo a tiros. Otro "caso aislado", de los que habla el lobby de la caza. Cada semana varios "casos aislados".

En realidad, son peligrosos psicópatas con escopeta y permiso de amas. #YoNoDisparo

4.188
10:43 - 4 ene. 2019
7.443 personas están hablando de esto
Información y privacidad de Twitter Ads
At the start of the video, a man teases the caged animal with a stick. When the cage door is opened, the animal makes a run for it, but is shot at by men armed with rifles who are waiting by the cage.

The release of the video, which has had 255,000 views, coincided with the launch of PACMA’s campaign against the start of fox-hunting season in Galicia. “Fox-hunting season in Galicia has started: hunts that hide behind environmental excuses, championships in which the only reason to compete is to kill. The hunters will be entitled to pursue and kill thousands of foxes in the countryside,” states PACMA.

As it notes on its website, PACMA is the only political group that opposes hunting, and it is currently demanding a nationwide ban. “No animal should die under fire,” say the group. “We will fight tirelessly until hunting becomes a crime.”

No animal should die under fire. We will fight tirelessly until hunting becomes a crime

PACMA

The animal rights group is preparing a report to send to the regional government of Galicia against fox hunts. “We are working hard to make it the first Spanish region to assign resources to protecting foxes instead of killing them,” says a source at PACMA.

Last month, a Spanish hunter who was filmed while he chased and tortured a fox was identified by the Civil Guard in the Spanish province of Huesca. The man, aged 35, is facing charges of crimes against wildlife.

And in November, animal rights groups and political parties reacted with indignation over a viral video shot in Cáceres province of 12 hunting dogs falling off a cliff edge, followed by the deer they were attacking.

"""


choosen_model = 'SVC'

with open('best_'+choosen_model+'.pickle', 'rb') as data:
	model = pickle.load(data)

with open('tfidf.pickle', 'rb') as data:
	tfidf = pickle.load(data)



sample = pd.DataFrame([sample])
sample.columns=[text]
sample[text] = sample[text].apply(lambda x:BeautifulSoup(x,'lxml').get_text()) 
sample[text] = sample[text].apply(lambda x:"".join([c for c in x if c not in string.punctuation]))
if use_regex == True:
	tokenizer = RegexpTokenizer(regex_used) 
	# RegEx Tokenizer and lower
	sample[text] = sample[text].apply(lambda x:tokenizer.tokenize(x.lower())) 
else:
	# Word Tokenizer and lower
	sample[text] = sample[text].apply(lambda x:word_tokenize(x.lower())) 

sample[text] = sample[text].apply(lambda x:[w for w in x if w not in stopwords.words('english')]) 

if stemming == True:
	# Stemming: converting a word into its root form. prefixes or suffixes get removed that results in a word that may or may not be meaningful.
	stemmer = PorterStemmer()
	sample[text] = sample[text].apply(lambda x:" ".join([stemmer.stem(i) for i in x]))
if lemmatization == True:
	# Lemmatization: converting a word into its root form but unlike stemming it always returns a proper word that can be found in a dictionary.
	lemmatizer = WordNetLemmatizer()
	sample[text] = sample[text].apply(lambda x:" ".join([lemmatizer.lemmatize(i) for i in x]))

features_sample = tfidf.transform(sample[text]).toarray()
prediction = model.predict(features_sample)[0]
prediction_prob = model.predict_proba(features_sample)[0]
for category, id_ in category_codes.items():    
    if id_ == prediction:
    	predicted_category = category 


if prediction_prob.max() < 0.65:
	predicted_category = 'other'

print("The predicted category using the %s model is %s." %(choosen_model,predicted_category) )
print("The conditional probability is: %a" %(prediction_prob.max()*100))

















##################ARCHIVE############################################ARCHIVE############################################ARCHIVE############################################ARCHIVE##########################

# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import RegexpTokenizer
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from nltk.stem.porter import PorterStemmer
# from bs4 import BeautifulSoup
# import string
# import pandas as pd

# d = ['''A corpus may contain texts in a single language (monolingual corpus) or text data in multiple languages (multilingual corpus).

# Multilingual corpora that have been specially formatted for side-by-side comparison are called aligned parallel corpora. There are two main types of parallel corpora which contain texts in two languages. In a translation corpus, the texts in one language are translations of texts in the other language. In a comparable corpus, the texts are of the same kind and cover the same content, but they are not translations of each other.[1] To exploit a parallel text, some kind of text alignment identifying equivalent text segments (phrases or sentences) is a prerequisite for analysis. Machine translation algorithms for translating between two languages are often trained using parallel fragments comprising a first language corpus and a second language corpus which is an element-for-element translation of the first language corpus.[2]

# In order to make the corpora more useful for doing linguistic research, they are often subjected to a process known as annotation. An example of annotating a corpus is part-of-speech tagging, or POS-tagging, in which information about each word's part of speech (verb, noun, adjective, etc.) is added to the corpus in the form of tags. Another example is indicating the lemma (base) form of each word. When the language of the corpus is not a working language of the researchers who use it, interlinear glossing is used to make the annotation bilingual.

# Some corpora have further structured levels of analysis applied. In particular, a number of smaller corpora may be fully parsed. Such corpora are usually called Treebanks or Parsed Corpora. The difficulty of ensuring that the entire corpus is completely and consistently annotated means that these corpora are usually smaller, containing around one to three million words. Other levels of linguistic structured analysis are possible, including annotations for morphology, semantics and pragmatics.

# Corpora are the main knowledge base in corpus linguistics. The analysis and processing of various types of corpora are also the subject of much work in computational linguistics, speech recognition and machine translation, where they are often used to create hidden Markov models for part of speech tagging and other purposes. Corpora and frequency lists derived from them are useful for language teaching. Corpora can be considered as a type of foreign language writing aid as the contextualised grammatical knowledge acquired by non-native language users through exposure to authentic texts in corpora allows learners to grasp the manner of sentence formation in the target language, enabling effective writing''']

# df_train = pd.DataFrame([d,d])
# df_train.columns = ['text']

# text = 'text'
# use_regex = False
# regex_used = r'\w+'
# stemming = False
# lemmatization = not stemming

# df_train[text] = df_train[text].apply(lambda x:BeautifulSoup(x,'lxml').get_text()) 
# df_train[text] = df_train[text].apply(lambda x:"".join([c for c in x if c not in string.punctuation]))
# if use_regex == True:
# 	tokenizer = RegexpTokenizer(regex_used) 
# 	# RegEx Tokenizer and lower
# 	df_train[text] = df_train[text].apply(lambda x:tokenizer.tokenize(x.lower())) 
# else:
# 	# Word Tokenizer and lower
# 	df_train[text] = df_train[text].apply(lambda x:word_tokenize(x.lower())) 
# df_train[text] = df_train[text].apply(lambda x:[w for w in x if w not in stopwords.words('english')]) 

# if stemming == True:
# 	# Stemming: converting a word into its root form. prefixes or suffixes get removed that results in a word that may or may not be meaningful.
# 	stemmer = PorterStemmer()
# 	df_train[text] = df_train[text].apply(lambda x:" ".join([stemmer.stem(i) for i in x]))
# if lemmatization == True:
# 	# Lemmatization: converting a word into its root form but unlike stemming it always returns a proper word that can be found in a dictionary.
# 	lemmatizer = WordNetLemmatizer()
# 	df_train[text] = df_train[text].apply(lambda x:" ".join([lemmatizer.lemmatize(i) for i in x]))

# # print(df_train.iloc[:,0].values)


