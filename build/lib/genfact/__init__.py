from genfact.util import *
from sklearn.ensemble import RandomForestClassifier
import pkg_resources
import pandas as pd
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
"""Author: Swarna Kamal Paul"""
"""email: swarna.kpaul@gmail.com"""

def generate_counterfactuals(data_df,dtype,targetclass_idx, model=None, C=15, clustsize = 20, datafraction = 0.4, maxiterations = 10):
	
	###### Prepare data #############
	"""
		If target attribute is continous then divide the attribute in C buckets with equal distribution ignoring duplicates.
		If duplicates are present the actual number of buckets formed will be lesser than C
		Each bucket is assigned a unique class. 
		The data is seperated into featuredata and classdata. It also returns a classdistribution if the target attribute is continous.
	"""
	logging.info('Preparing Data ..')
	if dtype[targetclass_idx] == 'con':
		data_df['bins']=pd.qcut(data_df.iloc[:,targetclass_idx], q=C, duplicates = 'drop') 
		data_df['class']=pd.qcut(data_df.iloc[:,targetclass_idx], q=C, duplicates = 'drop').cat.codes
		classdistribution = data_df[['bins','class']].drop_duplicates() 
		classdata = data_df['class'].values
		data_df.drop(columns=['bins','class'], axis = 1, inplace = True)
	else:
		classdata = data_df.iloc[:,targetclass_idx].values
		classdistribution = None
	data_df.drop(data_df.columns[targetclass_idx], axis = 1, inplace = True)
	featuredata = data_df.iloc[:, :].values
	dtype.pop(targetclass_idx)
	logging.info('Data preparation Complete! Got '+str(len(set(classdata)))+" unique classes!")
	if model == None:
	######## generate Randomforest prediction model ################
		logging.info('Training Random Forest model with max depth 10...')
		model = RandomForestClassifier(max_depth=10, random_state=0)
		model.fit(featuredata, classdata)
		logging.info('Model training complete!')
	logging.info('Clustering feature data...')
	classdata = model.predict(featuredata)
	####### create sorted clusters 
	sortedclusters= create_clusters(featuredata,classdata,clustsize)
	logging.info('Clustering feature data complete!')
	######### measure fractional size of data #######
	logging.info('Genrating counterfactuals ....')
	datasize = sum([ len(v) for i,v in sortedclusters])*datafraction
	############## generate factual and counterfactual pairs ############
	processeddatasize=0
	factuals = np.array([])
	cluster_no = 1
	for score,cluster in sortedclusters:
		logging.info('Running genetic algo for cluster# '+str(cluster_no)+' having size '+str(len(cluster))+' and having diversity score ' +str(score))
		cluster_no +=1
		population,classdata,counterfacts,counterfactsclass = run_genetic(cluster,model=model,dtype=dtype,maxiterations=maxiterations)
		if factuals.size == 0:
			factuals = population
			counterfactuals = counterfacts
			factclass = list(classdata)
			cfactclass = list(counterfactsclass)
		else:
			factuals = np.concatenate([factuals,population],axis=0)
			counterfactuals = np.concatenate([counterfactuals,counterfacts],axis=0)
			factclass = factclass + list(classdata)
			cfactclass = cfactclass + list(counterfactsclass)
		processeddatasize += len(cluster)
		if processeddatasize >= datasize: #len(factuals)>=30:
			break
	logging.info('Genrating counterfactuals complete!')
	return (factuals,counterfactuals,factclass,cfactclass,classdistribution)
	

def evaluate_counterfactuals(factuals,counterfactuals,factclass,cfactclass):
	########## Returns entropy and average euclidean distance among factuals and counterfactuals ################
	cfsamplclass=[str((i,j)) for i,j in zip (factclass,cfactclass)]
	classfraction=np.unique(cfsamplclass, return_counts=True)[1]/len(cfsamplclass)
	fitness = []
	for facts, cfacts in zip(factuals,counterfactuals):
		fitness.append(np.linalg.norm(facts-cfacts))
	return (entropy(classfraction,base=10), np.mean(fitness))
	
def load_data():
	"""Return an encoded dataframe about the Facebook Advertisement data.
	Source from https://www.kaggle.com/chrisbow/an-introduction-to-facebook-ad-analysis-using-r

	Contains the following fields:
		age: age of the person to whom the ad is shown.
		gender: gender of the person to whom the add is shown
		interest: a code specifying the category to which the person’s interest belongs (interests are as mentioned in the person’s Facebook public profile).
		Spentperclick: Amount paid by company xyz to Facebook, to show that ad and per click.
		Clicks: No. of clicks on an ad.
		Impressions: the number of times the ad was shown.
		Total conversion: Total number of people who enquired about the product after seeing the ad.
	"""
	# This is a stream-like object. If you want the actual info, call
	# stream.read()
	dtype = ['cat','cat','cat','con','con','con','con']
	targetclass_idx = 6	
	stream = pkg_resources.resource_stream(__name__, 'data/encoded_fbdata.csv')
	return (pd.read_csv(stream, encoding='latin-1'),dtype,targetclass_idx)