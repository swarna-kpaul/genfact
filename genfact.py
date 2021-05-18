from genfact.util import *
from sklearn.ensemble import RandomForestClassifier

def generate_counterfactuals(data_df,targetclass_idx,model,featuredtype, clustsize = 20, datafraction = 0.4, maxiterations = 10):
	classdata = data_df.iloc[:,targetclass_idx].values
	data_df.drop(data_df.columns[targetclass_idx], axis = 1, inplace = True)
	featuredata = data_df
	######## generate Randomforest prediction model 
	clf = RandomForestClassifier(max_depth=10, random_state=0)
	clf.fit(featuredata, classdata)
	classdata = clf.predict(featuredata)
	####### create sorted clusters 
	sortedclusters= create_clusters(featuredata,classdata,clustsize)
	######### 40% of data #######
	datasize = sum([ len(v) for i,v in sortedclusters])*datafraction
	processeddatasize=0
	factuals = np.array([])
	for score,cluster in sortedclusters:
		population,classdata,counterfacts,counterfactsclass = run_genetic(cluster,model=model,dtype=featuredtype,maxiterations=maxiterations)
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
	return (factuals,counterfactuals,factclass,cfactclass)
	
