import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
import random
import pickle
"""Author: Swarna Kamal Paul"""
"""email: swarna.kpaul@gmail.com"""
"""This file contains helper functions to run genfact algorithm
	create_clusters : Takes featuredata and classdata as array and clustersize as a number. 
	Returns clusters of featuredata sorted in the descending order of clusterscore.
	
	crossover: Takes population of featuredata, array of classdata for the population, predictive model, datatype of featuredata and offspringsize as a number.
	Returns new population after crossover and classdata for the new population
	
	evaluatefitness: Takes population of featuredata, array of classdata for the population.
	For each sample in population it finds the another sample in the population as counterfactual having different class and minimum euclidean distance (fitness) with respect to the sample. The fitness and counterfactuals are returned
	
	selectbest: Based on populationsize it returns the most fit set of factual and counterfactual pairs
	
	run_genetic: Using the clustered featuredata, datatype and the model it run the genetic algorithm and returns the factual and counterfactual pairs.
	"""

def create_clusters(featuredata,classdata,clustsize=20):
	k=round(len(classdata)/(clustsize*2))#round(len(np.unique(classdata))*1.5)
	kmeansk = KMeans(n_clusters=k)
	y_kmeans = kmeansk.fit_predict(featuredata)
	clusters=[]
	clusterclassfraction=[]
	clusterentropy = []
	for i in range(k):
		cdata = featuredata[y_kmeans == i]
		tclass = classdata[y_kmeans == i]
		classfraction=np.unique(tclass, return_counts=True)[1]/len(tclass)
		clusters.append(cdata)
		clusterclassfraction.append(classfraction)
	classfractionentropy=[entropy(i) for i in clusterclassfraction]
	sizeofclusters=[len(i) for i in clusters]
	clusterscore = [x/np.log(y+1) for x, y in zip(classfractionentropy, sizeofclusters)]
	sortedclusters = [(y,x) for y,x in sorted(zip(clusterscore,clusters),key=lambda pair: pair[0],reverse=True)]
	return sortedclusters
	
def crossover(population,classdata,model,dtype,offspringsize):
	#random.seed(10)
	currsize = 0
	search_time = 0
	while currsize <= offspringsize:
		search_time +=1
		p1_index = random.choice(range(len(population)))
		p1 = population[p1_index]
		p1_class = classdata[p1_index]
		f1 = random.choice(range(len(dtype)))
		##### create population for p2 having other than p1_class
		p2_population = population[classdata != p1_class]
		p2 = p2_population[random.choice(range(len(p2_population)))]
		offspring = pickle.loads(pickle.dumps(p1,-1))
		if dtype[f1] == 'cat':
			offspring[f1] = pickle.loads(pickle.dumps(p2[f1],-1))
		else:
			offspring[f1] = pickle.loads(pickle.dumps((p2[f1] + p1[f1])/2,-1))
		if str(offspring) in [str(i) for i in population] and search_time<=len(population)*2:     
			continue
		else:
			search_time=0 
		
		try:
			classdata = np.concatenate([classdata,model.predict([offspring])],axis = 0)
		except Exception as TypeError:
			classdata = np.concatenate([classdata, model.predict(offspring.reshape((1, offspring.shape[0])))], axis=0)
		population = np.vstack([population,offspring])
		currsize +=1
	return (population,classdata)


def evaluatefitness(samppopulation,sampclassdata):
	fitpopval = []
	counterfacts = []
	counterfactsclass = []
	for sample,cls in zip(samppopulation,sampclassdata):
		counterclass = sampclassdata[sampclassdata !=cls]	 
		tgt_population = samppopulation[sampclassdata !=cls]
		fitval=[np.linalg.norm(j-sample) for j in tgt_population]	
		minidx = np.argmin(fitval)
		counterfacts.append(tgt_population[minidx])
		counterfactsclass.append(counterclass[minidx]) 
		fitpopval.append(fitval[minidx])
	return (fitpopval,counterfacts,counterfactsclass)
 
 
def selectbest(fitness,population,classdata,counterfacts,counterfactsclass,populationsize):
	fitness =	 np.log(np.array(fitness)+1)
	#fitness =	 softmax(fitness)
	bestpopulation = [(y,x,z,a,b) for y,x,z,a,b in sorted(zip(fitness,population,classdata,counterfacts,counterfactsclass),key=lambda pair: pair[0])]
	bestpopulation=bestpopulation[0:populationsize]
	newpopulation = [list(i) for _,i,_,_,_ in bestpopulation]
	newclass = [j for _,_,j,_,_ in bestpopulation]
	counterfacts = [j for _,_,_,j,_ in bestpopulation]
	counterfactsclass = [j for _,_,_,_,j in bestpopulation]
	return (np.array(newpopulation),np.array(newclass),counterfacts,counterfactsclass)
	
def run_genetic(cluster,model,dtype,maxiterations=10):
	population = cluster
	classdata = model.predict(population)
	un_class,un_class_dist=np.unique(classdata, return_counts=True)
	un_class_dist = un_class_dist/len(classdata)
	class_dist = dict(zip(un_class,un_class_dist))
	newpopulation=population;newclassdata=classdata;
	for i in range(maxiterations):
		if len(np.unique(newclassdata)) == 1:
			break
		else:
			population = newpopulation
			classdata = newclassdata
		population,classdata = crossover(population,classdata,model,dtype,offspringsize=len(population))
		fitness,counterfacts,counterfactsclass = evaluatefitness(population,classdata)
		population,classdata,counterfacts,counterfactsclass = selectbest(fitness,population,classdata,counterfacts,counterfactsclass,populationsize=min(10,len(population)*2)) 
		newpopulation =np.concatenate((population,np.array(counterfacts)),axis=0)
		newclassdata = np.array(list(classdata)+list(counterfactsclass))
	return (population,classdata,counterfacts,counterfactsclass)


