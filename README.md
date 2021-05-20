# Genfact
## _A fast counterfactual generator for Causal analysis_

[![DOI](https://zenodo.org/badge/368590702.svg)](https://zenodo.org/badge/latestdoi/368590702)


## Background
Counterfactual examples are samples which are minimally modified with respect to the original sample to alter the predicted value by a model. Thus, counterfactual explanations provide statements as smallest changes required to alter certain predicted value or decision and it proves to be quite useful in finding causal relationships in data. _Genfact_ is a model agnostic and gradient-free optimization method for generating counterfactuals. It can generate multiple counterfactuals at once and can-do amortized inference, thus making the process fast. Given a dataset it can find counterfactuals pairs closest to each other and the pairs may not exist in the original dataset. This feature is useful in this context as the given dataset used for generating counterfactuals may not contain enough samples around the classification boundary, but _Genfact_ can generate samples around the boundary. Reference paper can be found [here](https://www.researchgate.net/publication/351697701_Counterfactual_causal_analysis_on_structured_data).

## Features

- Generate fact and counterfact pairs on arbitrary relational dataset
- Fast processing
- Evaluate generated counterfactuals based on entropy and fitness
- Preloaded with encoded Facebook test data 
- Inbuilt data preparation

## Usage
Follow the step by step guide to get started with.

## Installation
Either download from this github repo or install through pip
```sh
$ pip install genfact
```
## Using it
Import the library.
```Python
import genfact as gf
```
- Load the data in a pandas dataframe. Each attribute can be categorical or continous. Assign each categorical values a numeric code. Let the dataframe variable be data_df.
- Prepare a list containing the datatypes of each attribute in the data. A categorical attribute should be represented as 'cat' and a continous by 'con'. Let the list if stored in a variable dtype.
- Identify the targetvariable index and assign it to targetclass_idx

To generate counterfactuals run the following function
```Python
factuals,counterfactuals,factclass,cfactclass,classdistribution = gf.generate_counterfactuals(data_df,dtype,targetclass_idx, model=None, C=15, clustsize = 20, datafraction = 0.4, maxiterations = 10)
```
**Hyperparameters**
- model represents the predictive model for featuredata and classdata. If None is supplied a Random forest model is trained
- C represents the number of classes the target variable will be divided if it is a continous one. Please note If duplicates are present the actual number of buckets formed will be lesser than C.
- clustsize represents number of clusters to be generated using the feature data.
- datafraction represents the fraction of data that will be processed to generate counterfactuals
- maxiterations represents the number of iterations the genetic algorithm will run

The given values are the default values of the hyperparameters

**Output**
- factuals contain an array of facts from the feature data
- counterfactuals contain an array of counterfacts for each facts
- factclass contain an array of predicted class for each facts
- cfactclass contain an array of predicted class for each counterfacts
- classdistribution contain a dataframe representing the boundary of each classes if the target attribute in the input dataset is continous. Else it returns None.

## Example
The following example shows how to run the counterfactual generator using the test data and evaluate them.

```Python
import genfact as gf
### load test data
data_df,dtype,targetclass_idx = gf.load_data()
factuals,counterfactuals,factclass,cfactclass,classdistribution = gf.generate_counterfactuals(data_df,dtype,targetclass_idx)
entropy,fitness = gf.evaluate_counterfactuals(factuals,counterfactuals,factclass,cfactclass)
```
