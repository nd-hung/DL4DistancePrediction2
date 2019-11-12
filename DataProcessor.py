import numpy as np
import pickle as cPickle
import os
import os.path
import sys
import time
import datetime
import random
import gzip
import theano

from utils import RowWiseOuterProduct, SampleBoundingBox

import DistanceUtils
import config
from config import Response2LabelName, Response2LabelType

##this file contains some functions for data processing

def PriorDistancePotential(sequence=None, paramfile=None):
	##add pairwise distance potential here
	## an example paramfile is data4contact/pdb25-pair-dist.pkl
	if not os.path.isfile(paramfile):
		print("cannot find the parameter file: ", paramfile)
		exit(-1)
	fh = open(paramfile,'rb')

	potential = cPickle.load(fh, encoding='latin1')[0].astype(np.float32)
	fh.close()
	assert (len(potential.shape) == 4)
	potentialFeature = np.zeros((len(sequence), len(sequence), potential.shape[-1]), dtype=theano.config.floatX)

	##convert AAs to integers
	ids = [ ord(AA) - ord('A') for AA in sequence ]

	##the below procedure is not very effective. What we can do is to generate a full matrix of only long-range potential using OuterConcatenate and np.choose
	##and then using the np.diagonal() function to replace near-, short- and medium-range potential in the full matrix
	for i, id0 in zip(range(len(ids)), ids):
		for j, id1 in zip(range(i+1, len(ids)), ids[i+1:]):
			if j-i<6:
				sepIndex = 0
			elif j-i < 12:
				sepIndex = 1
			elif j-i < 24:
				sepIndex = 2
			else:
				sepIndex = 3

			if id0 <=id1:
				potentialFeature[i][j]=potential[sepIndex][id0][id1]
			else:
				potentialFeature[i][j]=potential[sepIndex][id1][id0]
			potentialFeature[j][i]=potentialFeature[i][j]

	return potentialFeature

##d is a dictionary for a protein
def LocationFeature(d):
	##add one specific location feature here, i.e., posFeature[i, j]=min(1, abs(i-j)/30.0 )
	posFeature = np.ones_like(d['ccmpredZ']).astype(theano.config.floatX)
	separation_cutoff = 30
	end = min(separation_cutoff - 1, posFeature.shape[0])

	for offset in range(0, end):
		i = np.arange(0, posFeature.shape[0]-offset)
		j = i + offset
		posFeature[i, j] = offset/(1. * separation_cutoff)

	for offset in range(1, end):
		i = np.arange(offset, posFeature.shape[0])
		j = i - offset
		posFeature[i, j] = offset/(1. * separation_cutoff)

	return posFeature

def CubeRootFeature(d):
	##the cube root of the difference between the two positions, the radius of protein is related to this feature
	seqLen = len(d['sequence'])
	feature = []
	for i in range(seqLen):
		dVector = [ abs(j-i) for j in range(seqLen) ]
		feature.append(dVector)
	posFeature = np.cbrt( np.array( feature ).astype(theano.config.floatX) )
	return posFeature
	
## load native distance matrix from a file	
def LoadNativeDistMatrixFromFile(filename):
	if not os.path.isfile(filename):
		print('WARNING: cannot find the native distance matrix file: ', filename)
		exit(-1)

	fh = open(filename, 'rb')
	distMatrix = cPickle.load(fh, encoding='latin1')
	fh.close()
	return distMatrix

## load native distance matrix by protein name and location
def LoadNativeDistMatrix(name, location='pdb25-7952-atomDistMatrix/'):
	filename = os.path.join(location, name+'.atomDistMatrix.pkl')
	if not os.path.isfile(filename):
		print('WARNING: cannot find the native distance matrix file: ', filename)
		return None

	fh = open(filename, 'rb')
	distMatrix = cPickle.load(fh, encoding='latin1')
	fh.close()
	return distMatrix

def LoadDistanceFeatures(files=None, modelSpecs=None, forTrainValidation=True):
	if files is None or len(files)==0:
		print('the feature file is empty')
		exit(-1)

	fhs = [ open(file, 'rb') for file in files ]
	data = sum([ cPickle.load(fh, encoding='latin1') for fh in fhs ], [])
	[ fh.close() for fh in fhs ]

	## each protein has sequential and  pairwise features as input and distance matrix as label
	proteinFeatures = []
	counter = 0

	for d in data:
		oneprotein = dict()
		oneprotein['name'] = d['name']

		## convert the primary sequence to a one-hot encoding
		oneHotEncoding = config.SeqOneHotEncoding(d['sequence'])

		## prepare features for embedding. Currently we may embed a pair of residues or a pair of residue+secondary structure
		if config.EmbeddingUsed(modelSpecs):
			if 'Seq+SS' in modelSpecs['seq2matrixMode']:
				embedFeature = RowWiseOuterProduct(oneHotEncoding, d['SS3'])
			else:
				embedFeature = oneHotEncoding
			oneprotein['embedFeatures'] = embedFeature

		##collecting sequential features...
		seqMatrices = [ oneHotEncoding ]

		## 3-state secondary structure shall always be placed before the other features, why?
		if 'UseSS' in modelSpecs and (modelSpecs['UseSS'] is True ):
			seqMatrices.append( d['SS3'] )

		if 'UseACC' in modelSpecs and (modelSpecs['UseACC'] is True ) :
			seqMatrices.append( d['ACC'] )

		if 'UsePSSM' in modelSpecs and (modelSpecs['UsePSSM'] is True ) :
			seqMatrices.append( d['PSSM'] )

		if 'UseDisorder' in modelSpecs and modelSpecs['UseDisorder'] is True:
			seqMatrices.append(d['DISO'])

		##membrane protein specific features
		useMPSpecificFeatures = 'UseMPSpecificFeatures' in modelSpecs and (modelSpecs['UseMPSpecificFeatures'] is True)
		if useMPSpecificFeatures:
			if 'MemAcc' in d:
				seqMatrices.append(d['MemAcc'])
			else:
				print('The data does not have a feature called MemAcc')
				exit(-1)
			if 'MemTopo' in d:
				seqMatrices.append(d['MemTopo'])
			else:
				print('The data does not have a feature called MemTopo')
				exit(-1)

		## Add sequence-template similarity score here. This is used to predict distance matrix from a sequence-template alignment.
		## this is mainly used for homology modeling
		if 'UseTemplate' in modelSpecs and modelSpecs['UseTemplate']:
			#print 'Using template similarity score...'
			if 'tplSimScore' not in d:
				print('the data has no key tplSimScore, which is needed since you specify to use template information')
				exit(-1)
			if d['tplSimScore'].shape[1] != 11:
				print('The number of features for query-template similarity shall be equal to 11')
				exit(-1)
			seqMatrices.append( d['tplSimScore'] )
		seqFeature = np.concatenate( seqMatrices, axis=1).astype(np.float32)

		##collecting pairwise features...
		pairfeatures = []
		##add one specific location feature here, i.e., posFeature[i, j]=min(1, abs(i-j)/30.0 )
		posFeature = LocationFeature(d)
		pairfeatures.append(posFeature)

		cbrtFeature = CubeRootFeature(d)
		pairfeatures.append(cbrtFeature)

		if 'UseCCM' in modelSpecs and (modelSpecs['UseCCM'] is True ) :
			if 'ccmpredZ' not in d:
				print('Something must be wrong. The data for protein ', d['name'], ' does not have the normalized ccmpred feature!')
				exit(-1)
			pairfeatures.append( d['ccmpredZ'] )

		if modelSpecs['UsePSICOV'] is True:
			pairfeatures.append(d['psicovZ'])

		if 'UseOtherPairs' in modelSpecs and (modelSpecs['UseOtherPairs'] is True ):
			pairfeatures.append( d['OtherPairs'] )

		##add template-related distance matrix. This code needs modification later
		## somewhere we shall also write code to add template-related sequential features such as secondary structure?
		if 'UseTemplate' in modelSpecs and modelSpecs['UseTemplate']:
		#print 'Using template distance matrix...'
			if 'tplDistMatrix' not in d:
				print('the data for ', d['name'], ' has no tplDistMatrix, which is needed since you specify to use template information')
				exit(-1)

			## Check to make sure that we use exactly the same set of inter-atom distance information from templates
			## currently we do not use HB and Beta information from template
			apts = d['tplDistMatrix'].keys()
			assert ( set(apts) == set(config.allAtomPairTypes) )
			##assert ( set(apts) == set(config.allAtomPairTypes) or set(apts)==set(config.allLabelNames) )

			tmpPairFeatures = dict()
			for apt, tplDistMatrix in d['tplDistMatrix'].items():
				##use one flagMatrix to indicate which entries are invalid (due to gaps or disorder) since they shall be same regardless of atom pair type
				if apt == 'CaCa':
					flagMatrix = np.zeros_like(tplDistMatrix)
					np.putmask(flagMatrix, tplDistMatrix < 0, 1)
					pairfeatures.append(flagMatrix)

				strengthMatrix = np.copy(tplDistMatrix)
				np.putmask(strengthMatrix, tplDistMatrix < 3.5, 3.5)
				np.putmask(strengthMatrix, tplDistMatrix < -0.01, 50)
				strengthMatrix = 3.5 / strengthMatrix

				if config.InTPLMemorySaveMode(modelSpecs):
					tmpPairFeatures[apt] = [ strengthMatrix ]
				else:
					tmpPairFeatures[apt] = [ strengthMatrix, np.square(strengthMatrix) ]

			## here we add the tmpPairFeatures to pairfeatures in a fixed order. This can avoid errors introduced by different ordering of keys in a python dict() structure
			## python of different versions may have different ordering of keys in dict() ?
			pairfeatures.extend( tmpPairFeatures['CbCb'] )
			pairfeatures.extend( tmpPairFeatures['CgCg'] )
			pairfeatures.extend( tmpPairFeatures['CaCg'] )
			pairfeatures.extend( tmpPairFeatures['CaCa'] )
			pairfeatures.extend( tmpPairFeatures['NO'] )

		if config.InTPLMemorySaveMode(modelSpecs):
			matrixFeature = np.dstack( tuple(pairfeatures) ).astype(np.float32)
		else:
			matrixFeature = np.dstack( tuple(pairfeatures) )
			#print 'matrixFeature.shape: ', matrixFeature.shape

		oneprotein['sequence'] = d['sequence']
		oneprotein['seqLen'] = seqFeature.shape[0]
		oneprotein['seqFeatures'] = seqFeature
		oneprotein['matrixFeatures'] = matrixFeature

		##collecting labels...
		if 'atomDistMatrix' in d:
			atomDistMatrix = d['atomDistMatrix']
			oneprotein['atomLabelMatrix'] = dict()

			for response in modelSpecs['responses']:
				responseName = Response2LabelName(response)
				labelType = Response2LabelType(response)
				if responseName not in atomDistMatrix:
					print('In the raw feature data, ', d['name'], ' does not have matrix for ', responseName)
					exit(-1)

				## atomDistMatrix is the raw data, so it does not have information about labelType
				distm = atomDistMatrix[responseName]

				if labelType.startswith('Discrete'):
					subType = labelType[len('Discrete'): ]

					## no need to discretize for HB and Beta-Pairing since they are binary matrices
					if responseName.startswith('HB') or responseName.startswith('Beta'):
						oneprotein['atomLabelMatrix'][response] = distm
					else:
						labelMatrix, _, _  = DistanceUtils.DiscretizeDistMatrix(distm, config.distCutoffs[subType], subType.endswith('Plus') )
						oneprotein['atomLabelMatrix'][response] = labelMatrix

				elif labelType.startswith('LogNormal'):
					labelMatrix = DistanceUtils.LogDistMatrix(distm)
					oneprotein['atomLabelMatrix'][response] = labelMatrix

				elif labelType.startswith('Normal'):
					oneprotein['atomLabelMatrix'][response] = distm
				else:
					print('unsupported response: ', res)
					exit(-1)

		elif forTrainValidation:
			print('atomic distance matrix is needed for the training and validation data')
			exit(-1)

		##at this point, finish collecting features and labels for one protein
		proteinFeatures.append(oneprotein)

		counter += 1
		if (counter %500 ==1):
			print('assembled features and labels for ', counter, ' proteins.')

	return proteinFeatures

##this function calculates the label distribution of the training proteins and then label weight for long-, medium-, short- and near-range labels
## to assign weight to a specific label matrix, please use another function CalcLabelWeightMatrix()

## data is the trainData generated by LoadDistanceFeatures() and it has already had labels assigned
## the weight factor for a continuous distance label (i.e., regression) is a 4*1 matrix
## the weight factor for a discrete distance label (i.e., classificaton) is a 4*numLabels matrix

def CalcLabelDistributionAndWeight(data=None, modelSpecs=None):
	## weight for different ranges (long, medium, short, and near-ranges)
	if 'weight4range' not in modelSpecs:
		modelSpecs['weight4range'] = np.array([3., 2.5, 1., 0.5]).reshape((4,1)).astype(np.float32)
	else:
		modelSpecs['weight4range'].reshape((4,1)).astype(np.float32)
	print('weight for range: ', modelSpecs['weight4range'])

	## weight for 3C, that is, three distance intervals, 0-8, 8-15, and > 15
	if 'LRbias' in modelSpecs:
		modelSpecs['weight4Discrete3C']= np.multiply(config.weight43C[modelSpecs['LRbias'] ], modelSpecs['weight4range'])
	else:
		modelSpecs['weight4Discrete3C']= np.multiply(config.weight43C['mid'], modelSpecs['weight4range'])
	print('LRbias= ', modelSpecs['LRbias'], 'weight43C= ', modelSpecs['weight4Discrete3C'])

	## weight for 2C
	modelSpecs['weight4HB_Discrete2C'] = np.multiply(config.weight4HB2C, modelSpecs['weight4range'])
	modelSpecs['weight4Beta_Discrete2C'] = np.multiply(config.weight4Beta2C, modelSpecs['weight4range'])

	## weight for real value
	modelSpecs['weight4continuous'] = np.multiply(np.array([1.] * 4).reshape((4, 1)).astype(np.float32), modelSpecs['weight4range'])

	## collect all discrete label matrices
	allLabelMatrices = dict()
	for response in modelSpecs['responses']:
		name = Response2LabelName(response)
		labelType = Response2LabelType(response)
		if labelType.startswith('LogNormal') or labelType.startswith('Normal'):
			continue
		allLabelMatrices[response] = [ d['atomLabelMatrix'][response] for d in data ]

	## calculate the discrete label distribution
	allRefProbs = dict()
	for response in modelSpecs['responses']:
		name = Response2LabelName(response)
		labelType = Response2LabelType(response)
		if labelType.startswith('LogNormal') or labelType.startswith('Normal'):
			allRefProbs[response] = np.array([1.] * 4).reshape((4, 1)).astype(np.float32)
			continue

		if 'UseBoundingBox4RefProbs' in modelSpecs and (modelSpecs['UseBoundingBox4RefProbs'] is True):
		## here we sample a sub label matrix using BoundingBox to account for the real training scenario
			newLabelMatrices = []
			for lMatrix in allLabelMatrices[response]:
				bounds = SampleBoundingBox( (lMatrix.shape[0], lMatrix.shape[1]),  modelSpecs['maxbatchSize'] )
				new_lMatrix = lMatrix[ bounds[0]:bounds[2], bounds[1]:bounds[3] ].astype(np.int32)
				newLabelMatrices.append(new_lMatrix)
			allRefProbs[response] = DistanceUtils.CalcLabelProb(data = newLabelMatrices, numLabels = config.responseProbDims[labelType])
		else:
			allRefProbs[response] = DistanceUtils.CalcLabelProb(data = [ m.astype(np.int32) for m in allLabelMatrices[response] ], numLabels = config.responseProbDims[labelType])

	modelSpecs['labelRefProbs'] = allRefProbs

	##for discrete labels, we calculate their weights by inferring from the weight intialized to 3 bins: 0-8, 8-15 and >15 or -1, which makes inference easier
	modelSpecs['weight4labels'] = dict()

	for response in modelSpecs['responses']:
		name = Response2LabelName(response)
		labelType = Response2LabelType(response)

		if labelType.startswith('LogNormal') or labelType.startswith('Normal'):
		## just need to assign range weight
			modelSpecs['weight4labels'][response] = modelSpecs['weight4continuous']
			continue

		if labelType.startswith('Discrete'):
			subType = labelType[ len('Discrete'): ]

			## if the response is for HB and BetaPairing
			if subType.startswith('2C'):
				modelSpecs['weight4labels'][response] = modelSpecs['weight4' + response]
				continue

			## if the response is 3C for normal atom pairs such as Cb-Cb, Ca-Ca, Cg-Cg, CaCg, and NO
			if subType.startswith('3C'):
				modelSpecs['weight4labels'][response] = modelSpecs['weight4Discrete3C']
				continue

			## calculate label weight for 12C, 25C, and 52C for the normal atom pairs such as Cb-Cb, Ca-Ca, Cg-Cg, CaCg, and NO
			modelSpecs['weight4labels'][response] = DistanceUtils.CalcLabelWeight(modelSpecs['weight4Discrete3C'], allRefProbs[response], config.distCutoffs[subType] )
			continue

		print('unsupported response in CalcLabelDistributionAndWeight: ', response)
		exit(-1)

	return modelSpecs['labelRefProbs'], modelSpecs['weight4labels']


## this function calculates the label weight matrix for a specific label matrix
## the same label may have different weights depending on if a residue pair is in near-range, short-range, medium-range or long-range.
## labelMatrices is a dictionary and has an entry for each response. 
## This function returns a dictionary object for labelWeightMatrix
def CalcLabelWeightMatrix(LabelMatrix=None, modelSpecs=None):
	if LabelMatrix is None:
		return None

	M1s = np.ones_like(LabelMatrix.values()[0], dtype=np.int16)
	np.fill_diagonal(M1s, 0)

	LRmask = np.triu(M1s, 24) + np.tril(M1s, -24)
	MLRmask = np.triu(M1s, 12) + np.tril(M1s, -12)
	SMLRmask = np.triu(M1s, 6) + np.tril(M1s, -6)
	SRmask = SMLRmask - MLRmask
	MRmask = MLRmask - LRmask
	NRmask = M1s - SMLRmask

	for response in modelSpecs['responses']:
		if response not in modelSpecs['weight4labels']:
			print('Cannot find the weight factor tensor for response ', response)
			exit(-1)

	##the below procedure is not very effective. We shall improve it later.
	labelWeightMatrices = dict()
	for response in modelSpecs['responses']:
		##name = Response2LabelName(response)
		labelType = Response2LabelType(response)
		labelWeightMatrices[response] = np.zeros_like(LabelMatrix[response], dtype=theano.config.floatX)

		## wMatrix is a matrix with dimension 4 * numLabels
		wMatrix = modelSpecs['weight4labels'][response]
		wMatrixShape = wMatrix.shape
		assert (wMatrixShape[0] == 4)

		if labelType.startswith('Normal') or labelType.startswith('LogNormal'):
			## if the label is real value, then for each range, there is only a single weight for all the possible values
			tmpWeightMatrices = []
			for i in range(4):
				tmp = wMatrix[i][ M1s ]
				## set the weight of the entries without valid distance to 0. An invalid entry in the label matrix is indicated by a negative value,e.g., -1
				np.putmask(tmp, LabelMatrix[response] < 0, 0 )
				tmpWeightMatrices.append(tmp)
		else:
			tmpWeightMatrices = [ wMatrix[i][LabelMatrix[response]] for i in range(4) ]

		LRw, MRw, SRw, NRw = tmpWeightMatrices
		labelWeightMatrices[response] += (LRmask * LRw + MRmask* MRw + SRmask * SRw + NRmask * NRw)

	return labelWeightMatrices


## this function prepares one batch of data for training, validation and test
## data is a list of protein features and possibly labels, generated by LoadDistanceFeatures
def AssembleOneBatch( data, modelSpecs ):
	if not data:
		print('WARNING: the list of data is empty')
		return None

	numSeqs = len(data)
	seqLens = [ d['seqLen'] for d in data ]
	maxSeqLen = max( seqLens )
	minSeqLen = min( seqLens )
	#print 'maxSeqLen= ', maxSeqLen, 'minSeqLen= ', minSeqLen

	X1d = np.zeros(shape=(numSeqs, maxSeqLen, data[0]['seqFeatures'].shape[1] ), dtype = theano.config.floatX)
	X2d = np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen, data[0]['matrixFeatures'].shape[2] ), dtype = theano.config.floatX)

	X1dem = None
	if 'embedFeatures' in data[0]:
		X1dem = np.zeros(shape=(numSeqs, maxSeqLen, data[0]['embedFeatures'].shape[1] ), dtype = theano.config.floatX)

	## Y shall be a list of 3D matrices, each for one atom type. Need to revise dtype for Y
	Y = []
	if 'atomLabelMatrix' in data[0]:
		for response in modelSpecs['responses']:
			labelType = Response2LabelType(response)
			dataType = np.int16
			if not labelType.startswith('Discrete'):
				dataType = theano.config.floatX
			rValDims = config.responseValueDims[labelType]
			if rValDims == 1:
				Y.append( np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = dataType ) )
			else:
				Y.append( np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen, nValDims), dtype = dataType ) )

	## when Y is empty, weight is useless. So When Y is None, weight shall also be None
	weightMatrix = []
	if Y and modelSpecs['UseSampleWeight']:
		weightMatrix = [ np.zeros(shape=(numSeqs, maxSeqLen, maxSeqLen), dtype = theano.config.floatX) ] * len( modelSpecs['responses'] )

	## for mask
	M1d = np.zeros(shape=(numSeqs, maxSeqLen - minSeqLen ), dtype=np.int8 )
	M2d = np.zeros(shape=(numSeqs, maxSeqLen - minSeqLen, maxSeqLen ), dtype=np.int8 )

	for j in range(len(data) ):
		seqLen = data[j]['seqLen']
		X1d[j, maxSeqLen - seqLen :, : ] = data[j]['seqFeatures']
		X2d[j, maxSeqLen - seqLen :, maxSeqLen - seqLen :, : ] = data[j]['matrixFeatures']
		M1d[j, maxSeqLen - seqLen : ].fill(1)
		M2d[j, maxSeqLen - seqLen :, maxSeqLen - seqLen : ].fill(1)

		if X1dem is not None:
			X1dem[j, maxSeqLen - seqLen :, : ] = data[j]['embedFeatures']

		if Y:
			for y, response in zip(Y, modelSpecs['responses']):
				if len(y.shape) == 3:
					y[j, maxSeqLen-seqLen :, maxSeqLen-seqLen : ] = data[j]['atomLabelMatrix'][response]
				else:
					y[j, maxSeqLen-seqLen :, maxSeqLen-seqLen:, ] = data[j]['atomLabelMatrix'][response]
		if weightMatrix:
			## we calculate the labelWeightMatrix here
			labelWeightMatrix = CalcLabelWeightMatrix(data[j]['atomLabelMatrix'], modelSpecs)
			for w, at in zip( weightMatrix, modelSpecs['responses']):
				w[j, maxSeqLen - seqLen :, maxSeqLen - seqLen : ] = labelWeightMatrix[at]

	onebatch = [X1d, X2d, M1d, M2d]

	if X1dem is not None:
		onebatch.append(X1dem)

	onebatch.extend(Y)
	onebatch.extend(weightMatrix)

	return onebatch

##split data into minibatch, each minibatch numDataPoints data points
def SplitData2Batches(data=None, numDataPoints=1000000, modelSpecs=None):

	if data is None:
		print('Please provide data for process!')
		sys.exit(-1)

	if numDataPoints < 10:
		print('Please specify the number of data points in a minibatch')
		sys.exit(-1)

	## sort proteins by length from large to small
	data.sort(key=lambda x: x['seqLen'], reverse=True)

	##seqDataset stores the resultant data
	batches = []
	names = []

	i = 0
	while i < len(data):
		currentSeqLen = data[i]['seqLen']
		numSeqs = min( len(data) - i, max(1, numDataPoints/np.square(currentSeqLen) ) )
		#print 'This batch contains ', numSeqs, ' sequences'
		names4onebatch = [ d['name'] for d in data[i: i+numSeqs] ]
		oneBatch = AssembleOneBatch( data[i : i+numSeqs], modelSpecs )
		batches.append(oneBatch)
		names.append(names4onebatch)
		i += numSeqs

	return batches, names

def CalcAvgWeightPerBatch(batches, modelSpecs):
	if not modelSpecs['UseSampleWeight']:
		return None

	numResponses = len(modelSpecs['responses'])
	allWeights = []

	for b in batches:
		oneBatchWeight = []
		for wMatrix in b[-numResponses: ]:
			bounds = SampleBoundingBox( (wMatrix.shape[1], wMatrix.shape[2]), modelSpecs['maxbatchSize'] )
			new_wMatrix = wMatrix[:,  bounds[0]:bounds[2], bounds[1]:bounds[3] ]
			wSum = np.sum(new_wMatrix) 
			oneBatchWeight.append(wSum)

		allWeights.append(oneBatchWeight)

	avgWeights = np.average(allWeights, axis=0)

	modelSpecs['batchWeightBase'] = np.array(avgWeights).astype(theano.config.floatX)

	maxWeights = np.amax(allWeights, axis=0)
	minWeights = np.amin(allWeights, axis=0)

	## reutrn the maximum deviation
	return maxWeights/avgWeights, minWeights/avgWeights
