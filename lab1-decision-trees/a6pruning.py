import dtree
import monkdata as m
import partition
import numpy as np
import matplotlib.pyplot as plt

splits = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
noOfRep = 500

def pruneTree(dataset, split, noOfRep, testdata):
	fractionCorrect=0
	pruned = []
	
	for rep in range(noOfRep):	
		#Estimate same on new rand split
		monkTrain, monkVal = partition.partition(dataset, split)

		#Build tree, based on new training data split
		prevTree = dtree.buildTree(monkTrain, m.attributes)
		
		prunedTrees = dtree.allPruned(prevTree)
		maxCorrect = 0
		bestTreeIndex = 0

		for i in range(0, len(prunedTrees)):
			pruneIn = []
			newPrunedTree=prunedTrees[i]

			#Check performance on validation dataset
			fractionCorrect=dtree.check(newPrunedTree, monkVal)
			
			if fractionCorrect > maxCorrect:
				maxCorrect = fractionCorrect
				bestTreeIndex = i
			
		bestTree = prunedTrees[bestTreeIndex]
		
		#Here we use actual test data to measure performance, do not build model on this though
		performance = dtree.check(bestTree, testdata)
		error = 1-performance
		pruned.append(error)
	
	#print(("MEAN:",np.mean(pruned)))
	
	
	return np.mean(pruned), np.var(pruned)

def plotMonk(dataset, testdata, title, plotMean):
	plt.figure(title)

	for x in range(5):
		meanErrors = []
		varianceErrors = []
		for split in range(len(splits)):
			meanError, varianceError = pruneTree(dataset, splits[split], noOfRep, testdata)
			meanErrors.append(meanError)
			varianceErrors.append(varianceError)
		print(title)

		if(plotMean):
			plt.plot(splits, meanErrors, marker='o')
			plt.ylabel("Mean error")
		else:
			plt.plot(splits, varianceErrors, marker='o')
			plt.ylabel("Variance")
	
	plt.title(title)
	plt.legend(['Batch 1', 'Batch 2', 'Batch 3', 'Batch 4', 'Batch 5'], loc='upper left')

def main():
	#Create our main trees
	tree1 = dtree.buildTree(m.monk1, m.attributes)
	tree2 = dtree.buildTree(m.monk2, m.attributes)
	tree3 = dtree.buildTree(m.monk3, m.attributes)

	#PLOT MONK1 - MEAN AND VARIANCE
	dataset = m.monk1
	testdata = m.monk1test

	#Overall error on test set
	benchmarkTreeMonk1 = dtree.buildTree(dataset,m.attributes)
	#print("BENCHMARK: ", 1-dtree.check(benchmarkTreeMonk1, testdata))

	plotMonk(dataset, testdata, "Mean error vs. Fraction - MONK1\n500 runs in each batch", True)
	plotMonk(dataset, testdata, "Variance vs. Fraction - MONK1\n500 runs in each batch", False)

	#PLOT MONK3 - MEAN AND VARIANCE
	dataset = m.monk3
	testdata = m.monk3test

	#Overall error on test set
	benchmarkTreeMonk3 = dtree.buildTree(dataset,m.attributes)
	#print("BENCHMARK: ", 1-dtree.check(benchmarkTreeMonk3, testdata))

	plotMonk(dataset, testdata, "Mean error vs. Fraction - MONK3\n500 runs in each batch", True)
	plotMonk(dataset, testdata, "Variance vs. Fraction - MONK3\n500 runs in each batch", False)
	plt.show()

main()
