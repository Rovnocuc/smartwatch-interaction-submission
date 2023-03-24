#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Model Trainer

This class loads the pretrained SVM model and finishes the training with gathered user data.

Author: REDACTED
E-mail: REDACTED
2022
'''

import copy
from datetime import datetime
import logging
import os
import random
import string
from os.path import join
from time import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap.umap_ as umap
from matplotlib.colors import Normalize
from sklearn import svm, neighbors, tree, ensemble, neural_network, naive_bayes, discriminant_analysis
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.stats import expon

# Minimum and maximum values derived from measured data.
MIN_VALUES = {
    'accX': -40,
    'accY': -50,
    'accZ': -40,
    'gyrX': -20,
    'gyrY': -20,
    'gyrZ': -20
}

MAX_VALUES = {
    'accX': 80,
    'accY': 40,
    'accZ': 40,
    'gyrX': 30,
    'gyrY': 20,
    'gyrZ': 20
}


class ModelTrain():
    '''
    Loads the pretrained SVM model and finishes the training with gathered user data.
    The user data is in following format:
        accX, accY, accZ, gyrX, gyrY, gyrZ, annotation
        6x float, string
        6x float, string
        ...
    '''

    def convertDataset(self, dataframe: pd.DataFrame, windowSize: int, balance: bool = True, repeats: float = 1.0) -> pd.DataFrame:
        '''
        Cuts the original dataset into windows of size `windowSize` and balances the dataset,
        so there is not a majority of non-gestures.

        Returns a new dataframe with feature vectors ready for training.
        '''

        dataframe = self._createDataset(dataframe, windowSize)
        # dataframe = self._balanceDataset(dataframe)
        if balance:
            dataframe = self._balanceEnhanceDataset(dataframe, repeats)

        return dataframe

    def _cutWindow(self, dataframe: pd.DataFrame, start: int, windowSize: int) -> list:
        '''
        Extracts first six values from every row of the dataframe and concatenates it to the list.
        The final feature vector is annotated by the majority of the all annotations occured.
        '''

        if start < 0:
            raise IndexError('start: Cut window starts from negative value.')
        if windowSize < 1:
            raise ValueError(
                f'windowSize: Invalid window size of {windowSize}.')

        # Check for dataframe length, if cutting is even possible.
        if dataframe.shape[0] < start+windowSize-1:
            raise IndexError('Cut window is out of range of the dataframe.')

        subframe = dataframe[start:start+windowSize]
        label = subframe['annotation'].mode().at[0]

        arr = subframe.iloc[:, :-1].to_numpy()
        # Flatten by column.
        flat = list(arr.flatten('F'))
        flat.append(label)

        return flat

    def _createDataset(self, dataframe: pd.DataFrame, windowSize: int) -> pd.DataFrame:
        '''
        Creates dataset of annotated feature vectors.
        Each feature vector is concatenated window of six parameters at a time,
        followed by the annotation.

        Labels in form of strings are swapped for integers.

        `dataframe`: The source dataframe, containing six columns of smartwatch data and a single column of annotation.
        `windowSize`: The length of the sliding window. Minimum is 1.
        '''
        if windowSize < 1:
            raise ValueError(
                f'windowSize: Invalid window size of {windowSize}.')

        annotationsToNumbersMapping = {
            'delete': -1,
            '#': 0,
            'up': 1,
            'down': 2,
            'left': 3,
            'right': 4,
            'clockwise': 5,
            'counterclockwise': 6,
            'punch': 7
        }

        dataframe.replace(annotationsToNumbersMapping, inplace=True)

        lst = []
        for i in range(dataframe.shape[0]-windowSize+1):
            lst.append(self._cutWindow(dataframe, i, windowSize))

        return pd.DataFrame(lst)

    def _enhanceDataset(self, dataframe: pd.DataFrame, alpha: float, repeats: int = 1) -> pd.DataFrame:
        '''
        Appends the original data with a noisy copy.
        The amount of noise is computed from the range of every variable across the dataset.
        The noise has an uniform distribution with a boundary given by `alpha` times range of the variable.

        `dataframe`: The original dataset.
        `alpha`: The rate of noise.
        `repeats`: How many noisy copies are appended.
        '''
        if alpha <= 0:
            raise ValueError(
                f'Invalid alpha value of {alpha}.')
        if repeats < 1:
            raise ValueError(
                f'Invalid repeats value of {repeats}.')

        mx = dataframe.iloc[:, :-1].max(axis=0).to_numpy()
        mn = dataframe.iloc[:, :-1].min(axis=0).to_numpy()
        varRange = abs(mn-mx)

        originalValues = dataframe.iloc[:, :-1].to_numpy()
        originalLabels = dataframe.iloc[:, -1].to_numpy()

        alphaRange = (varRange*alpha)[None, ...] + \
            np.zeros(dataframe.shape[0])[:, None]

        datasetShape = copy.copy(dataframe.shape)

        for i in range(repeats):
            ran = np.random.rand(
                datasetShape[0], datasetShape[1]-1) - 1/2
            newValues = originalValues + np.multiply(alphaRange, ran)
            newValues = np.concatenate(
                (newValues, originalLabels[:, None]), axis=1)
            dataNp = np.concatenate((dataframe.to_numpy(), newValues), axis=0)
            dataframe = pd.DataFrame(dataNp)

        dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    def _balanceDataset(self, dataframe: pd.DataFrame, balanceType: str = 'inner') -> pd.DataFrame:
        '''
        Removes randomly a number of samples of the most numeous label to balance dataset.
        The resulting number of previously abundant label will be the maximum of all other labels.
        It is used on already windowed dataset.

        `fullBalance`:
            `inner` removes samples randomly from every label except the least frequent one. The resulting dataset has number of samples equal to number of least frequent label times number of labels.

            `cut` removes randomly non-gestures, so their final amount is the same as the most frequent gesture.
        '''

        if balanceType == 'inner':
            counts = dataframe.iloc[:, -1].value_counts()
            minCount = counts.min()
            for label in dataframe.iloc[:, -1].unique():
                idx = dataframe.index[dataframe.iloc[:, -1] == label].tolist()
                dataframe.drop(random.sample(
                    idx, counts[label]-minCount), inplace=True)

            dataframe.reset_index(drop=True, inplace=True)

            return dataframe

        if balanceType == 'cut':
            # Compute most common label.
            counts = dataframe.iloc[:, -1].value_counts()
            maxCount = counts.max()
            maxLabel = counts.idxmax()
            # Exclude the most common label.
            counts.drop(maxLabel, inplace=True)
            # Get max of the rest.
            goalCount = int(counts.max())

            # Get indexes of every row with most common index.
            idx = dataframe.index[dataframe.iloc[:, -1] == maxLabel].tolist()
            # Drop indexes at random so the goal is achieved.
            dataframe.drop(random.sample(
                idx, maxCount-goalCount), inplace=True)

            dataframe.reset_index(drop=True, inplace=True)

            return dataframe

    def _balanceEnhanceDataset(self, dataframe: pd.DataFrame, repeats: float = 2.0, alpha: float = 0.1):
        counts = dataframe.iloc[:, -1].value_counts()
        for idx, val in counts.items():
            if idx == 0 or idx == -1:
                continue
            else:
                goalCount = int(val*repeats)
                break
        for label, count in zip(counts.index, counts):
            if label == -1:
                idx = dataframe.index[dataframe.iloc[:, -1]
                                      == label].tolist()
                dataframe.drop(idx, inplace=True)
                dataframe.reset_index(drop=True, inplace=True)
                continue
            if count > goalCount:
                # Get indexes of every row with label.
                idx = dataframe.index[dataframe.iloc[:, -1]
                                      == label].tolist()
                # Drop random rows, to reduce the amount of label.
                dataframe.drop(random.sample(
                    idx, count-goalCount), inplace=True)
                dataframe.reset_index(drop=True, inplace=True)
            if count < goalCount:
                # Get indexes of every row with label.
                idx = dataframe.index[dataframe.iloc[:, -1]
                                      == label].tolist()
                mx = dataframe.iloc[:, :-1].max(axis=0).to_numpy()
                mn = dataframe.iloc[:, :-1].min(axis=0).to_numpy()
                varRange = abs(mn-mx)

                alphaRange = (varRange*alpha)[None, ...] + \
                    np.zeros(goalCount-count)[:, None]
                # Fill numpy array with copies of the original datapoints.
                subset = dataframe.loc[random.choices(
                    idx, k=goalCount-count)].to_numpy()
                subset = subset[:, :-1]
                ran = np.random.rand(
                    subset.shape[0], subset.shape[1]) - 1/2
                subset = subset + np.multiply(alphaRange, ran)
                subset = pd.DataFrame(subset)
                subset[subset.shape[1]] = label
                dataframe = pd.concat((dataframe, subset))
                dataframe.reset_index(drop=True, inplace=True)

        return dataframe

    def normalize(self, dataframe: pd.DataFrame, minValues: dict, maxValues: dict):
        windowSize = int((dataframe.shape[1]-1)/6)

        minList = []
        maxList = []
        for name in minValues:
            minList.extend([minValues[name]]*windowSize)
            maxList.extend([maxValues[name]]*windowSize)
        minList = np.array(minList)
        maxList = np.array(maxList)
        rangeList = maxList-minList

        arr = dataframe.iloc[:, :-1].to_numpy()
        arr.clip(minList, maxList, arr)
        arr = (arr-minList)/rangeList

        return pd.concat((pd.DataFrame(arr), dataframe.iloc[:, -1]), axis=1)

    def _tuneHyperparametersSVM(self, X: pd.DataFrame, y: pd.DataFrame, C: list[float], gamma: list[float]) -> tuple[float, float]:
        '''
        Tune hyperparameters for the SVM. Perform a grid search over parameters C and gamma.
        The result is dumped into json file and also visualized as a heatmap.

        `X`: The training dataset.
        `y`: Labels of the training dataset.
        `C`: List of parameters.
        `gamma`: List of parameters.
        '''

        # Perform a grid search over C and gamma parameters.
        params = {'C': C, 'gamma': gamma}
        svc = svm.SVC()
        clf = GridSearchCV(svc, params, n_jobs=5, verbose=3)
        clf.fit(X, y)

        print(clf.best_params_)

        self._visualizeHyperparameters(C, gamma, clf)

    def _visualizeHyperparameters(self, CRange, gammaRange, clf) -> None:
        '''
        https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

        Utility function to move the midpoint of a colormap to be around
        the values of interest.
        '''
        class MidpointNormalize(Normalize):
            def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

            def __call__(self, value, clip=None):
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

        # Draw heatmap of the validation accuracy as a function of gamma and C

        scores = clf.cv_results_["mean_test_score"].reshape(
            len(CRange), len(gammaRange))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(
            scores,
            interpolation="nearest",
            cmap=plt.cm.hot,
            norm=MidpointNormalize(vmin=0.2, midpoint=0.92),
        )
        plt.xlabel("gamma")
        plt.ylabel("C")
        plt.colorbar()
        plt.xticks(np.arange(len(gammaRange)), gammaRange, rotation=45)
        plt.yticks(np.arange(len(CRange)), CRange)
        plt.title("Validation accuracy")
        plt.show()

    def loadData(self, datasetFolder: str, datasetNames: list[str], testFolder: str = None, testNames: list[str] = None):
        '''
        Loads datasets from folder, concatenates them and splits them to train and test datasets.

        Parameters
        ----------

        datasetFolder: Path to a folder containing csv datasets of individual subjects.

        datasetNames: List of filenames (without extension) of datasets used for training.

        testFolder: Path to a folder containing csv datasets of test subjects. `None` if you want to perform automatic train/test split.

        testNames: List of filenames (without extension) of datasets used for testing. `None` if you want to perform automatic train/test split.
        '''
        data = None
        for name in datasetNames:
            if data is None:
                data = pd.read_csv(
                    join(datasetFolder, name+'.csv'), header=None)
            else:
                data = pd.concat(
                    [data, pd.read_csv(join(datasetFolder, name+'.csv'), header=None)])

        X = data.iloc[:, :-1]
        y = data.iloc[:, -1].astype(int)

        if testFolder is None and testNames is None:
            XTrain, XTest, yTrain, yTest = train_test_split(X, y)
        else:
            testData = None
            for name in testNames:
                if testData is None:
                    testData = pd.read_csv(
                        join(testFolder, name+'.csv'), header=None)
                else:
                    testData = pd.concat(
                        [testData, pd.read_csv(join(testFolder, name+'.csv'), header=None)])
            XTrain = X
            yTrain = y
            XTest = testData.iloc[:, :-1]
            yTest = testData.iloc[:, -1].astype(int)

        return XTrain, yTrain, XTest, yTest

    def evaluateModel(self, dataset: pd.DataFrame, modelPath: string) -> None:
        '''
        Loads the `.pkl` model and predicts the labels of the dataset.
        '''
        svc = joblib.load(modelPath)
        return svc.predict(dataset)

    def visualizePCA(self, dataset: pd.DataFrame) -> None:
        '''
        Performs PCA on the provided dataset and shows it in a 2D visualization.
        '''

        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1].astype(int)
        y.rename('target', inplace=True)

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents, columns=[
                                   'principal component 1', 'principal component 2'])

        finalDf = pd.concat([principalDf, y], axis=1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        colors = ['#000000', '#5D8283', '#334748', '#703644',
                  '#C78743', '#9CAAAB', '#DBCDBD', '#967163']
        for target, color in zip(range(-1, 7, -1), colors):
            indicesToKeep = finalDf['target'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                       finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
        ax.legend(['non-gesture', 'up', 'down', 'left', 'right',
                   'clockwise', 'counterclockwise', 'punch'])
        ax.grid()

        plt.show()

    def visualizeUMAP(self, dataset) -> None:
        '''
        Shows a 2D UMAP of the dataset.

        Accepts either dataframe of windowed feature vectors with label as a last column,
        or a dictionary of multiple datasets. The keys in the dictionary are names of the participants
        and they will have different marks in the final visualizations.
        '''

        if type(dataset) == pd.DataFrame:
            X = dataset.iloc[:, :-1]
            y = dataset.iloc[:, -1].astype(int)
            z = pd.Series('untitled', index=range(y.shape[0]), name='name')
            names = ['untitled']
        else:
            X = None
            y = None
            z = None
            names = dataset.keys()
            for name in dataset:
                if type(X) == type(None):
                    X = dataset[name].iloc[:, :-1]
                    y = dataset[name].iloc[:, -1].astype(int)
                    z = pd.Series(name, index=range(y.shape[0]), name='name')
                else:
                    X = pd.concat([X, dataset[name].iloc[:, :-1]])
                    y = pd.concat([y, dataset[name].iloc[:, -1].astype(int)])
                    zLen = dataset[name].shape[0]
                    z = pd.concat(
                        [z, pd.Series(name, index=range(zLen))])

        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)
        y.rename('target', inplace=True)
        z.reset_index(drop=True, inplace=True)
        z.rename('name', inplace=True)

        reducer = umap.UMAP()

        embedding = pd.DataFrame(reducer.fit_transform(X))

        finalDf = pd.concat([embedding, y, z], axis=1)

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('UMAP', fontsize=20)
        colors = ['#666666',
                  '#7fc97f',
                  '#beaed4',
                  '#fdc086',
                  '#ffff99',
                  '#386cb0',
                  '#f0027f',
                  '#bf5b17']
        markers = ['o', 'v', '^', '<', '>', '8', 's',
                   'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
        for target, color in zip(range(8), colors):
            for name, marker in zip(names, markers):
                indicesToKeep = (finalDf['target'] == target) & (
                    finalDf['name'] == name)
                ax.scatter(finalDf.loc[indicesToKeep, 0],
                           finalDf.loc[indicesToKeep, 1], c=color, s=40, marker=marker)

        gestures = ['non-gesture', 'up', 'down', 'left', 'right',
                    'clockwise', 'counterclockwise', 'punch']
        legend = []
        for gesture in gestures:
            for name in names:
                legend.append(name + ' ' + gesture)

        ax.legend(legend)
        ax.grid()

        plt.show()


if __name__ == "__main__":

    logging.basicConfig(filename=f'{datetime.now().strftime("%Y-%m-%d-%H-%M")}.log',
                        encoding='utf-8', level=logging.INFO)

    mt = ModelTrain()

    '''
    Create windowed dataset.
    '''
    # # participants = map(lambda x: x.split('.')[0], os.listdir('data_2'))
    # participants = ['banana_4']
    # for name in participants:
    #     dataframe = pd.read_csv(join('data_3', name + '.csv'))
    #     dataframe = mt.convertDataset(dataframe, 20)
    #     dataframe = mt.normalize(dataframe, MIN_VALUES, MAX_VALUES)
    #     dataframe.to_csv(
    #         join('data_3', name + '_balanced_win20.csv'), header=False, index=False)

    '''
    Test different models.
    '''

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # names = [
    #     "Nearest Neighbors",
    #     "Linear SVM",
    #     "RBF SVM",
    #     "Gaussian Process",
    #     "Decision Tree",
    #     "Random Forest",
    #     "Neural Net",
    #     "AdaBoost",
    #     "Naive Bayes",
    #     "QDA",
    # ]

    # classifiers = [
    #     neighbors.KNeighborsClassifier(3),
    #     svm.SVC(kernel="linear", C=0.025),
    #     svm.SVC(gamma=2, C=1),
    #     tree.DecisionTreeClassifier(max_depth=5),
    #     ensemble.RandomForestClassifier(
    #         max_depth=5, n_estimators=1000, max_features=5),
    #     neural_network.MLPClassifier(alpha=1, max_iter=1000),
    #     ensemble.AdaBoostClassifier(),
    #     naive_bayes.GaussianNB(),
    #     discriminant_analysis.QuadraticDiscriminantAnalysis(),
    # ]

    # logging.info('Testing various classifiers.')
    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # for name, clf in zip(names, classifiers):
    #     print(f'{"_"*len(name)}\n{name}\n{"‾"*len(name)}')
    #     logging.info(f'Training {name}.')
    #     start = time()
    #     clf.fit(XTrain, yTrain)
    #     delta = time() - start

    #     print(f'\tTime to train: {int(delta/60)}:{int((delta/60%1)*60):02d}')
    #     logging.info(
    #         f'Time to train {name}: {int(delta/60)}:{int((delta/60%1)*60):02d} s.')

    #     score = clf.score(XTest, yTest)

    #     print(f'\tScore: {score:.3f}')
    #     logging.info(f'Score of {name}: {score:.3f}.')

    #     yPred = clf.predict(XTest)
    #     ConfusionMatrixDisplay.from_predictions(yTest, yPred)
    #     plt.savefig(name+'_conf_matrix.png')

    # joblib.dump(model, 'data_1_balanced_win20/model_banana_c1_y0001.pkl')

    '''
    Tune hyperparameters for SVM -- random search.
    '''
    # logging.info(
    #     'Tune hyperparameters of SVM, using Randomized Parameter Optimization.')

    # trainFolder = 'data_2_balanced_win20'
    # trainNames = ['coconut_balanced_win20', 'grapefruit_balanced_win20',
    #               'avocado_balanced_win20', 'lime_balanced_win20', 'apricot_2_balanced_win20']
    # testFolder = 'data_2_balanced_win20'
    # testNames = ['banana_2_balanced_win20']

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # params = {'C': expon(scale=100), 'gamma': expon(scale=.1),
    #           'kernel': ['rbf']}
    # svc = svm.SVC()
    # clf = RandomizedSearchCV(svc, params, n_iter=20, n_jobs=5, verbose=3)
    # clf.fit(XTrain, yTrain)
    # print(clf.best_params_)
    # df = pd.DataFrame(clf.cv_results_)
    # df.to_csv('svm_param_search_results.csv')

    '''
    Tune hyperparameters for SVM -- grid search.
    '''
    # logging.info(
    #     'Tune hyperparameters of SVM, using Grid Search.')

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # params = {'C': [10, 20, 40, 80, 160], 'gamma': [.1, .2, .4, .8, 5, 10],
    #           'kernel': ['rbf']}
    # svc = svm.SVC()
    # clf = GridSearchCV(svc, params, n_jobs=5, verbose=3)
    # clf.fit(XTrain, yTrain)
    # print(clf.best_params_)
    # df = pd.DataFrame(clf.cv_results_)
    # df.to_csv('svm_param_search_results.csv')

    '''
    Tune hyperparameters for kNN.
    '''
    # logging.info(
    #     'Tune hyperparameters of kNN, using Grid Search.')

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # params = {'n_neighbors': [1, 3, 5, 7, 9, 11], 'p': [1, 2]}
    # logging.info(f'Grid parameters: {params}.')

    # knn = neighbors.KNeighborsClassifier()
    # clf = GridSearchCV(knn, params, n_jobs=5, verbose=3)
    # clf.fit(XTrain, yTrain)
    # print(clf.best_params_)
    # df = pd.DataFrame(clf.cv_results_)
    # df.to_csv('knn_param_search_results.csv')

    '''
    Tune hyperparameters for Random Forest.
    '''

    # logging.info(
    #     'Tune hyperparameters of Random Forest, using Randomized Parameter Optimization.')

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # params = {'n_estimators': list(range(100, 400)),
    #           'max_features': list(range(1, 10))}
    # rfc = ensemble.RandomForestClassifier()
    # clf = RandomizedSearchCV(rfc, params, n_iter=40, n_jobs=5, verbose=3)

    # clf.fit(XTrain, yTrain)
    # print(clf.best_params_)
    # df = pd.DataFrame(clf.cv_results_)
    # df.to_csv('rfc_param_search_results_random_2.csv')

    '''
    Evaluate SVM model.
    '''

    # logging.info('Train SVM model.')

    # trainFolder = 'data_2_balanced_win20'
    # trainNames = ['coconut_balanced_win20', 'grapefruit_balanced_win20',
    #               'avocado_balanced_win20', 'lime_balanced_win20', 'apricot_2_balanced_win20']
    # testFolder = 'data_3'
    # testNames = ['banana_4_win20']

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # model = svm.SVC(C=10, gamma=1)
    # logging.info(f'SVM params: C={model.C}, gamma={model.gamma}')

    # model.fit(XTrain, yTrain)
    # # joblib.dump(model, 'data_2_balanced_win20/model_banana_2_c10_y1.pkl')

    # score = model.score(XTest, yTest)

    # print(f'\tScore: {score:.3f}')
    # logging.info(f'Score of SVM: {score:.3f}.')

    # yPred = model.predict(XTest)
    # ConfusionMatrixDisplay.from_predictions(yTest, yPred)
    # plt.savefig('svm_banana_4_conf_matrix.png')

    '''
    Evaluate kNN model.
    '''

    # logging.info('Train kNN model.')

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # knn = neighbors.KNeighborsClassifier()
    # logging.info(f'kNN params: n_neighbors={knn.n_neighbors}, p={knn.p}')

    # knn.fit(XTrain, yTrain)
    # score = knn.score(XTest, yTest)

    # print(f'\tScore: {score:.3f}')
    # logging.info(f'Score of kNN: {score:.3f}.')

    # yPred = knn.predict(XTest)
    # ConfusionMatrixDisplay.from_predictions(yTest, yPred)
    # plt.savefig('optimized_knn_conf_matrix.png')

    '''
    Evaluate Random Forest model.
    '''

    # logging.info('Train Random Forest model.')

    # trainFolder = 'data_1_balanced_win20'
    # trainNames = ['apple_balanced_win20', 'apricot_balanced_win20', 'lemon_balanced_win20',
    #               'pear_balanced_win20', 'orange_balanced_win20', 'blueberry_balanced_win20', 'peach_balanced_win20']
    # testFolder = 'data_1_balanced_win20'
    # testNames = ['banana_balanced_win20']

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # rfc = ensemble.RandomForestClassifier(n_estimators=400, max_features=6)
    # logging.info(
    #     f'rfc params: n_estimators={rfc.n_estimators}, max_features={rfc.max_features}')

    # rfc.fit(XTrain, yTrain)
    # score = rfc.score(XTest, yTest)

    # print(f'\tScore: {score:.3f}')
    # logging.info(f'Score of rfc: {score:.3f}.')

    # yPred = rfc.predict(XTest)
    # ConfusionMatrixDisplay.from_predictions(yTest, yPred)
    # plt.savefig('optimized_rfc_conf_matrix.png')

    '''
    Test SVM predict_proba
    '''

    # logging.info('Train SVM model for predicting all probabilities.')

    # trainFolder = 'data_2_balanced_win20'
    # trainNames = ['coconut_balanced_win20', 'grapefruit_balanced_win20',
    #               'avocado_balanced_win20', 'lime_balanced_win20', 'apricot_2_balanced_win20']
    # testFolder = 'data_2_balanced_win20'
    # testNames = ['banana_2_balanced_win20']

    # logging.info(f'Training data source folder: {trainFolder}')
    # logging.info(f'Training subjects: {trainNames}.')
    # logging.info(f'Test subjects: {testNames}.')

    # XTrain, yTrain, XTest, yTest = mt.loadData(
    #     trainFolder, trainNames, testFolder, testNames)

    # model = svm.SVC(C=10, gamma=1, probability=True)
    # logging.info(f'SVM params: C={model.C}, gamma={model.gamma}')

    # model.fit(XTrain, yTrain)
    # joblib.dump(
    #     model, 'data_2/model_banana_2_proba_c10_y1.pkl')

    # score = model.score(XTest, yTest)

    # print(f'\tScore: {score:.3f}')
    # logging.info(f'Score of SVM: {score:.3f}.')

    # yPred = model.predict_proba(XTest)
    # np.savetxt('predict_proba_banana_4.csv', yPred, delimiter=',')

    '''
    Visualize confusion matrix
    '''
    data = pd.read_csv(
        'data_2_balanced_win20/banana_2_balanced_win20.csv', header=None)
    y_pred = mt.evaluateModel(data.iloc[:, :-1],
                              'data_2_balanced_win20/model_banana_2_c10_y1.pkl')
    y_true = data.iloc[:, -1]
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

    '''
    PCA visualization.
    '''
    # data = pd.read_csv(join('data_1', 'banana_windowed.csv'), header=None)
    # mt.visualizePCA(data)

    '''
    UMAP visualization.
    '''
    # data = {'banana': pd.read_csv(join('data_1_balanced_win20', 'banana_balanced_win20.csv'), header=None),
    #         'banana_flash': pd.read_csv(join('data_1_balanced_win20', 'banana_flash_balanced_win20.csv'), header=None),
    #         'lemon': pd.read_csv(join('data_1_balanced_win20', 'lemon_balanced_win20.csv'), header=None),
    #         'apple': pd.read_csv(join('data_1_balanced_win20', 'apple_balanced_win20.csv'), header=None),
    #         'orange': pd.read_csv(join('data_1_balanced_win20', 'orange_balanced_win20.csv'), header=None),
    #         'blueberry': pd.read_csv(join('data_1_balanced_win20', 'blueberry_balanced_win20.csv'), header=None),
    #         'peach': pd.read_csv(join('data_1_balanced_win20', 'peach_balanced_win20.csv'), header=None),
    #         'apricot': pd.read_csv(join('data_1_balanced_win20', 'apricot_balanced_win20.csv'), header=None),
    #         'pear': pd.read_csv(join('data_1_balanced_win20', 'pear_balanced_win20.csv'), header=None)}
    # mt.visualizeUMAP(data)
