#!/usr/bin/env python
# coding: utf-8


import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from padelpy import from_smiles
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, DotProduct, ConstantKernel, \
    RationalQuadratic, ExpSineSquared
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.utils import resample
import os
import pickle
from yellowbrick.regressor import residuals_plot, ResidualsPlot
from matplotlib.ticker import MultipleLocator
import pprint
import scipy.optimize._minimize

def Filter_2D(df = None):
    """Filters dataframe to include only 2D Padel descriptors.

        Parameters:
        df (dataframe): A dataframe containing 2D + 3D Padel descriptors as columns.

        Returns:
        df (dataframe): A dataframe filtered by 2D Padel descriptors.

        """
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError('Input must be dataframe.')
    df_2D = pd.read_csv('C:/Users/skolmar/PycharmProjects/NewErrorModel/src/Features/2DDescriptors.csv', index_col='Name')
    output = df.iloc[:, -1]
    df = df.filter(df_2D.columns, axis=1)
    df[output.name] = output
    return df

def AddPipeLinestoDict(algs = ['ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', 'gp', 'nn'],
                       set_kernel = 'RBF', svr_opt_kern = False, set_normy = False, set_scaler = True, set_pca = True,
                       set_varthresh = None, gp_alpha = None, gp_opt = None):
    """

    Parameters:
    algs (list): List of desired regressors. (Default = ['ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', 'gp'])
    set_kernel (str): Sets kernel for Gaussian Processor. Can be any 'RBF', 'ConstantKernel', 'Matern',
                    'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel', or 'DotProduct'. (Default = 'RBF')
    svr_opt_kern (bool): If set to True, includes kernel in grid or random search CV for SVR. (Default = False)
    set_normy (bool): If set to True, normalizes y values for Gaussian Processor. (Default = False)
    set_scaler (bool): If set to True, uses scaler in pipelines. (Default = True)
    set_pca (bool): If set to True, uses PCA in pipelines. (Default = True)
    set_varthresh (float): If a float is provided, will set a variance threshold for features. (Default = None)
    gp_alpha (float): Value to add to the diagonal of the kernel matrix of GP. (Default = None)
    gp_opt (str): Optimizer to be used in GP, if None is passed then 'L-BGFS-B' will be used. (Default = None).

    Returns:

    pipe_dict (dict): Dictionary containing pipelines and parameter grids.
    """
    # Set Random seed
    seed = 42

    # Define Kernels
    kerns = {'RBF': RBF(), 'WhiteKernel': WhiteKernel(), 'Matern': Matern(), 'DotProduct': DotProduct(),
             'ExpSineSquared': ExpSineSquared(), 'ConstantKernel': ConstantKernel(),
             'RationalQuadratic': RationalQuadratic()
             }

    if set_kernel in kerns.keys():
        user_kern = kerns[set_kernel]

    #Define Regressors and PreProcessors
    scaler = StandardScaler()
    vt = VarianceThreshold(threshold= set_varthresh)
    pca = PCA()
    ridge = Ridge(random_state = seed)
    lasso = Lasso(random_state = seed)
    knn = KNeighborsRegressor()
    svr = SVR()
    dt = DecisionTreeRegressor(random_state = seed)
    rf = RandomForestRegressor(random_state= seed)

    # Set if clause to avoid GP alpha being set to 0.0
    if gp_alpha == 0.0:
        gp_alpha = 1.10E-10
    else:
        pass

    gp = GaussianProcessRegressor(normalize_y= set_normy, kernel= user_kern, n_restarts_optimizer=5, random_state= seed,
                                  alpha = gp_alpha, optimizer= gp_opt)
    xgb = XGBRegressor(random_state= seed)
    net = MLPRegressor(activation= 'relu', random_state= seed, max_iter= 500, solver= 'adam',
                      learning_rate_init= 0.0001)



    #Make list of regressors and zip into tuple with user input
    regtups = [('ridge', ridge), ('lasso', lasso), ('knn', knn),
               ('svr', svr), ('dt', dt), ('rf', rf), ('gp', gp), ('xgb', xgb), ('net', net)]

    # Add each pipeline and param grid to a pipe_dict
    pipe_dict = {}
    for reg in regtups:
        if reg[0] in algs:
            step1 = ('scaler_' + reg[0], scaler)
            step2 = ('vt_' + reg[0], vt)
            step3 = ('pca_' + reg[0], pca)
            step4 = (reg[0] + '_reg', reg[1])
            steps = []
            pipe_dict[reg[0]] = {'params': {}}
            if set_scaler == True:
                steps.append(step1)
            if set_varthresh is not None:
                steps.append(step2)
            if set_pca == True:
                steps.append(step3)
                pipe_dict[reg[0]]['params']['pca_{}__n_components'.format(reg[0])] = np.arange(1, 60, 2)
            steps.append(step4)
            if reg[0] == 'ridge':
                pipe_dict[reg[0]]['params']['ridge_reg__alpha'] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
            elif reg[0] == 'lasso':
                pipe_dict[reg[0]]['params']['lasso_reg__alpha'] = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
            elif reg[0] == 'knn':
                pipe_dict[reg[0]]['params']['knn_reg__n_neighbors'] = np.arange(1, 20, 1)
            elif reg[0] == 'svr':
                if svr_opt_kern is True:
                    pipe_dict[reg[0]]['params']['svr_reg__kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
                pipe_dict[reg[0]]['params']['svr_reg__C'] = [0.01, 0.1, 1.0, 10.0]
            elif reg[0] == 'dt':
                pipe_dict[reg[0]]['params']['dt_reg__max_depth'] = np.arange(1, 100, 2)
                pipe_dict[reg[0]]['params']['dt_reg__max_leaf_nodes'] = np.arange(2, 100, 10)
            elif reg[0] == 'rf':
                pipe_dict[reg[0]]['params']['rf_reg__n_estimators'] = np.arange(1, 200, 10)
                pipe_dict[reg[0]]['params']['rf_reg__max_depth'] = np.arange(1, 100, 2)
                pipe_dict[reg[0]]['params']['rf_reg__max_leaf_nodes'] = np.arange(2, 100, 10)
            elif reg[0] == 'xgb':
                pipe_dict[reg[0]]['params']['xgb_reg__eta'] = [0.1, 0.3, 0.5, 0.7, 0.9]
                pipe_dict[reg[0]]['params']['xgb_reg__gamma'] = [1, 2, 5, 10, 20, 50]
                pipe_dict[reg[0]]['params']['xgb_reg__max_depth'] = [3, 5, 10, 15, 20]
                pipe_dict[reg[0]]['params']['xgb_reg__subsample'] = [0.5, 0.6, 0.7, 0.8]
            elif reg[0] == 'net':
                pipe_dict[reg[0]]['params']['net_reg__hidden_layer_sizes'] = [(50,50,50), (100,100,100), (200,200,200),
                                                                             (500,500,500), (1000,1000,1000)]
                pipe_dict[reg[0]]['params']['net_reg__learning_rate'] = ['constant', 'invscaling', 'adaptive']

            pipe_dict[reg[0]]['pipe'] = Pipeline(steps = steps)

    return pipe_dict

def Descriptor_DataFrame(df=None, smiles_col=None, time=12, output=None, csv=None):
    """ Generates PaDel descriptors and outputs in dataframe format. Removes duplicate entries
        from dataframe. Accounts for RuntimeErrors.
    
        Parameters:
        
        df (dataframe): A pandas dataframe containing a SMILES column. (Default = None)
        smiles_col (str or int): Name or index of column which contains SMILES strings. (Default = None)
        time (int): Quantity of time in seconds before a single PaDel call will timeout. (Default = 12)
        output (str or int): Column in df which is the desired output parameter. (Default = None)
        csv (str): If a file name is included then will output CSV file. (Default = None)
        
        Returns:
        
        df_desc (dataframe): A pandas dataframe with index set as index of df and columns as descriptors,
                            with final column as desired output value.
                            
        """
    # Raise error if input is not a dataframe
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError('Input df must be a Pandas DataFrame.')

    # Raise error if smiles_col is empty
    if smiles_col is None:
        raise ValueError('Must designate columns containing SMILES strings.')

    #Raise error if smiles_col as str is not in df
    if isinstance(smiles_col, str) == True:
        if smiles_col not in df.columns:
            raise ValueError('Smiles_col must be a column in df.')

    #Raise error if smiles_col as int is not in df
    if isinstance(smiles_col, int) == True:
        if smiles_col >= len(df.columns):
            raise ValueError('Smiles_col index must be a column in df.')

    # Remove repeat indices from df
    df = df[~df.index.duplicated()]

    # Create empty df_desc with same index as that of df
    df_desc = pd.DataFrame(index=df.index)

    # Iterate over SMILES column in df, create descriptor dict,
    # convert descriptor dict to pandas series, and append series to df_desc
    #If smiles_col was loaded as string
    if isinstance(smiles_col, str):
        for item in df[smiles_col].iteritems():
            try:
                desc_dict = from_smiles(item[1], timeout=time)
                desc_series = pd.Series(desc_dict, name=item[0])
                df_desc = df_desc.append(desc_series)
            # The following exception allows from_smiles to continue if a single calculation times out
            except RuntimeError:
                pass
    #If smiles_col was loaded as int
    elif isinstance(smiles_col, int):
        for item in df.iloc[:, smiles_col].iteritems():
            try:
                desc_dict = from_smiles(item[1], timeout=time)
                desc_series = pd.Series(desc_dict, name=item[0])
                df_desc = df_desc.append(desc_series)
            # The following exception allows from_smiles to continue if a single calculation times out
            except RuntimeError:
                pass

    # Set df_desc output column as desired output column from df
    if output is None:
        pass
    elif isinstance(output, str) == True:
        df_desc[output] = df.loc[:, output]
    elif isinstance(output, int) == True:
        column = df.iloc[:, output]
        df_desc[column.name] = column

    # Drop NaN's
    df_desc.dropna(axis=0, how='any', inplace=True)

    # If statement to convert to CSV
    if csv != None:
        file = df_desc.to_csv(csv + '.csv')
        return file

    return df_desc

def MakeModels(df= None, filter_2D = False, set_normy = False, set_scaler = True, set_pca = True,
                algs = ['ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', 'gp', 'xgb', 'net'], save_fig = True, show_fig = False,
                set_kernel = 'RBF', svr_opt_kern = False, set_search = 'GSV', set_varthresh = None, gp_alpha = None,
               gp_opt = None):
    """
    Performs TrainTestSplit on DataFrame, generates models by applying pipelines,
    performs GridSearchCV or RandomSearchCV to tune hyperparameters, generates plots, and saves them as PNG's.

    Parameters:

    df (dataframe): Dataframe to generate training and test sets from. (Default = None)
    filter_2D (bool): If 'True' filters dataframe to include only 2D descriptors. (Default = False)
    set_kernel (str): Sets kernel for Gaussian Processor. Can be any 'RBF', 'ConstantKernel', 'Matern',
               'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel', or 'DotProduct'. (Default = 'RBF')
    svr_opt_kern (bool): If set to True, includes kernel in grid or random search CV for SVR. (Default = False)
    set_scaler (bool): If 'True' includes StandardScaler function in pipeline. (Default = True)
    set_pca (bool): If 'True' includes PCA function in pipeline. (Default = True)
    algs (list): List of algorithms to include. (Default = ['ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', 'xgb'])
    save_fig (bool): If 'True' saves the plots. (Default = True)
    show_fig (bool): If 'True' shows plots. (Default = False)
    set_normy (bool): If 'True' normalizes y values for Gaussian Processor.
    set_search (str): If set to 'GSV' will perform GridSearchCV,
                 if set to 'RSV' will perform RandomizedSearchCV. (Default = 'GSV')
    set_varthres (float): If a float is provided, will set a variance threshold for features. (Default = None)
    gp_alpha (float): Value for the uncertainty in each endpoint for Gaussian Process. If a value is given, will add
                     that value to the diagonal of the kernel matrix, adding uncertainty to the predictions. (Default = None)
    gp_opt (str): Optimizer to be used in GP, if None is passed, then 'L-BGFS-B' will be used. (Default = None)

    Returns (dictionary):

    pipe_dict (dictionary): A dictionary with the best parameters found during GridSearchCV or RandomizedSearchCV.
                           A best estimator object is also stored in this dictionary for further use of optimized
                           models.
    X_test (dataframe): A dataframe containing features for the test set which was formed during TrainTestSplit.
    y_test (series): A series containing output variable for the test set which was formed during TrainTestSplit.
    y_pred (series): A series containing values predicted from a model.
    df (dataframe): Dataframe used for training and testing.
    titlestr (str): Title string for file.

    """
    # Raise error if df not specified
    if df is None:
        raise ValueError('Must specify a dataframe.')
    if isinstance(df, pd.DataFrame) == False:
        raise ValueError('Input df must be a dataframe.')

    # If filter_2D is True, will only use 2D descriptors
    if filter_2D == True:
        df = Filter_2D(df)

    # Remove infinite values
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')

    # Set X and y
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    #AddPipelinestoDict
    pipe_dict = AddPipeLinestoDict(set_normy= set_normy, set_pca = set_pca, set_kernel = set_kernel,
                                   svr_opt_kern= svr_opt_kern, set_scaler = set_scaler, algs = algs,
                                   set_varthresh = set_varthresh, gp_alpha= gp_alpha, gp_opt = gp_opt)

    for alg in pipe_dict.keys():
        if set_search == 'GSV':
            search = GridSearchCV(pipe_dict[alg]['pipe'], pipe_dict[alg]['params'], cv=5, n_jobs=-1)
            search.fit(X_train, y_train)
            pipe_dict[alg]['best_estimator'] = search.best_estimator_
        if set_search == 'RSV':
            search = RandomizedSearchCV(pipe_dict[alg]['pipe'], pipe_dict[alg]['params'], cv=5,
                                        n_jobs=-1, n_iter=500)
            search.fit(X_train, y_train)
            pipe_dict[alg]['best_estimator'] = search.best_estimator_

        # Print training scores and params
        print("{}'s Best score is: {}".format(alg, search.best_score_))
        print("{}'s Best params are: {}".format(alg, search.best_params_))

        # Make and print predictions and scores
        if alg == 'gp':
            y_pred, sigma = pipe_dict[alg]['best_estimator'].predict(X_test, return_std = True)

        else:
            y_pred = pipe_dict[alg]['best_estimator'].predict(X_test)

        r2 = r2_score(y_pred=y_pred, y_true=y_test)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        print("{}'s R2 is: {}".format(alg, r2))
        print("{}'s RMSE is: {}".format(alg, rmse))

        # Plot prediction with error bars if GP
        if alg == 'gp':
            f1, ax = plt.subplots(figsize=(12, 12))
            plt.plot(y_test, y_pred, 'r.')
            plt.errorbar(y_test, y_pred, yerr=sigma, fmt='none', ecolor='blue', alpha=0.8)

        else:
            f1, ax = plt.subplots(figsize=(12, 12))
            plt.plot(y_test, y_pred, 'r.')

        # Plot line with ideal slope of 1
        x_vals = np.array(ax.get_xlim())
        y_vals = 0 + 1*x_vals
        plt.plot(x_vals, y_vals, "r--")

        # Add title string
        titlestr = []
        if filter_2D is True:
            title2D = '2D'
            titlestr.append(title2D)
        if set_pca == True:
            titlepca = 'PCA'
            titlestr.append(titlepca)
        if set_scaler == True:
            titlescaler = 'Scaled'
            titlestr.append(titlescaler)
        titlename = str(alg)
        titlestr.append(titlename)
        titley = y.name
        titlestr.append(titley)
        if 'gp' in algs:
            titlekern = set_kernel
            titlestr.append(titlekern)
            if set_normy is True:
                titlenormy = 'NormY'
                titlestr.append(titlenormy)
        if set_varthresh is not None:
            titlevt = 'VT_{}'.format(set_varthresh)
            titlestr.append(titlevt)

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = '\n'.join(f'{k}: {v}' for k, v in search.best_params_.items())
        textstr2 = 'R2 is: %.3f' % r2 + '\n' + 'RMSE is %.3f' % rmse + '\n' + '\n'.join(titlestr)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.text(0.05, 0.80, textstr2, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        # Plot Figure with relative path to folder
        filedir = os.path.dirname(os.path.abspath(__file__))
        parentdir = os.path.dirname(filedir)
        plotdir = os.path.join(parentdir, 'Plots')
        titledir = os.path.join(plotdir, titley)
        if save_fig == True:
            if not os.path.exists(titledir):
                os.makedirs(titledir)
        plt.title(titlename)
        plt.xlabel('{} Experimental'.format(titley))
        plt.ylabel('{} Predicted'.format(titley))
        if save_fig == True:
            figfile = os.path.join(titledir, '_'.join(titlestr))
            plt.savefig('{}.png'.format(figfile))
        if show_fig == True:
            plt.show()
        plt.close()

    return {'pipe_dict': pipe_dict, 'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred, 'df': df, 'titlestr': titlestr}

def RunBootStrap(X_test = None, y_test = None, model=None, n_iterations=500, titlestr = None):
    """ Performs bootstrap calculation on a machine learning model with dataframe of choice.
        Generates distributions of test statistics and provides plots.

    Parameters:

    X_test (dataframe): Features of the test set on which to compute test statistics. (Default = None)
    Y_test (series): Output variable of the test set on which to compute test statistics. (Default = None)
    model (Any sklearn objects with fit and predict methods): Model to generate test statistics for. (Default = None)
    n_iterations (int): Number of bootstrap iterations to generate. (Default = 500)
    titlestr (str): Title string containing file name for a model. (Default = None)

    Returns:

    List of test statistics.

    """
    # Initialize scores lists
    r2scores = []
    rmsescores = []
    for i in range(n_iterations):
        # Resample Lip and y from test dataframe
        sample_x, sample_y = resample(X_test, y_test)

        # Predict with model
        y_pred = model.predict(sample_x)

        # Calculate scores and add to lists
        r2 = r2_score(y_pred=y_pred, y_true=sample_y)
        r2scores.append(r2)
        rmse = mean_squared_error(y_pred= y_pred, y_true=sample_y, squared=False)
        rmsescores.append(rmse)

    # Calculate statistics
    r2stats = {}
    r2stats['mean'] = np.mean(r2scores)
    r2stats['sd'] = np.std(r2scores)
    r2stats['ci'] = np.percentile(r2scores, [2.5, 97.5])

    rmsestats = {}
    rmsestats['mean'] = np.mean(rmsescores)
    rmsestats['sd'] = np.std(rmsescores)
    rmsestats['ci'] = np.percentile(rmsescores, [2.5, 97.5])

    # Plot R2 Scores
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.hist(r2scores, bins='auto')
    ax1.autoscale()
    ax1.set_title('R2 Scores')

    # R2 Score text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    r2textstr = "Mean : %.2f" % r2stats['mean'] + '\n' + 'SD : %.2f' % r2stats['sd'] +\
                '\n' + 'CI : {0:.2f}, {1:.2f}'.format(r2stats['ci'][0], r2stats['ci'][1])
    ax1.text(0.05, 0.95, r2textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    # Plot RMSE Scores
    ax2.hist(rmsescores, bins='auto')
    ax2.autoscale()
    ax2.set_title('RMSE Scores')

    # RMSE Score text box
    rmsetextstr = "Mean : %.2f" % rmsestats['mean'] + '\n' + 'SD : %.2f' % rmsestats['sd'] +\
                  '\n' + 'CI : {0:.2f}, {1:.2f}'.format(rmsestats['ci'][0], rmsestats['ci'][1])
    ax2.text(0.05, 0.95, rmsetextstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    plt.show()

    if titlestr is not None:
        plt.savefig(r'C:\Users\skolmar\PycharmProjects\NewErrorModel\Plots\{}_RunBootStrap'.format('_'.join(titlestr)))

    return r2stats, rmsestats

def RunBootStrapAllDf(df = None, model=None, n_iterations=500, titlestr = None):
    """ Performs bootstrap calculation on a machine learning model with dataframe of choice.
        Generates distributions of test statistics and provides plots.

    Parameters:

    df (dataframe): Dataframe used for training and predicting with the model. (Default = None)
    model (Any sklearn objects with fit and predict methods): Model to generate test statistics for. (Default = None)
    n_iterations (int): Number of bootstrap iterations to generate. (Default = 500)
    titlestr (str): Title string for model input. (Default = None)

    Returns:

    List of test statistics.

    """

    # Initialize scores lists
    r2scores = []
    rmsescores = []
    for i in range(n_iterations):
        # Resample dataframe
        df_new = resample(df)
        X = df_new.iloc[:, :-1]
        y = df_new.iloc[:, 1]

        #Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2)

        # Fit and Predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate scores and add to lists
        r2 = r2_score(y_pred=y_pred, y_true=y_test)
        r2scores.append(r2)
        rmse = mean_squared_error(y_pred=y_pred, y_true=y_test, squared=False)
        rmsescores.append(rmse)

    # Calculate statistics
    r2stats = {}
    r2stats['mean'] = np.mean(r2scores)
    r2stats['sd'] = np.std(r2scores)
    r2stats['ci'] = np.percentile(r2scores, [2.5, 97.5])

    rmsestats = {}
    rmsestats['mean'] = np.mean(rmsescores)
    rmsestats['sd'] = np.std(rmsescores)
    rmsestats['ci'] = np.percentile(rmsescores, [2.5, 97.5])

    # Plot R2 Scores
    f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.hist(r2scores, bins='auto')
    ax1.autoscale()
    ax1.set_title('R2 Scores')

    # R2 Score text box
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    r2textstr = "Mean : %.2f" % r2stats['mean'] + '\n' + 'SD : %.2f' % r2stats['sd'] +\
                '\n' + 'CI : {0:.2f}, {1:.2f}'.format(r2stats['ci'][0], r2stats['ci'][1])
    ax1.text(0.05, 0.95, r2textstr, transform=ax1.transAxes, fontsize=12,
                verticalalignment='top', bbox=props)

    # Plot RMSE Scores
    ax2.hist(rmsescores, bins='auto')
    ax2.autoscale()
    ax2.set_title('RMSE Scores')

    # RMSE Score text box
    rmsetextstr = "Mean : %.2f" % rmsestats['mean'] + '\n' + 'SD : %.2f' % rmsestats['sd'] +\
                  '\n' + 'CI : {0:.2f}, {1:.2f}'.format(rmsestats['ci'][0], rmsestats['ci'][1])
    ax2.text(0.05, 0.95, rmsetextstr, transform=ax2.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    plt.show()

    if titlestr is not None:
        plt.savefig(r'C:\Users\skolmar\PycharmProjects\NewErrorModel\Plots\{}_RunBootStrapAllDF'.format('_'.join(titlestr)))

    return r2stats, rmsestats

def sampleNoise(df = None, logvalues = False, base_sigma = None, num_levels = 10):
    """
    Generates 'num_levels' levels of gaussian distributed noise calculated from the integer range of y values in the
    input dataframe.

    Parameters:
    df (dataframe): Input dataframe to generate noise values from, and to which columns will be added. (Default = None)
    logvaues (bool): Set to True if endpoint values are log10. This will convert to integer values before adding noise,
                    then convert the noisy output back to log10. (Default = False)
    base_sigma (float): Base value to generate noise levels from. If set to None, will be the range of endpoint values.
                        (Default = None)
    num_levels (int): Integer value that gives the number of noise levels to generate. (Default = 10)

    Returns:
    end_dict (dict of series): Dictionary of endpoint columns with added gaussian distributed noise.

    """

    # Define variable for endpoint column name
    y = df.iloc[:,-1].name

    # Define log10 to integer converter
    def convLog10toint(x):
        return 10**x
    # Get range of y_int values
    if logvalues is True:
        df['y_int'] = df[y].apply(convLog10toint)
        y_max = df['y_int'].max()
        y_min = df['y_int'].min()
    else:
        y_max = df[y].max()
        y_min = df[y].min()

    y_range = y_max-y_min
    if base_sigma is None:
        multiplier = y_range
    else:
        multiplier = base_sigma

    # Create list of noise levels and choose multiplier
    lvls = [*range(0, num_levels, 1)]
    noiselvls = [multiplier*x for x in lvls]

    # Create a dict of endpoints, each with increasing noise
    end_dict = {}
    for i,lvl in enumerate(noiselvls):

        # Get size of endpoint columns for gaussian number generator
        col_size = df[y].shape[0]

        # Generate gaussian distributed noise which is the same shape as endpoint column
        err = np.random.normal(0, scale = lvl, size = [col_size])

        # Make unique names endpoint column by level of noise
        col_name = 'Noise_' + str(i)

        # Add new endpoint column to dictionary
        if logvalues is True:
            df_dict[df_name] = df.iloc[:, :-2].copy(deep=True)
            df_dict[df_name][col_name] = df['y_int'] + err
            df_dict[df_name][col_name].apply(np.log10)
        else:
            end_dict[col_name] = df[y].copy(deep = True) + err

    return end_dict

def modelOnNoise(file = None, out = None, n_samples = None, multiplier = 0.01, reps = 5, n_levels = 15,
                 alg = None, set_search = 'GSV', set_kernel = 'RBF', set_normy = False, gp_opt = None, use_alpha = False):
    """
    Takes an input CSV and generates 'n_levels' noise levels, with 'reps' number of repetitions per noise level by adding
    in gaussian distributed noise to the true data. Makes models with 'alg' algorithm for each noise level, saving the
    estimators used, the RMSEs and R2s, and the noise modified data as a dictionary into a .pkl file named as the
    parameter 'out'.

    Parameters:
    file (string) : CSV file to transform to a pandas dataframe. The CSV should have Padel descriptors and endpoints.
                  (Default = None)
    out (string): Filename for .pkl file which will be written. The .pkl file will contain all the endpoints with noise
                  added, the estimator trained on each noise level, the predicted endpoints for each noise level, and the
                  RMSEs and R2s for each noise level. (Default = None)
    n_samples (int): The number of samples that will be randomly selected from the dataframe for processing. If None is
                     input, then the whole dataframe is used. (Default = None)
    multiplier (float): Base multiplier for the addition of noise to the dataframe. The multiplier is multiplied times
                        the range of the endpoint to give the base sigma for noise generation.
                        I.e. sig = y_range*multiplier*n. (Default = 0.01)
    reps (int): Number of repetitions to generate at each noise level. (Default = 5)
    n_levels (int): Number of noise levels to generate. (Default = 15)
    alg (string): Name of algorithm to use for model generation. Can be 'ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', 'gp'
                  or 'xgb'. (Default = None).
    set_search (string): Name of cross validation method to use. Can be 'RSV' for RandomSearchCV or 'GSV' for
                         GridSearchCV. (Default = 'GSV')
    set_kernel (string): Name of kernel to use if using GaussianProcess algorithm. Can be 'RBF', 'ConstantKernel',
                         'Matern', 'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel', or 'DotProduct'.
                         (Default = 'RBF')
    set_normy (bool): Will normalize the endpoint data if set to 'True', used with 'gp' algorithm. (Default = False)
    gp_opt (str): Optimizer to be used in GP, if None is passed, then 'L-BGFS-B' will be used. (Default = None)
    use_alpha (bool): If set to True, the uncertainty present in the dataset will be used in the 'gp' algorithm.
                    (Default = False)

    Returns:
    None.

    """
    # Read in dataframe from dataset
    df_1 = pd.read_csv(file, index_col = 0)

    # Drop columns with errors in descriptor column values
    cols = [x for x in df_1.columns if df_1[x].dtype == 'object']
    df_2 = df_1.drop(columns = cols)


    # Randomly sample dataframe and assign sigma from y range and multiplier
    if isinstance(n_samples, int):
        df_2 = df_2.sample(n=n_samples).copy(deep=True)
    sig = (df_2.iloc[:,-1].max() - df_2.iloc[:,-1].min())*multiplier

    # Make dictionary of column dictionaries
    noise_dict = {}
    for i in range(reps):
        title = 'col_dict_{}'.format(i)
        noise_dict[title] = sampleNoise(df = df_2, base_sigma= sig, num_levels = n_levels)

    # Iterate over each col_dict in noise_dict
    for coldict in noise_dict.keys():

        # Iterate over each column in col_dict
        for i,col in enumerate(noise_dict[coldict].keys()):

            # Copy dataframe and add column to it
            df = df_2.iloc[:,:-1].copy(deep=True)
            df[col] = noise_dict[coldict][col]

            # Extract the sigma term from the added gaussian error for each noise level, to input into GP, if GP is used
            if alg == 'gp':
                y_max = df.iloc[:, -1].max()
                y_min = df.iloc[:, -1].min()
                y_range = y_max - y_min
                if use_alpha is True:
                    gp_alpha = y_range*multiplier*i
                else:
                    gp_alpha = 1E-10

            # Make model and extract values from dictionary
            est = MakeModels(df = df, algs = alg, set_search= set_search, set_kernel= set_kernel,
                             set_normy= set_normy, svr_opt_kern= False, save_fig = False, gp_alpha= gp_alpha,
                             gp_opt = gp_opt)

            # Extract X_test, y_pred, y_true, and y_noise
            xtest = est['X_test']
            ind = xtest.index
            y_pred = est['y_pred']
            y_true_df = df_2.loc[ind]
            y_true = y_true_df.iloc[:,-1]
            y_noise = est['y_test']

            # Calculate RMSEs with and without noise
            rmse_nonoise = mean_squared_error(y_true= y_true, y_pred= y_pred, squared= False)
            rmse = mean_squared_error(y_true = y_noise, y_pred = y_pred, squared = False)
            noise_dict[coldict][col]['rmsenonoise'] = rmse_nonoise
            noise_dict[coldict][col]['rmse'] = rmse

            # Calculate R2s with and without noise
            r2_nonoise = r2_score(y_true= y_true, y_pred = y_pred)
            r2 = r2_score(y_true = y_noise, y_pred = y_pred)
            noise_dict[coldict][col]['r2nonoise'] = r2_nonoise
            noise_dict[coldict][col]['r2'] = r2

            # Save estimator dict
            noise_dict[coldict][col]['est'] = est

    # Save dictionary to pkl
    with open(out, 'wb') as fp:
        pickle.dump(noise_dict, fp)

def getSigmaForXAxis(noise_dict = None, multiplier = None, inds = None):
    """
    Generates list of sigma values used in error generation using a noise dictionary. This function extracts the range
    of y-values from the 'Noise_0' column in 'col_dict_0' of the supplied noise_dict. Multiplier is the multiplier used
    in the noise_dict and must be supplied to the function. The inds are the levels of noise in noise_dict. Each value
    is the range*multiplier*ind.

    Parameters:
    noise_dict (dict): Noise dictionary stored in a PKL file. (Default = None)
    multiplier (float): Multiplier used for generating error. (Default = None)
    inds (list of int): List of integers starting at 0 used in noise_dict. (Default = None)

    Returns:
    siglist (list of float): List of float values which are y_range*multiplier*ind.

    """
    # Obtain true data range
    truedata = noise_dict['col_dict_0']['Noise_0'][:-5]
    ymax = truedata.max()
    ymin = truedata.min()
    truerange = ymax-ymin

    # Calculate sigmalist
    sigmalist = [truerange*multiplier*ind for ind in inds]

    return sigmalist

def plotErrors(pkl=None, n_levels=15, multiplier = 0.01, out_path = None, x_tick_at = None, x_normalized = False,
               y_normalized = False):
    """
    Builds plots of RMSE versus noise level and R2 versus noise level. Takes a .pkl file as an input, extracts RMSEs and
    R2s for each noise level and repetition, and shows the plots.

    Parameters:
    pkl (string): PKL file which contains RMSEs and R2s organized by noise level and repetition. (Default = None)
    n_levels (int): Number of noise levels in the .pkl file. Should match 'n_levels' parameter of 'modelOnNoise'.
                    (Default = 15).
    multiplier (float): Multiplier used to generate noise levels. (Default = 0.01)
    out_path (str): Absolute path to save the figure to. If None is provided then figure won't be saved. (Default = None)
    x_tick_at (int or float): Interval for which x-ticks appear on the figure. (Default = None)
    x_normalized (bool): If True, the x-axes (sigma) of the plots will be normalized to
                      the RMSE of the true data (no noise). (Default = False)
    y_normalized (bool): If True, the y-axis (RMSE) will be normalized to the RMSE of the true data (no noise).
                        (Default = False)

    Returns:
    fig (plt figure): A figure with two subplots. The top subplot is RMSE vs. noise level, the bottom subplot is R2
                      versus noise level.

    """
    # Load pickle file
    noise_dict = pickle.load(open(pkl, 'rb'))

    # Make empty dictionaries for RMSE, RMSE_nonoise, r2, and r2_nonoise with keys as levels
    # Each level is a list
    RMSE_dict = {}
    RMSE_nonoise_dict = {}
    r2_dict = {}
    r2_nonoise_dict = {}
    for i in range(n_levels):
        RMSE_dict[i] = []
        RMSE_nonoise_dict[i] = []
        r2_dict[i] = []
        r2_nonoise_dict[i] = []

    # Iterate over noise_dict to put RMSEs and RMSENoNoises into dictionaries by level
    for coldict in noise_dict.keys():
        for i, col in enumerate(noise_dict[coldict].keys()):
            rmse = noise_dict[coldict][col]['rmse']
            rmsenonoise = noise_dict[coldict][col]['rmsenonoise']
            r2 = noise_dict[coldict][col]['r2']
            r2nonoise = noise_dict[coldict][col]['r2nonoise']
            RMSE_dict[i].append(rmse)
            RMSE_nonoise_dict[i].append(rmsenonoise)
            r2_dict[i].append(r2)
            r2_nonoise_dict[i].append(r2nonoise)

    # Make dataframes from RMSE and r2 dictionaries
    RMSE_df = pd.DataFrame.from_dict(data=RMSE_dict, orient='index')
    RMSE_nonoise_df = pd.DataFrame.from_dict(data=RMSE_nonoise_dict, orient='index')
    r2_df = pd.DataFrame.from_dict(data=r2_dict, orient='index')
    r2_nonoise_df = pd.DataFrame.from_dict(data=r2_nonoise_dict, orient='index')

    # Calculate RMSE averages, standard deviations, and the difference
    RMSE_df['ave'] = RMSE_df.mean(axis=1)
    RMSE_df['sd'] = RMSE_df.std(axis=1)
    RMSE_nonoise_df['ave'] = RMSE_nonoise_df.mean(axis=1)
    RMSE_nonoise_df['sd'] = RMSE_nonoise_df.std(axis=1)
    RMSE_df['diffs'] = RMSE_df['ave'] - RMSE_nonoise_df['ave']
    diffs = RMSE_df['diffs']

    # Calculate R2 averages, standard deviations, and the difference
    r2_df['ave'] = r2_df.mean(axis=1)
    r2_df['sd'] = r2_df.std(axis=1)
    r2_nonoise_df['ave'] = r2_nonoise_df.mean(axis=1)
    r2_nonoise_df['sd'] = r2_nonoise_df.std(axis=1)
    r2_df['diffs'] = r2_df['ave'] - r2_nonoise_df['ave']
    r2_diffs = r2_df['diffs']

    # Set indices
    inds = [x for x in range(n_levels)]

    # Obtain sigmalist for alternate x-axis of sigma values used in noise generation
    sigmalist = getSigmaForXAxis(noise_dict = noise_dict, inds = inds, multiplier = multiplier)

    # Extract title
    dataset = pkl.split('\\')
    datasettitle = dataset[7]
    if datasettitle != 'BACE':
        datasettitle = datasettitle.capitalize()
    title = noise_dict['col_dict_0']['Noise_0']['est']['titlestr'][2]
    if title == 'ridge':
        title = title.capitalize()
    else:
        title = title.upper()

    # Instantiate figure, add subplot 1 and title
    fig = plt.figure()
    fig.suptitle('{}, {}'.format(datasettitle, title))
    ax = fig.add_subplot(2, 1, 1)

    # Set base_rmse as the RMSE of the noiseless data
    base_rmse = RMSE_df.loc[0, 'ave']

    # Normalization clause for sigma
    if x_normalized is True:
        xaxis = [x / base_rmse for x in sigmalist]
        xaxislab = "$\u03C3$" + ' / ' + '$RMSE_{0}$'
    else:
        xaxis = sigmalist
        xaxislab = "$\u03C3$"

    # Normalization clause for RMSE
    if y_normalized is True:
        yaxis = RMSE_df['ave'].div(base_rmse)
        yaxiserr = RMSE_df['sd'].div(base_rmse)
        yaxislab = "RMSE" + ' /' + '$RMSE_{0}$'

        yaxistrue = RMSE_nonoise_df['ave'].div(base_rmse)
        yaxiserrtrue = RMSE_nonoise_df['sd'].div(base_rmse)
    else:
        yaxis = RMSE_df['ave']
        yaxiserr = RMSE_df['sd']
        yaxislab = 'RMSE'

        yaxistrue = RMSE_nonoise_df['ave']
        yaxiserrtrue = RMSE_nonoise_df['sd']


    # Set if clause for significant digits in the slopes
    if datasettitle in ['Alpha', 'Lip', 'Solv', 'BACE', 'Tox']:
        slopefigs = '\nSlope: %.3f'

    else:
        slopefigs = '\nSlope: %.2f'

    # Determine slope of RMSE vs sigma
    rmseslope, rmseint = np.polyfit(y=yaxis, x=xaxis, deg=1)

    # Plot RMSE vs sigma
    ax.errorbar(xaxis, yaxis, yerr=yaxiserr, label='RMSE' + slopefigs %rmseslope)

    # Determine slope of RMSEtrue vs sigma
    rmsetrueslope, rmsetrueint = np.polyfit(y=yaxistrue, x=xaxis, deg=1)

    # Plot RMSEtrue vs sigma
    ax.errorbar(xaxis, yaxistrue, yerr=yaxiserrtrue, label='$RMSE_{true}$' + slopefigs %rmsetrueslope, color = 'orange')

    # Set axis labels
    ax.set_xlabel(xaxislab)
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_at))
    ax.set_ylabel(yaxislab)

    # Set legend
    ax.legend()
    plt.grid(b = None)
    ax.tick_params(axis='both', which='major', length=5)

    # Add subplot 2
    ax2 = fig.add_subplot(2,1,2)

    # Determine slope of R2 vs sigma
    r2slope, r2int = np.polyfit(y=r2_df['ave'], x=xaxis, deg=1)

    # Plot R2 vs sigma
    ax2.errorbar(xaxis, r2_df['ave'], yerr=r2_df['sd'], label='$R^2$' + '\nSlope: %.4f' %r2slope)

    # Determine slope of R2true vs sigma
    r2trueslope, r2trueint = np.polyfit(y = r2_nonoise_df['ave'], x = xaxis, deg = 1)

    # Plot R2true vs sigma
    ax2.errorbar(xaxis, r2_nonoise_df['ave'], yerr=r2_nonoise_df['sd'],
                 label='$R^2_{true}$' '\nSlope: %.4f' % r2trueslope, color = 'orange')

    # Set axis labels
    ax2.set_xlabel(xaxislab)
    ax2.xaxis.set_major_locator(MultipleLocator(x_tick_at))
    ax2.set_ylabel('$R^2$')

    # Set Legend
    ax2.legend()
    ax2.tick_params(axis='both', which='major', length=5)
    plt.grid(b=None)

    # Save slopes for output
    slopes = {}
    slopes['rmseslope'] = rmseslope
    slopes['rmsetrueslope'] = rmsetrueslope
    slopes['r2slope'] = r2slope
    slopes['r2trueslope'] = r2trueslope
    slopes['base_rmse'] = base_rmse

    # Save figure
    if out_path is None:
        plt.tight_layout()
        plt.show()
        return fig, slopes
    else:
        plt.tight_layout()
        plt.savefig(out_path)
        plt.show()
        return fig, slopes

def getSampleInds(df_file=None, pklfile=None):
    """
    Extracts sample indices for the samples which were randomly selected from the original dataframe. These are needed
    to retrieve the training set for each estimator. The yellowbrick module requires training sets to be input even
    if the estimator it uses is already fit. This function retrieves the sample indices by matching against the values
    in the .pkl file. I know. It's a hassle.

    Parameters:
    df_file (string): Original CSV file which was used to generate the noisy data. (Default = None)
    pklfile (string): PKL file which contains all the noisy data. (Default = None)

    Returns:
    sample_inds (index): Pandas index object (iterable) which contains the indices of the randomly selected samples.

    """
    # Read in original df
    df = pd.read_csv(df_file, index_col=0)

    # Drop columns with errors in descriptor column values
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns = cols)

    # Remove infinite values
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')

    # Load noise_dict
    noise_dict = pickle.load(open(pklfile, 'rb'))

    # Read in fitted est, X_test, y_noise, and retrieve y_true from original df
    subdict = 'Noise_0'
    X_test = noise_dict['col_dict_0'][subdict]['est']['X_test']
    ind = X_test.index
    y_noise = noise_dict['col_dict_0'][subdict]['est']['y_test']
    y_true_df = df.loc[ind]
    y_true = y_true_df.iloc[:, -1]

    # Get sampled df values
    values = noise_dict['col_dict_0'][subdict]

    # Filter original df to get sampled df, extract indices
    df_sample = df[df.iloc[:, -1].isin(values[:-5])]
    sample_inds = df_sample.index

    return sample_inds

def buildResiduals(df_file = None, pklfile = None, coldict = 'col_dict_0', subdict = None, alg = 'ridge',
                   fig_path= None):
    """
    Uses the yellowbrick module to build distribution of residuals plots. The plots show distributions of residuals for
    both the training and test sets. Two plots are shown for each noise level, for both the noisy and noiseless test sets.
    The choice of 'coldict' determines which of the 'reps' are chosen to plot the data. The number of 'reps' is simply
    the number of repetitions made at each noise level; therefore, 'col_dict_0' arbitrarily chooses the first rep of
    each noise level. The parameter 'subdict' determines which noise level to generate a plot from. It is convenient to
    iterate through each noise level and call this function on each level, with 'subdict' as the noise level dictionary
    key.

    Parameters:
    df_file (string): CSV file which contains the original dataset. This is read in and transformed into a pandas
                      dataframe. (Default = None)
    pklfile (string): PKL file which contains the noise dictionary. This is used to retrieve sample indices.
                      (Default = None)
    coldict (string): String which is the key for the repetition number which is desired. (Default = 'col_dict_0')
    subdict (string): String of the format 'Noise_{}' where {} is an integer. For example, in a dictionary which
                      has 15 noise levels, one would have 'Noise_0' through 'Noise_14' as choices for subdict. Each
                      subdict is a noise level. It is convenient to iterate through each noise level and call this
                      function on each level. (Default = None).
    alg (string): Name of the algorithm used in the modeling. This is necessary to retrieve the estimator from the PKL
                  file. Can be 'ridge', 'lasso', 'knn', 'svr', 'dt', 'rf', or 'xgb'. (Default = 'ridge').
    fig_path (string): Path for the desired destination directory of the figures. (Default = None)

    Returns:
    viz, viz2 (figs): Yellowbrick residuals plots, one using noisy data as the test set, and one using noiseless data
                      as the test set.

    """

    # Load noise_dict
    noise_dict = pickle.load(open(pklfile, 'rb'))

    # Load original df
    df = pd.read_csv(df_file, index_col= 0)

    # Drop columns with errors in descriptor column values
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns = cols)

    # Remove infinite values
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')

    # Extract sample indices from original df
    sample_inds = getSampleInds(df_file = df_file, pklfile = pklfile)

    # Extract est, X_test, y_noise, and y_true
    est = noise_dict[coldict][subdict]['est']['pipe_dict'][alg]['best_estimator']
    X_test = noise_dict[coldict][subdict]['est']['X_test']
    X_inds = X_test.index
    X_vals = X_test.values
    y_noise = noise_dict[coldict][subdict]['est']['y_test']
    y_true_df = df.loc[X_inds]
    y_true = y_true_df.iloc[:, -1]

    # Extract df_sampled
    df_sampled = df.loc[sample_inds]

    # Get X_train and y_train
    train_df = df_sampled[~df_sampled.iloc[:, -1].isin(y_noise.values)]
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:,-1]

    # Set titles
    alg_title = noise_dict['col_dict_0']['Noise_0']['est']['titlestr'][-2].capitalize()
    noise_title = 'Residuals for {0} Model, Trained on {1}, {1}'.format(alg_title, subdict)
    true_title = 'Residuals for {0} Model, Trained on {1}, No Noise'.format(alg_title, subdict)

    # Set save filepaths
    noise_path = r'{}\{}\{}_{}.png'.format(fig_path,alg_title,subdict,subdict)
    nonoise_path = r'{}\{}\{}_NoNoise.png'.format(fig_path, alg_title, subdict)

    # Build residual plot on noisy data
    viz = ResidualsPlot(est, title = noise_title)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_noise)
    viz.show(outpath= noise_path)
    #viz.show()
    #viz = residuals_plot(model = est, X_train = X_train, y_train = y_train, X_test = X_test, y_test= y_noise,
                         #title = noise_title)

    # Build residual plot on true data
    #viz2 = residuals_plot(model = est, X_train = X_train, y_train = y_train, X_test = X_test, y_test= y_true,
                          #title = true_title)
    viz2 = ResidualsPlot(est, title = true_title)
    viz2.fit(X_train, y_train)
    viz2.score(X_test, y_true)
    viz2.show(outpath = nonoise_path)
    #viz2.show()

    return viz, viz2

def combineMetrics(file = None, n_levels = 15):
    r"""
    Takes a file name that represents the .pkl files for a specific dataset, and combines all the RMSE and R2 metrics
    from each algorithm into a single dictionary so that they can all be plotted together.

    Parameters:
    file (string): File name with curly brackets where .pkl files are stored for a particular dataset. I.e.
                   'C:\Users\skolmar\PycharmProjects\NewErrorModel\src\Plots\BACE\WithR2s\BACE_15_5_0p1_{}.pkl'
                   (Default = None)
    n_levels (int): Number of levels used in the generation of the data. (Default = 15)

    Returns:
    metrics (dict): Dictionary of dataframes, which contain RMSE and R2 statistics for each algorithm for a specific dataset.

    """

    # Extract dataset title
    dataset = file.split('\\')
    datasettitle = dataset[7]

    # Initialize empty dictionary
    metrics = {}
    metrics[datasettitle] = {}

    algs = ['ridge', 'knn', 'rf', 'svr']
    filelist = [file.format(x) for x in algs]
    tups = list(zip(algs, filelist))
    for alg,file in tups:

        # Load pickle file
        noise_dict = pickle.load(open(file, 'rb'))

        # Make empty dictionaries for RMSE, RMSE_nonoise, r2, and r2_nonoise with keys as levels
        # Each level is a list
        RMSE_dict = {}
        RMSE_nonoise_dict = {}
        r2_dict = {}
        r2_nonoise_dict = {}
        for i in range(n_levels):
            RMSE_dict[i] = []
            RMSE_nonoise_dict[i] = []
            r2_dict[i] = []
            r2_nonoise_dict[i] = []

        # Iterate over noise_dict to put RMSEs and RMSENoNoises into dictionaries by level
        for coldict in noise_dict.keys():
            for i, col in enumerate(noise_dict[coldict].keys()):
                rmse = noise_dict[coldict][col]['rmse']
                rmsenonoise = noise_dict[coldict][col]['rmsenonoise']
                r2 = noise_dict[coldict][col]['r2']
                r2nonoise = noise_dict[coldict][col]['r2nonoise']
                RMSE_dict[i].append(rmse)
                RMSE_nonoise_dict[i].append(rmsenonoise)
                r2_dict[i].append(r2)
                r2_nonoise_dict[i].append(r2nonoise)

        # Make dataframes from RMSE and r2 dictionaries
        RMSE_df = pd.DataFrame.from_dict(data=RMSE_dict, orient='index')
        RMSE_nonoise_df = pd.DataFrame.from_dict(data=RMSE_nonoise_dict, orient='index')
        r2_df = pd.DataFrame.from_dict(data=r2_dict, orient='index')
        r2_nonoise_df = pd.DataFrame.from_dict(data=r2_nonoise_dict, orient='index')

        # Calculate RMSE averages, standard deviations
        RMSE_df['ave'] = RMSE_df.mean(axis=1)
        RMSE_df['sd'] = RMSE_df.std(axis=1)
        RMSE_nonoise_df['ave'] = RMSE_nonoise_df.mean(axis=1)
        RMSE_nonoise_df['sd'] = RMSE_nonoise_df.std(axis=1)

        # Calculate R2 averages, standard deviations
        r2_df['ave'] = r2_df.mean(axis=1)
        r2_df['sd'] = r2_df.std(axis=1)
        r2_nonoise_df['ave'] = r2_nonoise_df.mean(axis=1)
        r2_nonoise_df['sd'] = r2_nonoise_df.std(axis=1)

        # Update dict with statistics for alg
        metrics[datasettitle][alg] = {'rmse_aves': RMSE_df['ave'],
                             'rmse_sd': RMSE_df['sd'],
                             'rmse_true_aves': RMSE_nonoise_df['ave'],
                             'rmse_true_sd': RMSE_nonoise_df['sd'],
                             'r2_aves': r2_df['ave'],
                             'r2_sd': r2_df['sd'],
                             'r2_true_aves': r2_nonoise_df['ave'],
                             'r2_true_sd': r2_nonoise_df['sd']
                            }

    return metrics

def plotMetrics(n_levels = 15, metrics_dict = None, pkl_file = None, x_ticks_at = None, multiplier = 0.01):
    """
    Takes a metrics dictionary from combineMetrics and plots RMSE/RMSEtrue versus noise, and R2/R2true versus noise, for each algorithm,
    for a single dataset. Shows plots.

    Parameters:
    metrics_dict (dict): Dictionary of RMSEs and R2s returned by function combineMetrics. (Default = None)
    n_levels (int): Number of noise levels in the metric dictionary. (Default = 15)
    pkl_file (str): Absolute path to PKL file to extract sigmalist for x-axis. (Default = None)
    x_ticks_At (int or float): Interval at which to put x-ticks on the plots. (Default = None)
    multiplier (float): Multiplier used to generate noise levels. (Default = 0.01)

    Returns:
    None

    """
    inds = [x for x in range(n_levels)]

    # Retrieve dataset key from dictionary
    dataset = list(metrics_dict.keys())[0]

    # Set title
    datasettitle = dataset
    if datasettitle != 'BACE':
        datasettitle = datasettitle.capitalize()

    # Initialize RMSE figure, fig
    fig = plt.figure()
    fig.suptitle('{}'.format(datasettitle))
    ax = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    # Load PKL file for sigmalist
    noise_dict = pickle.load(open(pkl_file, 'rb'))

    # Obtain sigmalist for x-axis
    sigmalist = getSigmaForXAxis(noise_dict= noise_dict, inds = inds, multiplier = multiplier)

    # Iterate through algorithms
    for alg in metrics_dict[dataset].keys():

        # Set RMSE variables and plot
        rmse = metrics_dict[dataset][alg]['rmse_aves']
        rmse_sd = metrics_dict[dataset][alg]['rmse_sd']
        rmse_true = metrics_dict[dataset][alg]['rmse_true_aves']
        rmse_true_sd = metrics_dict[dataset][alg]['rmse_true_sd']
        ax.errorbar(sigmalist, rmse, yerr = rmse_sd, label = alg)
        ax2.errorbar(sigmalist, rmse_true, yerr = rmse_true_sd, label = alg)

    ax.set_xlabel("$\u03C3$")
    ax2.set_xlabel("$\u03C3$")
    ax.set_ylabel("RMSE")
    ax2.set_ylabel("$RMSE_{true}$")
    ax.xaxis.set_major_locator(MultipleLocator(x_ticks_at))
    ax2.xaxis.set_major_locator(MultipleLocator(x_ticks_at))
    ax.tick_params(axis='both', which='major', length=5)
    ax2.tick_params(axis='both', which='major', length=5)

    # Find ax maximum to set ax ylim
    axy = []
    for line in ax.lines:
        ydata = line.get_ydata()
        ymax = np.amax(ydata)
        axy.append(ymax)
    axmax = max(axy)
    ax.set_ylim(0, axmax + axmax*0.1)

    # Find ax2 maximum to set ax2 ylim
    ax2y = []
    for line in ax2.lines:
        ydata = line.get_ydata()
        ymax = np.amax(ydata)
        ax2y.append(ymax)
    ax2max = max(ax2y)
    ax2.set_ylim(0, ax2max + ax2max*0.1)

    ax.grid(b=None)
    ax2.grid(b=None)
    ax.legend()
    plt.show()

    # Initialize R2 figure, fig2
    fig2 = plt.figure()
    fig2.suptitle('{}'.format(datasettitle))
    ax3 = fig2.add_subplot(2, 1, 1)
    ax4 = fig2.add_subplot(2, 1, 2)

    # Iterate through algorithms
    for alg in metrics_dict[dataset].keys():
        # Set RMSE variables and plot
        r2 = metrics_dict[dataset][alg]['r2_aves']
        r2_sd = metrics_dict[dataset][alg]['r2_sd']
        r2_true = metrics_dict[dataset][alg]['r2_true_aves']
        r2_true_sd = metrics_dict[dataset][alg]['r2_true_sd']
        ax3.errorbar(sigmalist, r2, yerr=r2_sd, label=alg)
        ax4.errorbar(sigmalist, r2_true, yerr=r2_true_sd, label=alg)

    ax3.set_xlabel("$\u03C3$")
    ax4.set_xlabel("$\u03C3$")
    ax3.set_ylabel("$R^2$")
    ax4.set_ylabel("$R^2_{true}$")
    ax3.xaxis.set_major_locator(MultipleLocator(x_ticks_at))
    ax4.xaxis.set_major_locator(MultipleLocator(x_ticks_at))
    ax3.tick_params(axis='both', which='major', length=5)
    ax4.tick_params(axis='both', which='major', length=5)

    # Find ax3 maximum to set ax3 ylim
    ax3y = []
    ax3ymins = []
    for line in ax3.lines:
        ydata = line.get_ydata()
        ymax = np.amax(ydata)
        ax3y.append(ymax)
        ymin = np.amin(ydata)
        ax3ymins.append(ymin)
    ax3max = max(ax3y)
    if min(ax3ymins) < 0:
        ax3min = min(ax3ymins) + min(ax3ymins) * 0.1
    else:
        ax3min = 0
    ax3.set_ylim(ax3min, ax3max + ax3max * 0.1)

    # Find ax4 maximum to set ax4 ylim
    ax4y = []
    ax4ymins = []
    for line in ax4.lines:
        ydata = line.get_ydata()
        ymax = np.amax(ydata)
        ax4y.append(ymax)
        ymin = np.amin(ydata)
        ax4ymins.append(ymin)
    ax4max = max(ax4y)
    if min(ax4ymins) < 0:
        ax4min = min(ax4ymins) + min(ax4ymins) * 0.1
    else:
        ax4min = 0
    ax4.set_ylim(ax4min, ax4max + ax4max * 0.1)

    ax3.grid(b=None)
    ax4.grid(b=None)
    ax3.legend()
    plt.show()

    return None


def fullModelOnNoise(file=None, outfile=None, algs=['ridge', 'knn', 'svr', 'rf', 'gp'],
                     n_samples=None, multiplier=0.01, reps=5, n_levels=15, set_search='GSV',
                     set_kernel='RBF', set_normy=False):
    """
    Takes a CSV file with descriptors and single endpoint and runs modelOnNoise function for each
    algorithm on this file.

    Parameters:
    file (str): Absolute path for CSV file containing descriptors and endpoint. (Default = None)
    outfile (str): Absolute path for output PKL file. Should be of the form
                  '...\{dataset_name}_{n_noise_levels}_{n_reps}_{multiplier}_{}.pkl'
                  This function will add algorithm name to the last bracketed term. Creates a PKL
                  file for each algorithm. (Default = None)
    algs (list of str): Algorithms to run for the provided dataset.
                       (Default = ['ridge', 'knn', 'svr', 'rf', 'gp'])
    n_samples (int): The number of samples that will be randomly selected from the dataframe for processing. If None is
                     input, then the whole dataframe is used. (Default = None)
    multiplier (float): Base multiplier for the addition of noise to the dataframe. The multiplier is multiplied times
                        the range of the endpoint to give the base sigma for noise generation.
                        I.e. sig = y_range*multiplier*n. (Default = 0.01)
    reps (int): Number of repetitions to generate at each noise level. (Default = 5)
    n_levels (int): Number of noise levels to generate. (Default = 15)
    set_search (string): Name of cross validation method to use. Can be 'RSV' for RandomSearchCV or 'GSV' for
                         GridSearchCV. (Default = 'GSV')
    set_kernel (string): Name of kernel to use if using GaussianProcess algorithm. Can be 'RBF', 'ConstantKernel',
                         'Matern', 'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel', or 'DotProduct'.
                         (Default = 'RBF')
    set_normy (bool): Will normalize the endpoint data if set to 'True', used with 'gp' algorithm. (Default = False)


    Returns:
    None.

    """

    # Define outlist, tups
    outlist = [outfile.format(x) for x in algs]
    tups = list(zip(algs, outlist))

    # Define model loop
    for alg, out in tups:
        if alg == 'rf':
            modelOnNoise(file=file, out=out, alg=alg, set_search='RSV', n_samples=n_samples,
                         multiplier=multiplier, reps=reps, n_levels=n_levels)
        elif alg == 'gp':
            modelOnNoise(file=file, out=out, alg=alg, set_normy=True, n_samples=n_samples,
                         multiplier=multiplier, reps=reps, n_levels=n_levels, set_kernel=set_kernel)
        else:
            modelOnNoise(file=file, out=out, alg=alg, n_samples=n_samples,
                         multiplier=multiplier, reps=reps, n_levels=n_levels, set_search=set_search)

    return None

def extractStdFromGP(pkl = None, csv = None):
    """
    Takes a gaussian process PKL file as argument, extracts the array of prediction errors for each repetition
    and noise level, stores statistics about the prediction error in a dictionary, and returns the dictionary.

    Parameters:

    pkl (str): Absolute path to a PKL file for Gaussian Process. (Default = None)
    csv (str): Absolute path to CSV file which contains dataset descriptors and endpoints. (Default = None)
    """
    # Read PKL file into a dictionary
    noise_dict = pickle.load(open(pkl, 'rb'))

    # Read in dataframe from original dataset
    df = pd.read_csv(csv, index_col = 0)

    # Drop columns with errors in descriptor column values
    cols = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(columns = cols)

    # Remove infinite values
    df[df.columns] = df[df.columns].astype(float)
    df = df.fillna(0.0).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0, how='any')

    # Instantiate empty dictionary
    std_dict = {}
    true_std_dict = {}

    # Iterate through each column dictionary and each noise level
    for col in noise_dict.keys():

        # Instantiate empty column dictionary
        std_dict[col] = {}
        true_std_dict[col] = {}

        for noise in noise_dict[col].keys():

            # Instantiate empty noise level dictionary
            std_dict[col][noise] = {}
            true_std_dict[col][noise] = {}

            # Extract estimator, X_test, y_test, X_true, and y_true
            est = noise_dict[col][noise]['est']['pipe_dict']['gp']['best_estimator']
            X_test = noise_dict[col][noise]['est']['X_test']
            y_test = noise_dict[col][noise]['est']['y_test']
            indices = X_test.index
            df_true =  df.loc[indices]
            X_true = df_true.iloc[:,:-1]
            y_true = df_true.iloc[:,-1]

            # Predict on X_test to get std array
            std = est.predict(X_test, return_std = True)
            stdarray = std[1]
            ave = np.average(stdarray)
            perc1 = np.percentile(stdarray, 2.5)
            perc2 = np.percentile(stdarray, 97.5)
            dev = np.std(stdarray)

            # Predict on X_true to get true_std array
            true_std = est.predict(X_true, return_std = True)
            true_stdarray = true_std[1]
            true_ave = np.average(true_stdarray)
            true_perc1 = np.percentile(true_stdarray, 2.5)
            true_perc2 = np.percentile(true_stdarray, 97.5)
            true_dev = np.std(true_stdarray)

            # Store std values in dictionary
            std_dict[col][noise]['ave'] = ave
            std_dict[col][noise]['2.5th p'] = perc1
            std_dict[col][noise]['97.5th p'] = perc2
            std_dict[col][noise]['CI'] = perc2 - perc1
            std_dict[col][noise]['std'] = dev

            # Store true_std values in dictionary
            true_std_dict[col][noise]['ave'] = true_ave
            true_std_dict[col][noise]['2.5th p'] = true_perc1
            true_std_dict[col][noise]['97.5th p'] = true_perc2
            true_std_dict[col][noise]['CI'] = true_perc2 - true_perc1
            true_std_dict[col][noise]['std'] = true_dev

    return std_dict, true_std_dict


def plotStdArraysGP(csv=None, pkl=None, out = None, normalized = False):
    """
    Takes a PKL of a Gaussian Process model and a CSV of the original dataset and plots the prediction error of the GP.
    The data is the array of prediction errors from the GP.

    Parameters:
    CSV (str): Absolute path for the CSV file which contains descriptors and endpoints. (Default = None)
    PKL (str) Absolute path for the PKL file which contains GP model information. (Default = None)
    out (str): Absolute path for the folder for the output figure. (Default = None)
    normalized (bool): If True will normalize the x-axis by the basermse. (Default = False)

    Returns:
    fig (plt): Matplotlib figure.

    """

    # Load noise dictionary from PKL
    noise_dict = pickle.load(open(pkl, 'rb'))

    # Extract prediction error arrays from each rep and noise level
    std_dict, true_std_dict = extractStdFromGP(pkl=pkl, csv=csv)

    # Define noise level and empty dictionaries
    n_levels = 15
    CI_dict = {}
    true_CI_dict = {}
    ave_dict = {}
    true_ave_dict = {}

    # Make a dictionary key for each noise level for CI and ave
    for i in range(n_levels):
        CI_dict[i] = []
        true_CI_dict[i] = []
        ave_dict[i] = []
        true_ave_dict[i] = []

    # Add the five reps of CI or ave to each noise level
    for col in std_dict.keys():
        for i, noise in enumerate(std_dict[col].keys()):
            CI = std_dict[col][noise]['CI']
            true_CI = true_std_dict[col][noise]['CI']
            ave = std_dict[col][noise]['ave']
            true_ave = true_std_dict[col][noise]['ave']
            CI_dict[i].append(CI)
            true_CI_dict[i].append(true_CI)
            ave_dict[i].append(ave)
            true_ave_dict[i].append(true_ave)

    # Convert CI and ave dictionaries into dataframes
    CI_df = pd.DataFrame.from_dict(data=CI_dict, orient='index')
    true_CI_df = pd.DataFrame.from_dict(data=true_CI_dict, orient='index')
    ave_df = pd.DataFrame.from_dict(data=ave_dict, orient='index')
    true_ave_df = pd.DataFrame.from_dict(data=true_ave_dict, orient='index')

    # Calculate mean and standard deviation for each noise level
    CI_df['ave'] = CI_df.mean(axis=1)
    CI_df['sd'] = CI_df.std(axis=1)
    true_CI_df['ave'] = true_CI_df.mean(axis=1)
    true_CI_df['sd'] = true_CI_df.std(axis=1)
    ave_df['ave_of_ave'] = ave_df.mean(axis=1)
    ave_df['sd'] = ave_df.std(axis=1)
    true_ave_df['ave_of_ave'] = true_ave_df.mean(axis=1)
    true_ave_df['sd'] = true_ave_df.std(axis=1)

    # Calculate base RMSE for sigmalistbase
    # Make empty dictionaries for RMSE with keys as levels
    # Each level is a list
    RMSE_dict = {}
    for i in range(n_levels):
        RMSE_dict[i] = []

    # Iterate over noise_dict to put RMSEs into dictionaries by level
    for coldict in noise_dict.keys():
        for i, col in enumerate(noise_dict[coldict].keys()):
            rmse = noise_dict[coldict][col]['rmse']
            RMSE_dict[i].append(rmse)

    # Make dataframes from RMSE dictionaries
    RMSE_df = pd.DataFrame.from_dict(data=RMSE_dict, orient='index')

    # Calculate RMSE average
    RMSE_df['ave'] = RMSE_df.mean(axis=1)
    basermse = RMSE_df.loc[0, 'ave']

    # Define indices for sigma axis and obtain sigma axis
    inds = [x for x in range(n_levels)]

    # Normalize sigma if normalized is true
    sigmalist = getSigmaForXAxis(noise_dict=noise_dict, inds=inds, multiplier=0.01)
    if normalized is True:
        xaxis = sigmalist/basermse
        xaxlabel = '$\u03C3$' + ' /' + '$RMSE_{0}$'
        yCI = CI_df['ave'].div(basermse)
        yCIerr = CI_df['sd'].div(basermse)
        yCIlabel = '$\u03C3_{\u0177}$' + ' 95% CI /' + '$RMSE_{0}$'
        yave = ave_df['ave_of_ave'].div(basermse)
        yaveerr = ave_df['sd'].div(basermse)
        yavelabel = 'Mean ' + '$\u03C3_{\u0177}$' + '/' + '$RMSE_{0}$'
        aveslope, aveint = np.polyfit(y=yave, x=xaxis, deg=1)
        CIslope, CIint = np.polyfit(y=yCI, x=xaxis, deg=1)
    else:
        xaxis = sigmalist
        xaxlabel = '$\u03C3$'
        yCI = CI_df['ave']
        yCIerr = CI_df['sd']
        yCIlabel = '$\u03C3_{\u0177}$' + ' 95% CI'
        yave = ave_df['ave_of_ave']
        yaveerr = ave_df['sd']
        yavelabel = 'Mean ' + '$\u03C3_{\u0177}$'
        aveslope, aveint = np.polyfit(y=yave, x=xaxis, deg=1)
        CIslope, CIint = np.polyfit(y=yCI, x=xaxis, deg=1)

    # Set x_tick
    x_tick_at = round(max(xaxis)/6)
    if x_tick_at < 1:
        x_tick_at = round(max(xaxis)/6) + 0.2

    # Instantiate figure, CI subplot
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)
    ax.errorbar(xaxis, yCI, yerr=yCIerr, label='CI' + '\nSlope: %.4f' %CIslope)
    ax.set_xlabel(xaxlabel)
    ax.set_ylabel(yCIlabel)
    ax.legend()
    plt.grid(b=None)
    ax.tick_params(axis='both', which='major', length=5)
    ax.xaxis.set_major_locator(MultipleLocator(x_tick_at))

    # Generate plot title
    terms = pkl.split('\\')[10]
    terms2 = terms.split('_')
    title = terms2[0]
    fig.suptitle(title + ' Gaussian Process Prediction Error')

    # Add ave subplot
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.errorbar(xaxis, yave, yerr=yaveerr, label='ave'+ '\nSlope: %.4f' %aveslope)
    ax2.set_xlabel(xaxlabel)
    ax2.set_ylabel(yavelabel)
    ax2.legend()
    plt.grid(b=None)
    ax2.tick_params(axis='both', which='major', length=5)
    ax2.xaxis.set_major_locator(MultipleLocator(x_tick_at))
    plt.tight_layout()
    if normalized is True:
        outfile = out + '\\' + terms.split('.')[0] + '_norm' + '.png'
    elif normalized is False:
        outfile = out + '\\' + terms.split('.')[0] + '.png'
    plt.savefig(outfile)
    plt.show()

    return fig


def createAlphaDict(pkl=None):
    """
    Creates a dictionary of alpha values from a Gaussian Process PKL file.

    Parameters:
    pkl (str): Absolute path to PKL file for Gaussian Process data. (Default = None)

    Returns:
    alpha_dict (dict): Dictionary of alpha values sorted by noise level.

    """

    # Load PKL file into noise_dict
    noise_dict = pickle.load(open(pkl, 'rb'))

    # Extract number of noise lvls
    lvls = len(noise_dict['col_dict_0'].keys())

    # Create alpha dictionary with noise levels
    alpha_dict = {}
    noiselvls = ['Noise_{}'.format(x) for x in range(lvls)]
    for noiselvl in noiselvls:
        alpha_dict[noiselvl] = []

    # Iterate through noise dictionary to add alpha values by level
    for coldict in noise_dict.keys():
        for noiselvl in noise_dict[coldict].keys():
            alpha = noise_dict[coldict][noiselvl]['est']['pipe_dict']['gp']['pipe']['gp_reg'].alpha
            alpha_dict[noiselvl].append(alpha)

    # Pretty print dictionary for inspection
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(alpha_dict)

    return alpha_dict

def plotExplainedVariance(pkl = None, out = None):
    """
    Takes a PKL file and plots the explained variance ratio of the PCA versus number of components in the PCA, for each
    noise level and repetition. Saves the plots

    Parameters:
    pkl (str): Absolute path to PKL file. (Default = None)
    out (str): Absolute path for directory of plot output. (Default = None)

    Returns:


    """
    pkl_dict = pickle.load(open(pkl, 'rb'))

    for coldict in pkl_dict.keys():
        for noisedict in pkl_dict[coldict].keys():
            best_est = list(pkl_dict[coldict][noisedict]['est']['pipe_dict'].values())[0]['best_estimator']
            pca = list(best_est.named_steps.values())[1]
            plt.plot(np.cumsum(pca.explained_variance_ratio_))
            plt.xlabel('Number of components')
            plt.ylabel('Cumulative explained variance')
            title = coldict + ' ' + noisedict + 'PCA'
            plt.title(title)
            outfile = out + '\\{}.png'.format(title)
            plt.savefig(outfile)
            plt.close()

    return

def optimCOBYLA(obj_func, initial_theta, bounds):
    optresult = scipy.optimize.minimize(obj_func, initial_theta, method = 'COBYLA', jac = False)
    theta_opt = optresult.x
    func_min = optresult.fun
    return theta_opt, func_min

def optimTNC(obj_func, initial_theta, bounds):
    optresult = scipy.optimize.minimize(obj_func, initial_theta, method = 'TNC', jac = True)
    theta_opt = optresult.x
    func_min = optresult.fun
    return theta_opt, func_min

def optim2(obj_func, initial_theta, bounds):
    optresult = scipy.optimize.minimize(obj_func, initial_theta, method = 'COBYLA', jac = False)
    theta_opt = optresult.x
    func_min = optresult.fun
    return theta_opt, func_min

def plotComponents(pkl = None, out = None, n_levels = 15):
    """
    Takes a PKL file and plots number of PCA components versus sigma of the added error. The
    x-axis is the sigma for a given noise level, with the average as the y-axis value and error bars
    from the standard deviation of the repetitions.

    Parameters:
    pkl (str): Absolute path to the PKL file. (Default = None)
    out (str): Absolute path to the folder for the plot. (Default = None).
    n_levels (int): Number of noise levels in the dictionary. (Default = 15)

    Returns:


    """
    # Load PKL file into dictionary
    pkldict = pickle.load(open(pkl, 'rb'))

    # Instantiate dictionary with noise levels as keys
    compdict = {}
    noises = ['Noise_{}'.format(x) for x in range(n_levels)]
    for noise in noises:
        compdict[noise] = []

    # Iterate through pkldict and add a list of PCA n_components for each noise level as each dict value
    for coldict in pkldict.keys():
        for noisedict in pkldict[coldict].keys():
            best_est = list(pkldict[coldict][noisedict]['est']['pipe_dict'].values())[0]['best_estimator']
            pca = list(best_est.named_steps.values())[1]
            comp = pca.n_components_
            compdict[noisedict].append(comp)

    # Transform compdict into a dataframe
    comp_df = pd.DataFrame.from_dict(data = compdict, orient = 'index')
    comp_df['ave'] = comp_df.mean(axis = 1)
    comp_df['sd'] = comp_df.std(axis = 1)

    # Generate sigmalist
    inds = [x for x in range(n_levels)]
    sigmalist = getSigmaForXAxis(noise_dict=pkldict, inds=inds, multiplier=0.01)

    # Generate plot title
    terms = pkl.split('\\')[10]
    terms2 = terms.split('_')
    title = terms2[0] + ' ' + terms2[4].split('.')[0]

    # Plot values
    plt.errorbar(sigmalist, comp_df['ave'], yerr = comp_df['sd'])
    plt.xlabel('$\u03C3$')
    plt.ylabel('Number of PCA components')
    plt.title(title)
    outfile = out + '\\' + terms.split('.')[0] + '.png'
    plt.savefig(outfile)
    plt.show()

    return

