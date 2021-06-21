# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.optimize._minimize
from src import Filter_2D, addPipeLinestoDict

# Functions

def makeModels(df= None, filter_2D = False, set_normy = False, set_scaler = True, set_pca = True,
                algs = ['ridge', 'knn', 'svr', 'dt', 'rf', 'gp'], save_fig = True, show_fig = False,
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

    Returns:

    return_pipe_dict (dictionary): A dictionary with the best parameters found during GridSearchCV or RandomizedSearchCV.
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
    pipe_dict = addPipeLinestoDict(set_normy= set_normy,
                                   set_pca = set_pca,
                                   set_kernel = set_kernel,
                                   svr_opt_kern= svr_opt_kern,
                                   set_scaler = set_scaler,
                                   algs = algs,
                                   set_varthresh = set_varthresh,
                                   gp_alpha= gp_alpha,
                                   gp_opt = gp_opt)

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
        print('{}'.format(alg))
        print("Best training score is: {}".format(search.best_score_))
        print("Best params are: {}".format(search.best_params_))

        # Make and print predictions and scores
        if alg == 'gp':
            y_pred, sigma = pipe_dict[alg]['best_estimator'].predict(X_test, return_std = True)

        else:
            y_pred = pipe_dict[alg]['best_estimator'].predict(X_test)

        r2 = r2_score(y_pred=y_pred, y_true=y_test)
        rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
        print("{}'s Test R2 is: {}".format(alg, r2))
        print("{}'s Test RMSE is: {}".format(alg, rmse))
        print('\n')

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

        pipe_return_dict = {'pipe_dict': pipe_dict,
                            'X_test': X_test,
                            'y_test': y_test,
                            'y_pred': y_pred,
                            'df': df,
                            'titlestr': titlestr}

    return pipe_return_dict

def sampleNoise(df = None, base_sigma = None, num_levels = 10):
    """
    Generates 'num_levels' levels of gaussian distributed noise calculated from the integer range of y values in the
    input dataframe.

    Parameters:
    df (dataframe): Input dataframe to generate noise values from, and to which columns will be added. (Default = None)
    base_sigma (float): Base value to generate noise levels from. If set to None, will be the range of endpoint values.
                        (Default = None)
    num_levels (int): Integer value that gives the number of noise levels to generate. (Default = 10)

    Returns:
    end_dict (dict of series): Dictionary of endpoint columns with added gaussian distributed noise.

    """

    # Define variable for endpoint column name
    y = df.iloc[:,-1].name

    # Get range of y_int values
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
    file (string) : Absolute path to CSV file to transform to a pandas dataframe. The CSV should have Padel descriptors
                    and endpoints. (Default = None)
    out (string): Absolute path for PKL file which will be written. The PKL file will contain all the endpoints with noise
                  added, the estimator trained on each noise level, the predicted endpoints for each noise level, and the
                  RMSEs and R2s for each noise level. (Default = None)
    n_samples (int): The number of samples that will be randomly selected from the dataframe for processing. If None is
                     input, then the whole dataframe is used. (Default = None)
    multiplier (float): Base multiplier for the addition of noise to the dataframe. The multiplier is multiplied times
                        the range of the endpoint to give the base sigma for noise generation.
                        I.e. sig = y_range*multiplier*n. (Default = 0.01)
    reps (int): Number of repetitions to generate at each noise level. (Default = 5)
    n_levels (int): Number of noise levels to generate. (Default = 15)
    alg (string): Name of algorithm to use for model generation. Can be 'ridge', 'knn', 'svr', 'dt', 'rf', or 'gp'.
                (Default = None).
    set_search (string): Name of cross validation method to use. Can be 'RSV' for RandomSearchCV or 'GSV' for
                         GridSearchCV. (Default = 'GSV')
    set_kernel (string): Name of kernel to use if using GaussianProcess algorithm. Can be 'RBF', 'ConstantKernel',
                         'Matern', 'RationalQuadratic', 'ExpSineSquared', 'WhiteKernel', or 'DotProduct'.
                         (Default = 'RBF')
    set_normy (bool): Will normalize the endpoint data if set to 'True', used with 'gp' algorithm. (Default = False)
    gp_opt (str): Optimizer to be used in GP, if None is passed, then 'L-BGFS-B' will be used. The current
                options are optimCOBYLA and optimTNC. (Default = None)
    use_alpha (bool): If set to True, the uncertainty present in the dataset will be used in the 'gp' algorithm.
                    (Default = False)

    Returns:
    None.

    """
    # Read in dataframe from dataset
    df_1 = pd.read_csv(filepath_or_buffer= file, index_col = 0, header = 0)

    # Filter dataframe so that the final column is the experimental endpoint
    df_1 = df_1.iloc[:, :-2]

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
            else:
                gp_alpha = None

            # Make model and extract values from dictionary
            est = makeModels(df = df,
                             algs = alg,
                             set_search= set_search,
                             set_kernel= set_kernel,
                             set_normy= set_normy,
                             svr_opt_kern= False,
                             save_fig = False,
                             gp_alpha= gp_alpha,
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

            # Print statement
            print('Algorithm: {}'.format(alg))
            print('{}, {}:'.format(coldict, col))
            print('RMSE(true)' + ': {}'.format(rmse_nonoise))
            print('RMSE: {}'.format(rmse))
            print('R^2(true)' + ': {}'.format(r2_nonoise))
            print('R^2' + ': {}'.format(r2))
            print('\n')

    # Save dictionary to pkl
    with open(out, 'wb') as fp:
        pickle.dump(noise_dict, fp)

    return

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
            modelOnNoise(file=file,
                         out=out,
                         alg=alg,
                         set_search='RSV',
                         n_samples=n_samples,
                         multiplier=multiplier,
                         reps=reps,
                         n_levels=n_levels)
        elif alg == 'gp':
            modelOnNoise(file=file,
                         out=out,
                         alg=alg,
                         set_normy=True,
                         n_samples=n_samples,
                         multiplier=multiplier,
                         reps=reps,
                         n_levels=n_levels,
                         set_kernel=set_kernel)
        else:
            modelOnNoise(file=file,
                         out=out,
                         alg=alg,
                         n_samples=n_samples,
                         multiplier=multiplier,
                         reps=reps,
                         n_levels=n_levels,
                         set_search=set_search)

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



