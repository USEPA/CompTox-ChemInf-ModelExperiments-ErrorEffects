# Libary imports
import numpy as np
import pandas as pd
from padelpy import from_smiles
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, DotProduct, ConstantKernel, \
    RationalQuadratic, ExpSineSquared

# Functions

def filter_2D(df = None):
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

def addPipeLinestoDict(algs = ['ridge', 'knn', 'svr', 'dt', 'rf', 'gp'],
                       set_kernel = 'RBF', svr_opt_kern = False, set_normy = False, set_scaler = True, set_pca = True,
                       set_varthresh = None, gp_alpha = None, gp_opt = None):
    """

    Parameters:
    algs (list): List of desired regressors. (Default = ['ridge', 'knn', 'svr', 'dt', 'rf', 'gp'])
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
    kerns = {'RBF': RBF(),
             'WhiteKernel': WhiteKernel(),
             'Matern': Matern(),
             'DotProduct': DotProduct(),
             'ExpSineSquared': ExpSineSquared(),
             'ConstantKernel': ConstantKernel(),
             'RationalQuadratic': RationalQuadratic()
             }

    if set_kernel in kerns.keys():
        user_kern = kerns[set_kernel]

    #Define Regressors and PreProcessors
    scaler = StandardScaler()
    vt = VarianceThreshold(threshold= set_varthresh)
    pca = PCA()
    ridge = Ridge(random_state = seed)
    knn = KNeighborsRegressor()
    svr = SVR()
    dt = DecisionTreeRegressor(random_state = seed)
    rf = RandomForestRegressor(random_state= seed)

    # Set if clause to avoid GP alpha being set to 0.0
    if gp_alpha == 0.0:
        gp_alpha = 1.10E-10
    else:
        pass

    gp = GaussianProcessRegressor(normalize_y= set_normy,
                                  kernel= user_kern,
                                  n_restarts_optimizer=5,
                                  random_state= seed,
                                  alpha = gp_alpha,
                                  optimizer= gp_opt)

    #Make list of regressors and zip into tuple with user input
    regtups = [('ridge', ridge),
               ('knn', knn),
               ('svr', svr),
               ('dt', dt),
               ('rf', rf),
               ('gp', gp)]

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

            pipe_dict[reg[0]]['pipe'] = Pipeline(steps = steps)

    return pipe_dict


def descriptor_DataFrame(df=None, smiles_col=None, time=12, output=None, csv=None):

    """

    Generates PaDel descriptors and outputs in dataframe format. Removes duplicate entries
    from dataframe. Accounts for RuntimeErrors.

    Parameters:

    df (dataframe): A pandas dataframe containing a SMILES column. (Default = None)
    smiles_col (str or int): Name or index of column which contains SMILES strings. (Default = None)
    time (int): Quantity of time in seconds before a single PaDel call will timeout. (Default = 12)
    output (str or int): Column in df which is the desired output parameter. (Default = None)
    csv (str): Absolute file path to output CSV file. (Default = None)

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

    # Raise error if smiles_col as str is not in df
    if isinstance(smiles_col, str) == True:
        if smiles_col not in df.columns:
            raise ValueError('Smiles_col must be a column in df.')

    # Raise error if smiles_col as int is not in df
    if isinstance(smiles_col, int) == True:
        if smiles_col >= len(df.columns):
            raise ValueError('Smiles_col index must be a column in df.')

    # Remove repeat indices from df
    df = df[~df.index.duplicated()]

    # Create empty df_desc with same index as that of df
    df_desc = pd.DataFrame(index=df.index)

    # Iterate over SMILES column in df, create descriptor dict,
    # convert descriptor dict to pandas series, and append series to df_desc
    # If smiles_col was loaded as string
    if isinstance(smiles_col, str):
        for item in df[smiles_col].iteritems():
            try:
                desc_dict = from_smiles(item[1], timeout=time)
                desc_series = pd.Series(desc_dict, name=item[0])
                df_desc = df_desc.append(desc_series)
            # The following exception allows from_smiles to continue if a single calculation times out
            except RuntimeError:
                pass
    # If smiles_col was loaded as int
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

    # Filter to 2D
    df_desc = filter_2D(df_desc)

    # If statement to convert to CSV
    if csv != None:
        file = df_desc.to_csv(csv)
        return file

    return df_desc
