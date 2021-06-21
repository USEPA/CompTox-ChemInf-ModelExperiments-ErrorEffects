# Library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import MultipleLocator
from yellowbrick.regressor import ResidualsPlot
import pprint

# Functions

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

def plotErrors(pkl=None, n_levels=15, multiplier = 0.01, out_path = None, x_normalized = False, y_normalized = False):
    """
    Builds plots of RMSE versus noise level and R2 versus noise level. Takes a .pkl file as an input, extracts RMSEs and
    R2s for each noise level and repetition, and shows the plots.

    Parameters:
    pkl (string): PKL file which contains RMSEs and R2s organized by noise level and repetition. (Default = None)
    n_levels (int): Number of noise levels in the .pkl file. Should match 'n_levels' parameter of 'modelOnNoise'.
                    (Default = 15).
    multiplier (float): Multiplier used to generate noise levels. (Default = 0.01)
    out_path (str): Absolute path to save the figure to. If None is provided then figure won't be saved. (Default = None)
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
    x_tick_at = round(max(xaxis)/6)
    if x_tick_at < 1:
        x_tick_at = round(max(xaxis)/6) + 0.2
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
    df = pd.read_csv(filepath_or_buffer = csv, index_col = 0, header = 0)
    df = df.iloc[:, :-2]

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

