## Determining the Predictive Limit of QSAR Models - Execution of the Experiment
#### Author: Scott Kolmar

### Script Development and Execution
All scripts here were created by Scott Kolmar, reviewed by Chris Grulke, and shared as open source to
support the transparency of this research effort and encourage peer review and reuse.  All conde snippets in this readme
are intended to be run on a Python command line from the base directory of the repo.

### Generating Padel Descriptors
The following code will generate a CSV file containing 2D Padel descriptors. A SMILES column and an endpoint column 
should be designated in the input dataframe.

```bash
# Library imports
import pandas as pd
import os
from src.Features.Preprocessing import descriptor_DataFrame

# Define input CSV file
dirname = os.path.normpath(os.path.dirname(__file__))
relpath = 'DataSets\Original_Files\g298atom_EXAMPLE.csv'
csvfile = os.path.join(dirname, '..', relpath)

# Transform into dataframe
df = pd.read_csv(filepath_or_buffer = csvfile, index_col = 0, header = 0)

# Run descriptor function
outrelpath = 'DataSets\SI Data Files\g298atom_desc_README_example.csv'
outputcsvfile = os.path.join(dirname, '..', outrelpath)
descriptor_DataFrame(df = df, smiles_col = 'smiles', output = 'g298_atom', csv = outputcsvfile)
```

### Generating Data for a Single Model
The following code performs all the tasks necessary to generate a single model. Various preprocessing options are available
in the function docstring. A figure is shown but not saved using the current parameters.

```bash
# Library imports
import pandas as pd
from src.Features.Modeling import makeModels
import os

# Define input CSV file
dirname = os.path.normpath(os.path.dirname(__file__))
relpath = 'DataSets\SI Data Files\g298atom_SI.csv'
csvfile = os.path.join(dirname, '..', relpath)

# Transform data into a Pandas dataframe
df = pd.read_csv(filepath_or_buffer = csvfile,
                 index_col = 0,
                 header = 0)

# Filter dataframe so that the endpoint column is the final column
df = df.iloc[:, :-2]

# Sample dataframe to manageable size
df = df.sample(n = 1000)

# Run modeling and plotting function
makeModels(df = df,
           algs = 'ridge',
           save_fig = False,
           show_fig = True,
           )

```


### Generating Data for a Single Algorithm with Error Laden Data
The following code performs all the tasks necessary to generate an RMSE/R2 versus error plot for the Ridge algorithm, as
found in the manuscript.

The code will perform the following steps:
1. Take a CSV file containing Padel descriptor values and an experimental endpoint
   * This CSV file must be of the form provided in this repository, namely with an index column first, followed by all the 2D Padel descriptors, an experimental endpoint column, and then two QSAR smiles columns.
   * This form is required because the modeling function locates the feature and endpoint columns by numerical index rather than name.
2. Generate a series of error laden data with a user specified amount of added error
   * Error is added as (base_sigma)*(multiplier) for each level
   * User can specify the base sigma; if none is provided, the range of the dataset will be used
   * User can specify multiplier
   * User can specify number of noise levels added to dataset
3. Generate models using the user specified algorithm on each level of error laden data
   * There are a User specified number of repititions on a User specified number of noise levels, and each rep has its own model
   * Each model is optimized using GridSearchCV
4. Generate a PKL file containing all the information from the models, including:
    * Best optimized parameters for each individual model
    * Each error laden dataset
    * Performance metrics for each model
5. Generate a PNG file which shows the plot of the performance metrics versus added error
   * Both the x-axis and y-axis can be normalized to the base_rmse (RMSE obtained on a noiseless dataset) via User specified parameters

```bash
# Library imports
import os
from src.Features.Modeling import modelOnNoise
from src.Features.Plotting import plotErrors

# Define input CSV file
dirname = os.path.normpath(os.path.dirname(__file__))
relpath = 'DataSets\SI Data Files\g298atom_SI.csv'
csvfile = os.path.join(dirname, '..', relpath)

# Define output PKL file
relpklpath = 'Results'
outpklfile = 'g298_15_5_0p01_ridge_READMETest.pkl'
pklfile = os.path.join(dirname, '..', relpklpath, outpklfile)

# Run modeling function
modelOnNoise(file = csvfile,
             out = pklfile,
             n_samples = 1000,
             multiplier = 0.01,
             reps = 5,
             n_levels = 15,
             alg = 'ridge',
             set_search = 'GSV',
             )

# Define output PNG file
relpngpath = 'Results'
outpngfile = 'g298_15_5_0p01_ridge_READMETEST.png'
pngfile = os.path.join(dirname, '..', relpngpath, outpngfile)

# Run plotting function
plotErrors(pkl = pklfile,
           n_levels = 15,
           multiplier = 0.01,
           out_path = pngfile,
           x_normalized = True,
           y_normalized = True)

```
If a User wishes to generate data for several algorithms, these function calls can be repeated with the abbreviations
for the other supported algorithms found in the docstrings of the functions. These include k-Nearest Neighbors,
Support Vector Machines, Decision Trees, Random Forest, and Gaussian Process.

### Generating Data for the Gaussian Process algorithm
When generating data using the Gaussian Process algorithm, some unique parameters can be set, such as: 
   * The kernel (set_kernel)
     * Any kernel found in sklearn's documentation for Gaussian Process Regressor can be used https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
   * The optimizer used for optimizing kernel parameters (gp_opt)
     * optimCOBYLA and optimTNC can be used as an alternative to 'L-BGFS-B', which is invoked when None is called.
   * The option to normalize the endpoint column of the dataset (set_normy)
   * The option to incorporate uncertainty in the experimental endpoint (use_alpha)

### Generating Gaussian Process Prediction Error Plots
The following code will generate a prediction error plot from Gaussian Process data.

```bash
# Library imports
from src.Features.Plotting import plotStdArraysGP
import os

# Define input CSV file
dirname = os.path.normpath(os.path.dirname(__file__))
relpath = 'DataSets\SI Data Files\Alpha_SI.csv'
csvfile = os.path.join(dirname, '..', relpath)

# Define input PKL file
relpklpath = 'Results\Alpha\PKL\Alpha_15_5_0p01_gp.pkl'
pklfile = os.path.join(dirname, '..', relpklpath)

# Define output PNG file
relpngpath = 'Results'
pngdirectory = os.path.join(dirname, '..', relpngpath)

# Run plotting function
plotStdArraysGP(csv = csvfile,
                pkl = pklfile,
                out = pngdirectory,
                normalized = True)

```

