## Determining the Predictive Limit of QSAR Models - Results
#### Author: Scott Kolmar

### Result files
Running the experiment results in a multitude of statistics and graphics for researchers to review.  This directory
contains the exemplar results used for the development of the manuscript, but re-running the software will necessarily 
(due to the random nature of the error insertion) lead to results that are not identical.

Results for the Alpha dataset are included for direct download in the *Alpha* directory, but due to file sizes, results for the remaining datasets
are provided as .zip files. 

The files in the Alpha directory were generated on *10/15/20*.
The files in *Results.zip* were generated between *10/01/20* and *01/31/21*.

**File Naming Conventions Used**

For each file, details about the modeling method are separated by '_'. This is done for PKL files, which store all the data, and PNG files, which contain plots.

For example, in the string *A_B_C_D__E.pkl* :



* A = dataset
* B = number of error levels generated
* C = number of repetitions generated at each error level
* D = multiplier used to generate the amount of error at each level
* E = algorithm

For the file *Alpha_15_5_0p01_gp.pkl* :
* Dataset = Alpha
* n_levels = 15
* n_reps = 5
* multiplier = 0.01
* algorithm = Gaussian Process

**Included Directories and Files**
* *Data Distribution*
  * Directory contains plots showing data distribution figures
    * *alpha_box_viol.png* contains a box and whisker plot as well as a violin plot to show the distribution of endpoint
    values in the dataset
    * *Alpha_hist.png* contains a histogram of endpoint values in the dataset
* *GP*
  * Directory contains prediction error plots for the Gaussian Process algorithm
    * *Alpha_15_5_0p01_gp.png* contains a prediction error versus added random error plot
    for the Gaussian Process algorithm applied to this dataset.
    * *Alpha_15_5_0p01_gp_witherror.png* contains a prediction error versus added random error plot for the Gaussian Process algorithm,
    but with experimental error information provided to the algorithm.
* *PKL*
  * Directory contains PKL files which store all the data needed to generate figures for this project. Each PKL file is 
    a python dictionary, organized by error level. For more information regarding specific organization, look in the source 
    code in *src/Modeling.py*, in the *modelOnNoise* function. Information about the original dataset, the dataset with 
    added error, algorithm hyperparameters, and test set scores are all included. Pretty printing the dictionary can be 
    instructive.
    * *Alpha_15_5_0p01_gp.pkl* contains RMSE/R2 versus added experimental error data for the Gaussian Process algorithm
    * *Alpha_15_5_0p01_gp_witherror.pkl* contains RMSE/R2 versus added experimental error data for the Gaussian Process
    algorithm, but with experimental error information provided to the algorithm
    * *Alpha_15_5_0p01_knn.pkl* contains RMSE/R2 versus added experimental error data for the k-Nearest Neighbors algorithm
    * *Alpha_15_5_0p01_rf.pkl* contains RMSE/R2 versus added experimental error data for the Random Forest algorithm
    * *Alpha_15_5_0p01_ridge.pkl* contains RMSE/R2 versus added experimental error data for the Ridge algorithm
    * *Alpha_15_5_0p01_svr.pkl* contains RMSE/R2 versus added experimental error data for the Support Vector Regressor algorithm
  
* PNG
  * Directory contains PNG files, which contain plots for RMSE/R2 verus added experimental error results. For specific information about plotting
    parameters, see *src/Plotting.py*, specifically the *plotErrors* function.
    * *Alpha_15_5_0p01_gp.png* contains RMSE/R2 versus added experimental error data for the Gaussian Process algorithm
    * *Alpha_15_5_0p01_gp_witherror.png* contains RMSE/R2 versus added experimental error data for the Gaussian Process
    algorithm, but with experimental error information provided to the algorithm
    * *Alpha_15_5_0p01_knn.png* contains RMSE/R2 versus added experimental error data for the k-Nearest Neighbors algorithm
    * *Alpha_15_5_0p01_rf.png* contains RMSE/R2 versus added experimental error data for the Random Forest algorithm
    * *Alpha_15_5_0p01_ridge.png* contains RMSE/R2 versus added experimental error data for the Ridge algorithm
    * *Alpha_15_5_0p01_svr.png* contains RMSE/R2 versus added experimental error data for the Support Vector Regressor algorithm
