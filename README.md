## Determining the Predictive Limit of QSAR Models
#### Author: Scott Kolmar

### Project Description
The research done to evaluate how the predictivity of models are effected by error in 
either the training or the test set is simple to describe conceptually.  Benchmark datasets
are downloaded from reputable sources.  Then the datasets are split into training and test sets.
Randomized error is added and then models created on both error laden and native training sets.
Those models are used to predict both error laden and native test sets.  Differences in standard
statistics commonly used to assess predictivity are observed.

### Datasets
More information on the benchmark datasets used in this experiment is available in DataSets/README.md

### Execution of the Experiment
More information on running the experiment is documented in src/Features/README.md

### Results
More information on the results from a single run of the experiment are available in Results/README.md