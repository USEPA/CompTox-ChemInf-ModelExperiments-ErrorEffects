## Determining the Predictive Limit of QSAR Models - DataSets
#### Author: Scott Kolmar

### DataSet selection and collection
Datasets used in this research product were collected from 3 sources: MoleculeNet, ToxCast, and Gadaleta et al.
Datasets downloaded from these sources are available in the Datasets/Original_Files folder.  Datasets were 
selected to span a variety of endpoint complexities from quantum mechanical datasets to in vivo toxicity to be 
representative of the datasets commonly used in QSAR modeling.

The following files available in *DataSets/Original_Files* were taken from MoleculeNet
(http://moleculenet.ai/datasets-1):

* *G298Atom_Alpha_Orig.csv* on 09/30/2020
  * *https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv*
* *Lip_Orig.csv* on 10/01/2020
  * *https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv*
* *Solv_Orig.csv* on 10/01/2020
  * *https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv*
* *BACE_Orig.csv* on 10/26/2020
  * *https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv*

The file *Tox_Orig.xslx*, available in *DataSets/Original_Files*, was taken from the EPA's ToxCast website,
(https://www.epa.gov/chemical-research/exploring-toxcast-data-downloadable-data), on 11/29/2020.
The entire MySQL database, along with an associated package written in the R programming language,
can be downloaded at (https://doi.org/10.23645/epacomptox.6062623).

* The file *Tox_Orig.xslx* contains three sheets; sheet *pparg.aeids* contains information about assay IDs,
  sheet *mc5* contains data used in this project, and sheet *mc7* contains data not used in this project
* The following steps were taken to process the data from the MySQL database and produce *Tox_Orig.xslx* Sheet *mc5*:
  * The *tcplSubsetChid* function was called to obtain unique *DTXSID/aeid* relationships
  * This selection of data was loaded as a Pandas dataframe
  * The *dsstox_substance_id* column was set as the index
  
* The following steps were taken to produce the Tox102 dataset:
  * The dataframe was filtered to only include entries where *aeid* = 102
  * All entries containing white space were filled with np.nan
  * All rows containing np.nan in the *modl_ga* column were dropped
  * All duplicate rows were dropped
  * All columns except for *dsstox_substance_id* and *modl_ga* were dropped
  
* The following steps were taken to produce the Tox134 dataset:
  * The dataframe was filtered to only include entries where *aeid* = 134
  * All entries containing white space were filled with np.nan
  * All rows containing np.nan in the *modl_ga* column were dropped
  * All duplicate rows were dropped
  * All columns except for *dsstox_substance_id* and *modl_ga* were dropped

The file *LD50_Orig.txt*, available in DataSets/Original_Files, was downloaded from the National Toxicology Program website
from the National Institute of Environmental Health and Safety and the National Institute of Health. 
(https://ntp.niehs.nih.gov/whatwestudy/niceatm/test-method-evaluations/acute-systemic-tox/models/index.html)
The dataset was downloaded on 11/25/2020.
* The following processing steps were taken before descriptor generation:
    * Dataset loaded as Pandas dataframe
    * *DTXSID* was set as the index
    * Duplicates were dropped from the column 'Canonical_QSARr'
    * Blank entries were replaced with *np.nan*
    * Rows with *np.nan* were dropped

The file *g298atom_EXAMPLE.csv* available in *DataSets/Original_Files* is intended as an example file to support 
testing of the source code in *src*.  Please see *src/README.md* for more information.

### Descriptor Files
All datasets with their generated descriptors are available in *Datasets/SI_Data_Files*.

For some datasets, descriptors were generated using the OPERA software, developed at the US
Environmental Protection Agency. (https://github.com/NIEHS/OPERA)
The OPERA software uses Padel as a backend to generate molecular descriptors.
Descriptors were generated with the "Standardize" option On.

The function *descriptor_DataFrame* in *src/Preprocessing.py* and OPERA utilize essentially the same Padel backend,
but with slightly different settings for the underlying Padel software. Due to these small differences, they produce
similar but not completely identical results. The OPERA software is faster and can run in the background, which can be advantageous to the user.

The following datasets used descriptors generated in OPERA:
* Alpha
* Solv
* Lip
* BACE
* Tox102
* Tox134
* LD50


