## Common TB gene signature model development 
###   Network-based meta-analysis combining with machine-learning modeling 

This repository contains a collection of scripts and notebooks working towards the development of a common gene signature model predictive of stage of TB disease progression as well as monitoring treatment response.

 For the detail, check out the preprint on bioRxiv (https://www.biorxiv.org/content/10.1101/2022.11.28.518302v3.abstract)

### code
* The project are composed of four parts
    - Data process: microarray and sequencing data process (The sequencing raw data were downloaded and processed separately by our bulk RNAseq pipeline (not included in this repository) where "Salmon" is used to align and quantify the sequencing reads and runs on the AWS parallel computing platform)
        * `r-df-process.ipynp`: This script processes the datasets for network analysis as well as training model building. 
        * `r-validate-data-process.ipynp`: This script processes the independent datasets for model validation.
        * `r-viralinfect-data-process.ipynp`: This script processes the independent viral infection datasets for model validation.
        * `mega-data-list-model-building.csv`: The list of the datasets and its conditions used for network analysis and the training model construction.
        * `validate_data_list.csv`: The list of the datasets used for ML model validation.
         * `viral infection datasets.xlsx`: The list of the viral infection datasets used for ML model validation.
        
    - Network analysis: Build the gene covariation network and generate a set of genes for ML 
        * `py-network-building.ipynp`: Build the gene covariation network across all training datasets
        * `py-gene-signature.ipynp`: Identify a set of differentially expressed genes which respond consistently across multiple clinical conditions
        
    - Machine-learning model building: Build the predictive model for TB progression risk estimation and TB treatment response monitoring
        * `py-predictive-model.ipynp`: Given the gene signature from network analysis, this script perform feature selection, search the optimal model and its hyperparameters based on the training datasets
    
    - Ml model validation: Use the independent datasets to evaluate the predictive model 
        * `py-model-validation-progression-individual.ipynp`: Model validation in each of independent TB progression cohorts.  
        * `py-model-validation-progression-pool.ipynp`: Model validation in a combination of independent TB progression cohorts. Evaluate incipient TB and active TB diagonsis. Establish a probabilistic model for TB risk estimation. 
        * `py-model-validation-treatment.ipynp`: Model validation in each of independent TB treatment cohorts.  Evaluate model prediction in treatment monitoring and clinical outcomes and how the predictive score correlating with bioassay/microbiological results.
        

### fun/
* All the R and python functions go here.
        
### data/
* All the figures in pdf format generated in the code are saved here.




