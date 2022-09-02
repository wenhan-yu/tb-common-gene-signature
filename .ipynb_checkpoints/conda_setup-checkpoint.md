# Set up a conda environment with R and Python together 

### Create a conda environment with a specific name
```
$conda create --name tb-gene-signature-update
```
### Activate a speific conda environment and give a basic environment setup 
```
$conda activate tb-gene-signature-update
# Install python and R
$conda install -c conda-forge r-base=4.0.3 python=3.10.1 ipykernel ipyparallel r-irkernel 
#Put this new conda env in jupyterLab's kernel options
$ipython kernel install --user --name=tb-gene-signature-update
#Add the R-kernel to Jupyter by installing a kernel spec. This allows Jupyter to recognize the kernel and work with it interactively:
$R -e 'IRkernel::installspec()'
```

## R environment setup
Activate the conda environment before installing R package to ensure R packages will be installed in /shared/software/anaconda3/env/env-name/lib/R
```
#Preacquisition for biomRt and MetaIntegrator
$conda install -c conda-forge boost-cpp

#R
#Sys.setenv(RENV_PATHS_ROOT = "/shared/software/R-libraries")
#Sys.setenv(RENV_PATHS_CACHE = "/shared/software/R-libraries/cache")
#renv::init()

install.packages(c('renv','devtools','BiocManager','Jmisc','viridis','openxlsx'))
BiocManager::install(c('multtest','Biobase','GEOquery','GEOmetadb','limma','edgeR','AnnotationDbi','org.Hs.eg.db','biomaRt'))
install.packages('MetaIntegrator')

#####temporal solution for preprocessCore_1.55.2 installation from Github###
git clone https://github.com/bmbolstad/preprocessCore.git
cd preprocessCore/
R CMD INSTALL --configure-args="--disable-threading"  .
############################################################################

```

## Python environment setup
Activate the conda environment before installing to ensure python packages will be installed in /shared/software/anaconda3/env/env-name/lib/python3.6
```
#Install packages using conda install
$conda install -c conda-forge numpy scipy pandas matplotlib seaborn fastcluster networkx python-louvain openpyxl
#adding conda-forge to your channels (https://github.com/conda-forge/scikit-learn-feedstock)
$conda config --add channels conda-forge
$conda config --set channel_priority strict
$conda install scikit-learn
$pip install venn statannotations
$pip install --upgrade tsmoothie
```
