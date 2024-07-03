# switch to project folder
cd build-ml-pipeline-for-short-term-rental-prices

# create project environment
conda env create -f environment.yml
conda activate nyc_airbnb_dev

# check if connected to wandb
wandb login

# download data
mlflow run . -P steps="download"

# eda
mlflow run src/eda

# clean data
mlflow run . -P steps="basic_cleaning"

# data checks
mlflow run . -P steps="data_check"

# split data
mlflow run . -P steps="data_split"

# train model
mlflow run . -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_depth=10,50,100 modeling.random_forest.n_estimators=100,200,500 -m"

# test model
mlflow run . -P steps=test_regression_model