# DEcancer Pipelines
# Installation
Create a virtual environment and install the required dependencies.

```
python -m venv venv
venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Usage
## Nanoparticles
Processed data is provided in `perseus_imputation/` for each of the 5 SPIONS and diluted plasma (DP). Note here that hyperparameters need to be specified for each algorithm which can be added to `src/parameters.py` when performing hyperparameter optimisation. Also not included is the steps taken for feature selection. Implementation fits between running the pipeline initially and before hyperparameter tuning found in `main.py`.

Hyperparameters are in the form of a tuple:
```py
params = (("name", (value1, value2, value3)),)
```

Run the pipeline with `python main.py`

## CancerSEEK
Data is found in `cancerseek_data/`. Similar to Nanoparticles, hyperparameters must be specified for hyperparameter optimisation found in `src/cancerseek/constants.py`. Additionally, the recursive feature elimination step has not been included for feature selection. Implementation can be added in `src/cancerseek/pipeline.py`

This pipeline is a 3-step process outlined in `main_cancerseek.py` with each section requiring different options. Run the pipeline with `python main_cancerseek.py`
