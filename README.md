# Gig Income Prediction Project

## Overview
The Gig Income Prediction project aims to develop a predictive model for estimating income from gig economy jobs. This project encompasses data processing, feature engineering, model training, and evaluation, providing a comprehensive framework for understanding and predicting gig income.

## Project Structure
- **data/**: Contains raw and processed datasets used for training and evaluation.
- **features/**: 
  - `feature_engineering.py`: Functions for building engineered features such as rolling means and inter-arrival times (ISI).
- **models/**: 
  - `pipeline.py`: Defines the preprocessing and training pipeline for the models, including data transformations and model selection.
  - `train.py`: Responsible for training multiple models and evaluating their performance based on specified metrics.
  - `evaluate.py`: Tests the trained models, generates performance metrics, creates plots, and assesses feature importances.
- **utils/**: 
  - `io_utils.py`: Provides helper functions for loading and saving data, including reading from and writing to various file formats.
  - `config.py`: Contains configuration variables such as paths for datasets, model parameters, and random seeds for reproducibility.
- **notebooks/**: Intended for exploratory analysis and prototyping, containing Jupyter notebooks for data exploration and visualization.
- **artifacts/**: Stores trained models, reports, and charts generated during the evaluation process.
- **main.py**: The entry script that runs the entire pipeline from data loading to model training and evaluation.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd gig_income_prediction
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the datasets by placing them in the `data/` directory.

## Usage
To run the entire pipeline, execute the following command:
```
python main.py
```

This will load the data, perform feature engineering, train the models, and evaluate their performance.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
