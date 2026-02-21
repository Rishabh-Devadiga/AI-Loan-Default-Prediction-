# Loan Default Prediction System

This project is an end-to-end Loan Default Prediction System that utilizes machine learning to predict the likelihood of loan defaults based on borrower data. The system includes data processing, model training, and a user-friendly multi-page Streamlit application that provides business insights. 

# Deployed App Link:
https://ai-loan-default-prediction.streamlit.app/

## Project Structure

```
loan-default-prediction
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── pages                 # Contains different pages for the Streamlit app
│   │   ├── 1_overview.py     # Overview Dashboard page
│   │   ├── 2_insights.py     # Loan Risk Prediction page
│   │   └── 3_predict.py      # Model Performance page
│   ├── data_processing        # Data processing scripts
│   │   ├── preprocess.py      # Data preprocessing tasks
│   │   └── features.py        # Feature extraction and preparation
│   ├── models                # Machine learning model scripts
│   │   ├── train.py           # Model training logic
│   │   ├── inference.py       # Inference process for predictions
│   │   └── model_utils.py     # Utility functions for model handling
│   ├── utils                 # Helper functions
│   │   └── helpers.py         # Various utility functions
│   └── config.py             # Configuration settings
├── data
│   ├── raw                   # Raw dataset
│   │   └── Loan_Default.csv   # Dataset used for training
│   └── processed             # Processed dataset
│       └── processed_data.csv # Dataset after preprocessing
├── notebooks                 # Jupyter notebooks for analysis
│   └── eda.ipynb             # Exploratory Data Analysis notebook
├── models                    # Saved models
│   └── best_model.pkl        # Best-performing model
├── tests                     # Unit tests
│   └── test_pipeline.py      # Tests for data processing and model training
├── requirements.txt          # Project dependencies
├── .gitignore                # Files to ignore in version control
└── README.md                 # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd loan-default-prediction
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Navigate through the different pages:
   - **Overview Dashboard**: View portfolio metrics, risk distribution, and default trends.
   - **Loan Risk Prediction**: Input borrower details and get predictions along with risk gauges and recommendations.
   - **Model Performance**: Compare different models and view their performance metrics.

## Dataset

The dataset was taken from Kaggle.
The raw dataset `Loan_Default.csv` is located in the `data/raw` directory. After preprocessing, the processed dataset can be found in `data/processed/processed_data.csv`.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
