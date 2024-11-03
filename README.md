Online-Bookstore-Analytics/
│
├── README.md                        # Project overview and instructions
├── report/
│   ├── project_report.docx          # Final project report document
│   └── images/                      # Folder for images used in the report
│       ├── average_order_value_by_month_and_year.png
│       ├── average_total_amount_per_top_10_product_categories.png
│       ├── correlation_heatmap_of_numerical_features.png
│       ├── distribution_of_total_order_value.png
│       ├── monthly_orders_count_by_year.png
│       ├── relationship_between_order_quantity_and_total_order_value.png
│       └── top_10_most_popular_product_categories.png
│
├── data/
│   ├── preprocessed_data/              # Folder for processed datasets
│   │   ├── X_train.csv                 # Preprocessed training features
│   │   ├── X_test.csv                  # Preprocessed test features
│   │   ├── y_train.csv                 # Training labels/target values
│   │   └── y_test.csv                  # Test labels/target values
│   └── online_bookstore_dataset.csv    # Full dataset used for analysis
│
├── scripts/
│   ├── data_preparation.py                     # Script for data cleaning and preprocessing
│   ├── model_training.py                       # Script for training the models
│   ├── model_optimization.py                   # Script for hyperparameter tuning
│   ├── model_evaluation.py                     # Script for evaluating model performance
│   └── exploratory_data_analysis_EDA.py        # Script for Exploratory Data Analysis (EDA)
│
└── results/
    ├── model_metrics.txt               # Text file with evaluation metrics and scores
    ├── model_optimization_results.txt  # Hyperparameter tuning results
    └── model_evaluation_results.txt    # Final model evaluation summary
