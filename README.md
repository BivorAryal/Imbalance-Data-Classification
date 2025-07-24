## Classification on Imbalanced Data: Insurance Claim Prediction
A machine learning project addressing class imbalance in insurance claim datasets using Python.

### Overview
This project tackles the challenge of imbalanced data in classification tasks, where the majority class (no claims) vastly outnumbers the minority class (claims). Using an insurance claim dataset (58,592 entries, 41 features), we:
      Perform exploratory data analysis (EDA) to visualize class imbalance and feature distributions.
      Apply oversampling (to balance classes) and feature selection (to identify key predictors).
      Train a Random Forest classifier and evaluate performance using metrics like precision, recall, and F1-score.
      
### Key Result: The model achieves 99% accuracy with 98% precision and 100% recall for the minority class.

### Tools & Techniques
      Python Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn
      Data Resampling: Oversampling (to address class imbalance)
      Feature Selection: Random Forest-based importance ranking
      Model: Random Forest Classifier
      Evaluation Metrics: Precision, Recall, F1-Score, AUROC

### Model Performance
Classification Report:

Class            	Precision	Recall	F1-Score	Support
0 (No Claim)	1.00	      0.98	      0.99	      16,574
1 (Claim)	      0.98	      1.00      	0.99      	16,333
Accuracy			                  0.99	      32,907
Macro Avg	      0.99	      0.99      	0.99      	32,907
Weighted Avg	0.99	      0.99	      0.99	      32,907

### Interpretation:
      Class 0 (No Claim): Perfect precision (1.00), high recall (0.98).
      Class 1 (Claim): Near-perfect recall (1.00), precision (0.98).
      Macro/Weighted Averages: 0.99 across all metrics, indicating balanced performance.

### Key Steps
1.Exploratory Data Analysis (EDA)
      - Visualized class imbalance and feature distributions.
      - Analyzed correlations between features and claim_status.
2.Handling Class Imbalance
      - Used oversampling (minority class replicated) to balance classes (54,844 entries each).
3.Feature Selection
      - Identified top 10 influential features (e.g., customer_age, vehicle_age, region_code).
      - Dropped non-predictive features like policy_id.
4.Model Training & Evaluation
      - Trained a Random Forest Classifier on the balanced dataset.
      - Achieved 99% accuracy with high recall (100%) for the minority class.



