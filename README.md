# Early Prediction of Diabetes Using Machine Learning  

## 📌 Project Overview  
This project predicts diabetes risk using the **BRFSS 2015 dataset** with machine learning models: **Logistic Regression, Random Forest, and XGBoost**.  

Key contributions:  
- **Multiclass classification**: No Diabetes, Pre-Diabetes, Diabetes  
- **Explainable AI (SHAP)** for interpretability  
- **Deployment via Streamlit** for real-time prediction and lifestyle recommendations  

---

## 📂 Project Workflow  

1. **Step 1 – Run the Jupyter Notebook (`main.ipynb`)**  
   - Preprocess the dataset (missing values, scaling, stratified split)  
   - Train ML models (Logistic Regression, Random Forest, XGBoost)  
   - Evaluate performance (Accuracy, F1-score, Confusion Matrices)  
   - Generate **SHAP visualizations** for explainability  

2. **Step 2 – Run the Web App (`Dproject/app.py`)**  
   - Launches a **Streamlit web application**  
   - Users input health indicators (age, BMI, physical activity, etc.)  
   - The app provides **risk predictions, SHAP explanations, and lifestyle recommendations**  

---

## 📂 Project Structure  

├── main.ipynb # ML pipeline: preprocessing, training, evaluation, SHAP analysis
├── cleaning_data/
│ └── cleaned_data.csv # Preprocessed dataset
├── full_data_copy.csv # Raw dataset (BRFSS 2015)
├── Dproject/
│ ├── app.py # Streamlit app entry point
│ ├── datahandler.py # Handles data preprocessing
│ ├── diabetes_model.py # ML model training & evaluation
│ ├── recommendations.py # Personalized lifestyle recommendations
│ └── utils.py # Helper functions (plots, SHAP, formatting)
├── requirements.txt # Project dependencies
└── README.md # Project documentation (this file)

Run the ML Pipeline
jupyter notebook main.ipynb


streamlit run Dproject/app.py
📎 Links

GitHub Repository: MONABHANDARI/masterproject

Streamlit App (Deployed): PS C:\Users\user> python -m streamlit run "F:\master project\Diabetes\Dproject\app.py"

Demo Video: Ihttps://youtu.be/VkXDqwsN3CI
