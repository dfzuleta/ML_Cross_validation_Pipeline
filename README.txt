ML_Cross_Validation
===================

This project provides a reusable pipeline to evaluate multiple machine learning models using different cross-validation strategies. It also includes preprocessing steps, feature selection via Lasso, and metric visualization.

📁 Project Structure
--------------------

ML_Cross_Validation/
├── main.py                   # Main script to run the workflow
├── requirements.txt          # Python dependencies
├── outputs/                  # Model results and charts (generated after running)
├── data/                     # (Optional) Your dataset here
├── src/                      # Project modules
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── feature_selection.py
│   └── visualization.py

🚀 How to Use
--------------

1. Clone the repository:

   git clone https://github.com/dfzuleta/ML_Cross_Validation.git
   cd ML_Cross_Validation

2. Install dependencies (Python 3.8+ recommended):

   pip install -r requirements.txt

3. Add your data:

   Update main.py:

       X = pd.read_csv("data/your_dataset.csv")
       y = X.pop("target_column")
       Columnas_numero = ["column1", "column2", ...]

4. Run the pipeline:

   python main.py

This will:
- Encode categorical variables
- Scale numerical features
- Train and evaluate multiple models
- Save results to outputs/resultados_modelos.xlsx
- Generate performance charts
- Perform feature selection using Lasso

🧪 Optional: Run in Spyder or VS Code
-------------------------------------
1. Open the ML_Cross_Validation folder as your working directory.
2. Open main.py
3. Run sections step-by-step (Spyder supports # %% cell separation)

📊 Output Files
----------------
All results and plots are saved in the outputs/ directory:
- resultados_modelos.xlsx: evaluation scores
- Accuracy_comparativa_modelos.png, F1 Score_...: charts for each metric

📦 Requirements
----------------
All dependencies are listed in requirements.txt. If needed:

   pip install pandas scikit-learn matplotlib seaborn openpyxl

