# Housing Price Predictor (ML Regressions)

A complete ML pipeline that trains, evaluates, and deploys 13 regression models to predict housing prices based on real estate data. Includes a Flask-based web app for live prediction and performance comparison.

>  Select a model, input housing features, and get price predictions directly from your browser.

---

### What it does
- Loads and preprocesses the `USA_Housing.csv` dataset
- Trains 13 different regression models:
  - Linear, Ridge, Lasso, ElasticNet
  - Polynomial Regression
  - SGD, Huber (Robust)
  - Random Forest, SVM, KNN
  - LightGBM, XGBoost
  - MLPRegressor (ANN)
- Evaluates each model using MAE, MSE, and R² score
- Saves all models as `.pkl` files
- Displays a table of model metrics
- Lets users select a model and input features for prediction

---

###  Demo
![demo.png]

---


---

### 💻 How to run it

```bash
git clone https://github.com/nikhilmahapatro/Housing_Price_Predictor_-MLRegressions.git
cd Housing_Price_Predictor_-MLRegressions

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Run model training
python model.py

# Launch web app
python app.py

