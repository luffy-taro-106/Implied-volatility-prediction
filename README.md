# 🧠 Implied Volatility Forecasting using Machine Learning

This repository presents a machine learning pipeline to forecast Implied Volatility (IV) for NIFTY 50 Index Options based on historical option chain data.

The notebook and Python script cover the full workflow, from data preprocessing to model evaluation, using clean, structured code and reusable components.

---

## 📁 Project Structure

```
.
├── model.py               # Python script for reusable model training
├── pipeline.ipynb         # Structured Jupyter Notebook for exploration
├── train_data.parquet     # Training dataset (from NK Securities)
├── test_data.parquet      # Test dataset (from NK Securities)
├── README.md              # Documentation
```

---

## 📊 Overview

- **Objective:** Predict implied volatility (IV) for option contracts
- **Data Source:** Option chain data provided by **NK Securities**
- **Tech Stack:** Pandas, Scikit-learn, XGBoost, Parquet, Jupyter
- **Output:** Predicted IV for test contracts, suitable for quantitative trading strategies

---

## ⚙️ Setup Instructions

**Clone the repository:**
   ```bash
   git clone https://github.com/<luffy-taro-106>/Implied-volatility-prediction.git
   cd Implied-volatility-prediction
   ```
---

## 📈 Pipeline Stages

1. **Data Melting:** Convert wide-format IV data into long format
2. **Feature Engineering:**
   - Moneyness
   - Time to expiry
   - Option type (call/put)
3. **Scaling and Cleaning:**
   - Filter extreme IV values
   - Normalize features
4. **Modeling:**
   - XGBoost Regressor
   - Performance metrics (MSE, R², visualization)
5. **Prediction and Output:**
   - Forecast IV for test data
   - Generate output for submission or strategy integration

---

## 📌 Example Features Used

- Moneyness = Strike / Spot
- Days to Expiry
- Option Type Encoding
- One-hot encoded Expiry Dates

---

## 🔬 Model Summary

- **Model Used:** XGBoost Regressor
- **Target:** Implied Volatility (`iv`)
- **Metrics:** Mean Squared Error (MSE), R² Score, Visual Comparison
- **Advantages:** Fast training, good generalization, interpretable feature importance

---

## 📤 Results

Final output includes a `.csv`  file with predicted IVs for given test option contracts.

---


## ⚠️ Disclaimer & License

📂 **Data Usage:**  
The dataset used in this project was **provided by NK Securities** for research and academic purposes only.  
Unauthorized redistribution, publication, or commercial usage of this data is strictly prohibited.

📄 **License:**  
This project is for educational and non-commercial use only. Please cite appropriately if reused.

---

## 👤 Author

Maintained by: **Darshan Sonawane**  
📬 Email: [dsonawane1110@gmail.com]
