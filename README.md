
# 💳 Credit Risk Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-success" />
  <img src="https://img.shields.io/badge/Project-Credit%20Risk-orange" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-live-success" />
  <img src="https://img.shields.io/badge/deployed-Streamlit%20Cloud-brightgreen" />
  <img src="https://img.shields.io/badge/license-educational-lightgrey" />
</p>

---

## 🌐 Live Application

🚀 **Deployed App**
👉 [https://credit-system.streamlit.app/](https://credit-system.streamlit.app/)

📁 **GitHub Repository**
👉 [https://github.com/Sahilkumar8084/-Credit-Risk-Prediction-System.git](https://github.com/Sahilkumar8084/-Credit-Risk-Prediction-System.git)

---

## 🧠 Introduction

The **Credit Risk Prediction System** is a **machine learning–based Streamlit web application** that predicts whether a customer is **creditworthy (Low Risk)** or **high risk (High Risk)** based on financial and personal attributes.

This project demonstrates a **real-world ML use case** commonly applied in **banking and finance**, showcasing model inference and deployment.

---

## 📌 Project Overview

Financial institutions use credit risk models to:

* 💰 Reduce loan default rates
* 📊 Assess borrower reliability
* 🏦 Improve lending decisions

This application allows users to input customer details and instantly receive a **credit risk prediction**.

---

## 🎯 Objective

To build a **production-ready ML web application** that:

* Accepts user financial data
* Applies consistent preprocessing
* Predicts credit risk accurately
* Displays results in a clean UI

---

## 🧠 Machine Learning Approach

### 🔹 Problem Type

* **Binary Classification**

  * `0` → Low Credit Risk
  * `1` → High Credit Risk

### 🔹 Model Used

* **Classification Model** (e.g., Logistic Regression / Tree-based model)

  * Suitable for structured financial data
  * Interpretable predictions
  * Industry-relevant use case

---

## 📊 Features Used

| Feature              | Description              |
| -------------------- | ------------------------ |
| Age                  | Applicant age            |
| Income               | Annual income            |
| Loan Amount          | Requested loan amount    |
| Credit History       | Past credit behavior     |
| Employment Status    | Job stability            |
| Debt-to-Income Ratio | Financial risk indicator |
| Loan Duration        | Repayment period         |

*(Exact features may vary based on dataset)*

---

## 🔄 Data Preprocessing

The following steps are applied to ensure **model accuracy**:

* Categorical encoding
* Numerical feature scaling
* Handling missing values
* Feature alignment with trained model

✔️ Same preprocessing logic is used during **training and inference**

---

## 🖥️ Web Application (Streamlit)

### UI Highlights

* 📋 Organized input fields
* ▶️ Predict button
* 📊 Clear risk classification output
* 🎨 Clean, responsive layout
* 🌐 Fully cloud-deployed

### Prediction Output

* ✅ **Low Credit Risk**
* ❌ **High Credit Risk**

---

## 📁 Project Structure

```text
Credit-Risk-Prediction-System/
│
├── models/
│   ├── scaler.pkl
│   └── model.pkl
│
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
├── README.md              # Documentation
└── venv/                  # Virtual environment
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sahilkumar8084/-Credit-Risk-Prediction-System.git
cd Credit-Risk-Prediction-System
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

App runs at:

```
http://localhost:8501
```

---

## 📦 Requirements

* `streamlit`
* `pandas`
* `scikit-learn`
* `joblib`
* `numpy`

---

## 🧪 Model Inference Flow

```text
User Input
   ↓
Preprocessing
   ↓
Feature Scaling
   ↓
ML Model
   ↓
Credit Risk Prediction
```

---

## 🚀 Future Improvements

* Add prediction probability scores
* Improve model accuracy with ensemble models
* Add loan approval recommendation
* Store predictions in a database
* User authentication
* Detailed model evaluation metrics

---

## 🏆 Learning Outcomes

* Real-world ML application in finance
* End-to-end ML deployment
* Streamlit UI development
* Handling production inference pipelines
* Writing industry-ready ML projects

---

## 👨‍💻 Author

**Sahil Kumar**
Machine Learning Enthusiast
India 🇮🇳

---

## 📜 License

This project is intended for **educational and learning purposes**.

---

⭐ **Live, deployed & internship-ready project!**
