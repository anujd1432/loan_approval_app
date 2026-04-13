🏦 Loan Approval Predictor
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
A machine learning web application that predicts loan approval status based on applicant details using 
Logistic Regression. Built with Streamlit for an interactive and user-friendly experience.


🔗 Live Demo: Click here to try the app
---https://loanapprovalapp-63saptmhgg2z8h4y9zsgnp.streamlit.app/
📸 Preview
---
✨ Features
🔍 Instant Prediction — Real-time loan approval prediction
📊 Confidence Score — Shows approval/rejection probability
📈 Visual Chart — Probability bar chart for better understanding
📋 Input Summary — Clean summary of all entered details
📱 Responsive UI — Works on desktop and mobile browsers
⚡ Fast & Lightweight — Cached model loading for quick responses
---
🧠 Model Details
Property	Details
Algorithm	Logistic Regression
Features	17 (one-hot encoded)
Target	Loan Approved (1) / Not Approved (0)
Libraries	Scikit-learn, Pandas, NumPy
Input Features Used:
Gender, Marital Status, Dependents
Education, Self Employment Status
Applicant & Co-Applicant Income
Loan Amount & Loan Term
Credit History
Property Area
---
🗂️ Project Structure
```
loan_approval_app/
│
├── app.py               # Main Streamlit application
├── model.pkl            # Trained Logistic Regression model
├── features.pkl         # Feature names for model input
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```
---
🚀 Run Locally
1. Clone the repository
```bash
git clone https://github.com/your-username/loan_approval_app.git
cd loan_approval_app
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```
4. Open in browser
```
http://localhost:8501
```
---
📦 Requirements
```
streamlit
joblib
pandas
numpy
scikit-learn
```
---
🌐 Deploy on Streamlit Cloud
Push code to GitHub
Go to share.streamlit.io
Connect your GitHub repository
Set main file as `app.py`
Click Deploy 🎉
---
📊 How It Works
```
User Input → Feature Engineering → One-Hot Encoding → Logistic Regression → Prediction
```
User fills in applicant details via the web form
Input is transformed to match model's expected feature format
Trained Logistic Regression model predicts the outcome
Result is displayed with confidence probability
---
🙋‍♂️ Author
Amand — B.Tech Student | ML Enthusiast
![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)
---
📄 License
This project is licensed under the MIT License — feel free to use and modify!
---
⭐ If you found this helpful, please give it a star!
