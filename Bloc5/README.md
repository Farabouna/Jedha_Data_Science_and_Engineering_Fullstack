## Jedha_Data_Science_and_Engineering_Fullstack
## Repository for Certification
# 🚗 GetAround Analysis

GetAround is the **Airbnb for cars**, allowing users to rent vehicles for a few hours or days. Founded in **2009**, the platform now boasts over **5 million users** and **20,000 available cars worldwide**.

---

## 📌 Context

When renting a car, users must complete a **check-in** and **check-out flow** to:

- **Assess the car’s condition** (damage reporting).
- **Compare fuel levels** before and after rental.
- **Track the kilometers driven**.

### ✅ Check-in & Check-out Methods
GetAround offers three ways to check in and check out a vehicle:

1. **📱 Mobile Rental Agreement:** The driver and owner meet in person, signing the agreement on the owner’s smartphone.
2. **🔑 Connect (Keyless Access):** The driver opens the car using their smartphone without meeting the owner.
3. **📝 Paper Contract (Rarely Used).**

---

## 🚧 Project Overview

GetAround drivers book cars for a specific **time period**, but **late returns** at checkout cause **major friction** for the next driver if the car was scheduled for another rental.

To solve this, **GetAround implemented a minimum delay between rentals**, preventing cars from appearing in search results if bookings are too close together.

### 🎯 Goals

The **Product Manager** needs insights to make decisions regarding:

- **⏳ Threshold:** What should be the minimum delay between rentals?
- **🚗 Scope:** Should the feature apply to all cars or just **Connect** cars?

### 🔍 Key Analyses

- What **percentage of owner revenue** is affected by this feature?
- How many rentals would be impacted based on **different delay thresholds**?
- How often are **drivers late** for check-in? What is the impact?
- How many issues could this feature **prevent**?

---

## 🏗 Project Breakdown

### **📊 Part 1: Data Analysis & Web Dashboard**
- **Analyze** rental delays and their impact.
- **Develop a Streamlit dashboard** to help the Product Management team visualize key insights.

### **🧠 Part 2: Machine Learning (Pricing Optimization)**
The **Data Science team** is optimizing car rental prices using **Machine Learning**.

1️⃣ **Train & Manage Model via MLflow**
   - A **Linear Regression model** is trained to **suggest optimal rental prices**.
   - The model is deployed using **MLflow**, managing multiple experiments and storing artifacts.

2️⃣ **Make Predictions via API**  
   - The API provides a **/predict** endpoint to return price predictions.
   - Hosted on **Hugging Face Spaces**, it can be accessed via:  
     🔗 **[https://farabouna-getaroudapispace.hf.space/predict](https://farabouna-getaroudapispace.hf.space/predict)**  

### **📜 Part 3: API Documentation**
A **FastAPI-based API** will be documented at:
🔗 **[https://farabouna-getaroudapispace.hf.space/docs](https://farabouna-getaroudapispace.hf.space/docs)**  

#### API Features:
- **🚀 /predict Endpoint:** Accepts JSON input and returns price predictions.
- **📖 Full Documentation:** Describes available endpoints, expected inputs, and outputs.

---

🛠 Technologies Used
Python 🐍
FastAPI ⚡ (Backend)
MLflow 📊 (Model Tracking)
Scikit-Learn 🎯 (Machine Learning)
Uvicorn 🚀 (API Server)
Streamlit 📈 (Dashboard)

---
🔗 Useful Links  
📦 GitHub Repo: [GitHub](https://github.com/Farabouna/Jedha_Data_Science_and_Engineering_Fullstack) <br>  
📊 MLflow Tracking: [MLflow UI](https://huggingface.co/spaces/Farabouna/GetAroundPricing) <br>  
📄 API Docs: [FastAPI UI](https://huggingface.co/spaces/Farabouna/GetAroudApiSpace) <br>  
📊 Live Dashboard: [Streamlit](https://huggingface.co/spaces/Farabouna/GetAround)
