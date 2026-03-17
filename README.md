# Cuisine Classification App

This project predicts the **type of cuisine** based on restaurant details using a machine learning pipeline built with **XGBoost** and **TF-IDF vectorization**.  
The model achieves an accuracy of approximately **97.7%** on the test dataset.

---

## Features

- Predicts cuisine type based on restaurant name, dishes liked, and average cost.  
- Uses a trained **XGBoost Classifier** for high accuracy and robustness.  
- Interactive user interface built with **Streamlit**.  
- Supports local and cloud deployment using **Streamlit Cloud**.  

---

## Technology Stack

- **Programming Language:** Python  
- **Libraries and Frameworks:**  
  - scikit-learn  
  - XGBoost  
  - Streamlit  
  - pandas  
  - numpy  

---

## Project Structure

| File | Description |
|------|--------------|
| `app.py` | Main Streamlit web application. |
| `xgboost_cuisine_model.pkl` | Trained XGBoost model for cuisine classification. |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer for text feature extraction. |
| `label_encoder.pkl` | Label encoder for cuisine labels. |
| `requirements.txt` | List of dependencies required to run the project. |
| `cuisine_classification.ipynb` | Jupyter notebook used for model training and experimentation. |

---





## Setup and Installation
### 1. Clone the Repository
  ```bash
  git clone https://github.com/Nidaafreen0786/Cuisine_Classification.git
  cd Cuisine_Classification  
  ```

### 2. Create and Activate a Virtual Environment
  python -m venv .venv
  .venv\Scripts\activate    # On Windows
  (# source .venv/bin/activate  # On macOS/Linux)

### 3. Install Dependencies
  pip install -r requirements.txt

### 4. Run the Application
  streamlit run app.py

---

## Model Performance
      Metric	      Score
      Accuracy	      97.7%
      Model        	  XGBoost
      Vectorization   TF-IDF

---

## Deployment
This application can be deployed for free on Streamlit Cloud.
To deploy:
- Push this repository to GitHub.
- Go to Streamlit Cloud Deployment.
- Connect your GitHub repository.
- Set the main file to app.py.
- Add required secrets or environment variables if necessary.

---

## Author

**Nida Afreen**  
Machine Learning Enthusiast  
GitHub: [Nidaafreen0786](https://github.com/Nidaafreen0786)


---


