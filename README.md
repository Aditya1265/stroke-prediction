# Stroke Prediction Model

This is a Stroke Prediction Model. The app allows users to input relevant health and demographic details to predict the likelihood of having a stroke. This proof-of-concept application is designed for educational purposes and should not be used for medical advice. 

[Run the model here](https://stroke-prediction-model.streamlit.app/)

## Features
- User-Friendly Interface: Simple and interactive input forms for all relevant details.
- BMI Calculation: Automatically calculates BMI based on the user's height and weight.
- Risk Assessment: Provides stroke risk prediction categorized into:
  - High Risk
  - Moderate Risk
  - Low Risk
- Health Suggestions: Displays warnings and suggestions based on the prediction results.

## How It Works
1. Users provide the following information:
    - Gender
    - Age
    - Health conditions (hypertension, heart disease)
    - Marital status
    - Work type
    - Residence type (Urban/Rural)
    - Average glucose level
    - Height and weight (to calculate BMI)
    - Smoking status
2. The app processes the input and predicts the likelihood of a stroke using `predict` function (in `prediction.py` module).
3. Results are displayed with actionable recommendations:
    - High Risk: Urgent consultation advised.
    - Moderate Risk: General health improvement suggestions.
    - Low Risk: Encouragement to maintain good health.

## Running Locally
Clone the repository:

```
git clone https://github.com/your-username/stroke-prediction-app.git
cd stroke-prediction-app
```

Install the required Python packages:
```
pip install -r requirements.txt
Run the application:
```

```
streamlit run app.py
```

The app will open up in a new browser window.

## Disclaimer
This application is for educational purposes only and is not a substitute for professional medical advice. Always consult a healthcare provider for any medical concerns.
