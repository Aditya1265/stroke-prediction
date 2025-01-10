import joblib
import pandas as pd

input_col = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]
numeric_col = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "avg_glucose_level",
    "bmi",
]
categorical_col = ["work_type", "Residence_type", "smoking_status"]


def predict(
    gender,
    age,
    hypertension,
    heart_disease,
    ever_married,
    work_type,
    residence_type,
    avg_glucose_level,
    bmi,
    smoking_status,
):
    # stroke_prediction_model.pkl
    best_rf = joblib.load("stroke_prediction_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")
    encoder = joblib.load("encoder.pkl")

    test_df = pd.DataFrame(
        {
            "gender": [gender],
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "ever_married": [ever_married],
            "work_type": [work_type],
            "Residence_type": [residence_type],
            "avg_glucose_level": [avg_glucose_level],
            "bmi": [bmi],
            "smoking_status": [smoking_status],
        }
    )

    test_df[numeric_col] = imputer.transform(test_df[numeric_col])
    test_df[numeric_col] = scaler.transform(test_df[numeric_col])

    test_df[categorical_col] = test_df[categorical_col].fillna("Unknown")
    encoded_features = encoder.transform(test_df[categorical_col])
    encoded_df = pd.DataFrame(
        encoded_features, columns=encoder.get_feature_names_out(categorical_col)
    )

    # numeric + encoded
    test_input = pd.concat([test_df[numeric_col], encoded_df], axis=1)

    # align cols with the trained model
    x_test = test_input.reindex(columns=best_rf.feature_names_in_, fill_value=0)

    stroke_prediction = bool(best_rf.predict(x_test)[0])
    stroke_probability = float(best_rf.predict_proba(x_test)[:, 1][0])
    risk_level = (
        "High"
        if stroke_probability > 0.7
        else "Moderate" if stroke_probability > 0.4 else "Low"
    )

    return {
        "stroke_probability": stroke_probability,
        "stroke_prediction": stroke_prediction,
        "risk_level": risk_level,
    }


result = predict(
    gender=0,
    age=65,
    hypertension=1,
    heart_disease=1,
    ever_married=1,
    work_type="Private",
    residence_type="Urban",
    avg_glucose_level=208.5,
    bmi=32.5,
    smoking_status="formerly smoked",
)

print(f"Stroke Probability: {result['stroke_probability']}")
print(f"Prediction: {result['stroke_prediction']}")
print(f"Risk Level: {result['risk_level']}")
