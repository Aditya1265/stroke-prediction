import streamlit as st
from prediction import predict

work_types = {
    "Private": "Private",
    "Self-employed": "Self-employed",
    "Children": "Children",
    "Government job": "Govt_job",
    "Never worked": "Never_worked",
}

smoking_statuses = {
    "Formerly smoked": "formerly smoked",
    "Never smoked": "never smoked",
    "Smokes": "smokes",
    "Unknown": "Unknown",
}


def main():
    st.title("Stroke Prediction")
    st.write(
        "This is a Stroke Prediction Model developed by Aditya Pandey and Dheeraj. Input relevant health and demographic details below to predict the likelihood of having a stroke. "
    )
    st.info(
        "This model is not a substitute for professional medical advice. Please consult a doctor for accurate diagnosis and treatment."
    )

    # input data from user ---------------------------------------------------

    gender = st.radio("gender", ["male", "female"], horizontal=True)
    age = st.slider("Age", 20, 80, 30)
    hypertension = st.radio(
        "Do you have hyper tension?", ["yes", "no"], horizontal=True
    )
    heart_disease = st.radio(
        "Do you have heart diseases?", ["yes", "no"], horizontal=True
    )
    ever_married = st.radio("Ever married?", ["yes", "no"], horizontal=True)
    work_type = st.selectbox(
        "Work type?",
        ["Private", "Self-employed", "Children", "Government job", "Never worked"],
    )
    residence_type = st.radio("Residence type?", ["Urban", "Rural"], horizontal=True)
    avg_glucose_level = st.number_input("Average Glucose Level", 60, 300, 100)

    c1, c2 = st.columns(2)
    with c1:
        height = st.number_input("Height", 100, 200, 150)
    with c2:
        weight = st.number_input("Weight", 30, 200, 60)

    bmi = weight / (height / 100) ** 2
    st.text(f"Your BMI is {bmi}")

    smoking_status = st.selectbox(
        "Smoking status?", ["Formerly smoked", "Never smoked", "Smokes", "Unknown"]
    )

    # prediction -------------------------------------------------------------

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            gender = 1 if gender == "male" else 0
            hypertension = 1 if hypertension == "yes" else 0
            heart_disease = 1 if heart_disease == "yes" else 0
            ever_married = 1 if ever_married == "yes" else 0
            work_type = work_types[work_type]
            smoking_status = smoking_statuses[smoking_status]

            result = predict(
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
            )

        # DEBUGGING

        # st.write(
        #     [
        #         gender,
        #         age,
        #         hypertension,
        #         heart_disease,
        #         ever_married,
        #         work_type,
        #         residence_type,
        #         avg_glucose_level,
        #         bmi,
        #         smoking_status,
        #     ]
        # )

        # st.json(result)

        if result["risk_level"] == "High":
            st.error(
                f"Your probability of having a stroke is {result['stroke_probability']}%"
            )
            st.error("Please consult a doctor immediately.")
        elif result["risk_level"] == "Moderate":
            st.warning(
                f"Your probability of having a stroke is {result['stroke_probability']}%"
            )
            st.warning("Please take care of your health.")
        else:
            st.success(
                f"You are safe from stroke. Your probability is {result['stroke_probability']}%"
            )
            st.success("Keep up the good work!")


if __name__ == "__main__":
    main()
