"""
======================================================================
AI Exposure Predictor
----------------------------------------------------------------------

Author      : Divya Nehete
Application : Streamlit Web App

Purpose
-------
This application predicts the AI Exposure Score of a job task using
the trained Random Forest model developed in this project.

Tech Stack
----------
• Streamlit
• Scikit-Learn
• Pandas
• NumPy
• Joblib

======================================================================
"""

# ==============================================================
# Import Libraries
# ==============================================================

import streamlit as st
import joblib
from pathlib import Path

from predictor import predict_ai_exposure
from utils import risk_level, automation_percentage


# ==============================================================
# Page Configuration
#
# Concept:
# Streamlit allows configuring the page before rendering.
# This gives the application a professional appearance.
# ==============================================================

st.set_page_config(
    page_title="AI Exposure Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==============================================================
# Load Task Types
#
# Concept:
# Load the fitted OneHotEncoder and retrieve the available
# task categories dynamically.
# This avoids hardcoding dropdown values.
# ==============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

encoder = joblib.load(
    BASE_DIR / "data" / "processed" / "tasktype_encoder.pkl"
)

task_types = list(encoder.categories_[0])


# ==============================================================
# Sidebar
#
# Concept:
# The sidebar provides additional project information
# without cluttering the main interface.
# ==============================================================

with st.sidebar:

    st.title(" AI Exposure")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        This application predicts the AI Exposure Score
        of individual work tasks using a Machine Learning model.

        The prediction is based on:

        • Task Description

        • Task Type
        """
    )

    st.markdown("---")

    st.subheader("Model Information")

    st.info(
        """
        ✔ Algorithm:
        Random Forest Regressor

        ✔ Text Features:
        TF-IDF Vectorizer

        ✔ Task Encoding:
        OneHotEncoder

        ✔ Numerical Features:
        StandardScaler
        """
    )

    st.markdown("---")

    st.subheader("Project Statistics")

    st.metric("Dataset Size", "15,810 Tasks")

    st.metric("Model R² Score", "0.625")

    st.metric("Features", "1004")


# ==============================================================
# Main Header
#
# Concept:
# A clean landing section improves user experience and
# makes the application resemble a professional SaaS tool.
# ==============================================================

st.title(" AI Exposure Predictor")

st.caption(
    "Estimate how susceptible a job task is to AI automation using Machine Learning."
)

st.markdown("---")


# ==============================================================
# Welcome Container
# ==============================================================

with st.container(border=True):

    st.subheader(" How It Works")

    st.write(
        """
        1. Enter a work task description.

        2. Select the corresponding task type.

        3. Click **Predict AI Exposure**.

        4. The trained Random Forest model estimates the AI Exposure Score.
        """
    )

st.markdown("")

# ==============================================================
# End of Part 1
#
# Part 2 will add:
# • Input Form
# • Example Tasks
# • Predict Button
# ==============================================================
# ==============================================================
# User Input Section
#
# Concept:
# Streamlit forms group multiple inputs together and execute
# only when the user clicks the submit button.
#
# Tech Stack:
# • st.form()
# • st.text_area()
# • st.selectbox()
# • st.form_submit_button()
# ==============================================================

st.markdown("## Task Details")

with st.container(border=True):

    st.write(
        "Enter the task information below and click **Predict AI Exposure**."
    )

    with st.form("prediction_form"):

        # ------------------------------------------------------
        # Task Description
        # ------------------------------------------------------

        task_description = st.text_area(
            label="Task Description",
            placeholder="Example: Analyze financial reports and prepare monthly summaries.",
            height=180
        )

        # ------------------------------------------------------
        # Task Type
        # ------------------------------------------------------

        task_type = st.selectbox(
            "Task Type",
            task_types
        )

        # ------------------------------------------------------
        # Submit Button
        # ------------------------------------------------------

        predict_button = st.form_submit_button(
            "Predict AI Exposure",
            use_container_width=True
        )


# ==============================================================
# Example Tasks
#
# Concept:
# Shows users sample inputs they can try without requiring
# additional coding or HTML.
# ==============================================================

with st.expander("View Sample Tasks"):

    st.markdown("""
**Example 1**

Task Description:
Analyze financial reports and prepare monthly summaries.

Task Type:
Data

---

**Example 2**

Task Description:
Review legal documents and identify compliance issues.

Task Type:
Core

---

**Example 3**

Task Description:
Respond to customer support emails and resolve common issues.

Task Type:
Communication

---

**Example 4**

Task Description:
Inspect manufactured products for quality assurance.

Task Type:
Physical
""")


# ==============================================================
# Input Validation
#
# Concept:
# Prevent prediction if mandatory information is missing.
# ==============================================================

if predict_button:

    if task_description.strip() == "":

        st.error("Please enter a task description before continuing.")

        st.stop()

    # ----------------------------------------------------------
    # Progress Indicator
    #
    # Simulates processing to improve user experience.
    # ----------------------------------------------------------

    with st.spinner("Analyzing task..."):

        score = predict_ai_exposure(
            task_description,
            task_type
        )

        risk = risk_level(score)

        percentage = automation_percentage(score)

# ==============================================================
# Prediction Results Dashboard
#
# Concept:
# Display prediction results in a clean dashboard using
# Streamlit metrics and containers.
#
# Tech Stack:
# • st.columns()
# • st.metric()
# • st.progress()
# • st.container()
# ==============================================================

    st.markdown("---")
    st.header("Prediction Results")

    # ----------------------------------------------------------
    # Result Cards
    # ----------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="AI Exposure Score",
            value=f"{score:.3f}"
        )

    with col2:
        st.metric(
            label="Automation Potential",
            value=f"{percentage}%"
        )

    with col3:
        st.metric(
            label="Risk Level",
            value=risk
        )

    # ----------------------------------------------------------
    # Progress Bar
    # ----------------------------------------------------------

    st.markdown("### Exposure Level")

    st.progress(min(score, 1.0))

    # ----------------------------------------------------------
    # Interpretation
    # ----------------------------------------------------------

    if score < 0.33:

        interpretation = """
The task has relatively low AI exposure.

AI may assist with repetitive subtasks, but human expertise
remains the primary requirement.
"""

    elif score < 0.66:

        interpretation = """
The task has moderate AI exposure.

Several parts of the task can benefit from AI assistance,
while important decisions still require human involvement.
"""

    else:

        interpretation = """
The task has high AI exposure.

AI tools can automate or significantly assist many parts
of this task, although human oversight may still be required.
"""

    with st.container(border=True):

        st.subheader("Interpretation")

        st.write(interpretation)

    # ----------------------------------------------------------
    # Recommendations
    # ----------------------------------------------------------

    with st.container(border=True):

        st.subheader("Recommendations")

        if score < 0.33:

            st.write(
                """
• Human expertise is essential.

• AI can be used for administrative support.

• Automation opportunities are limited.
"""
            )

        elif score < 0.66:

            st.write(
                """
• Consider AI for repetitive tasks.

• Human review is recommended before final decisions.

• Combine AI assistance with domain expertise.
"""
            )

        else:

            st.write(
                """
• AI can automate a significant portion of this workflow.

• Human oversight is recommended for quality assurance.

• Evaluate suitable AI tools before deployment.
"""
            )

    # ----------------------------------------------------------
    # Summary Table
    # ----------------------------------------------------------

    st.markdown("### Prediction Summary")

    summary = {
        "Task Description": task_description,
        "Task Type": task_type,
        "AI Exposure Score": round(score, 3),
        "Automation Potential (%)": percentage,
        "Risk Level": risk
    }

    st.json(summary)
# ==============================================================
# Model Information
#
# Concept:
# Streamlit expanders hide additional information until
# the user chooses to view it, keeping the interface clean.
#
# Tech Stack:
# • st.expander()
# ==============================================================

st.markdown("---")

with st.expander("About the Machine Learning Model"):

    st.write("""
### Model Used

This application predicts the AI Exposure Score using a **Random Forest Regressor**.

### Feature Engineering

The prediction is based on the following features:

• TF-IDF representation of the Task Description

• OneHot Encoded Task Type

• Scaled numerical feature (Incumbents Responding)

### Training Pipeline

1. Data Cleaning
2. Feature Engineering
3. Random Forest Training
4. Hyperparameter Tuning
5. Model Evaluation
6. Model Deployment using Streamlit

### Model Performance

• MAE : 0.471

• RMSE : 0.596

• R² Score : 0.625
""")


# ==============================================================
# Download Prediction
#
# Concept:
# Allows users to download prediction results as a CSV file.
#
# Tech Stack:
# • Pandas
# • Streamlit Download Button
# ==============================================================

if "score" in locals():

    import pandas as pd

    result_df = pd.DataFrame({

        "Task Description": [task_description],

        "Task Type": [task_type],

        "AI Exposure Score": [round(score, 3)],

        "Automation Potential (%)": [percentage],

        "Risk Level": [risk]

    })

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        label="Download Prediction Report",

        data=csv,

        file_name="prediction_report.csv",

        mime="text/csv",

        use_container_width=True

    )


# ==============================================================
# Frequently Asked Questions
#
# Concept:
# Helps users understand how predictions should be interpreted.
# ==============================================================

st.markdown("---")

with st.expander("Frequently Asked Questions"):

    st.markdown("""
### What does AI Exposure Score mean?

It estimates how much of a task could potentially be assisted or automated using AI technologies.

---

### Does a high score mean AI will replace humans?

No.

A higher score indicates greater potential for AI assistance or automation of certain parts of the task. Human expertise, oversight, and decision-making may still be essential.

---

### How is the prediction generated?

The application processes the task description using TF-IDF feature extraction, combines it with the selected task type and numerical features, and predicts the score using a trained Random Forest model.
""")


# ==============================================================
# Footer
#
# Concept:
# Provides project details in a professional manner.
# ==============================================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:

    st.caption("Developed By")

    st.write("Divya Nehete")

with col2:

    st.caption("Technology")

    st.write("Python | Scikit-Learn | Streamlit")

with col3:

    st.caption("Model")

    st.write("Random Forest Regressor")

st.markdown("---")

st.caption(
    "AI Exposure Predictor • Portfolio Project • 2026"
)    