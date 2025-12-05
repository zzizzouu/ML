## using streamlit 
# ========================================
#  Used Car Price Prediction (Streamlit)
# ========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
#  Streamlit UI Configuration
# -------------------------------
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title(" Used Car Price Prediction App")
st.markdown("### Predict car resale prices using Machine Learning (Random Forest)")

# -------------------------------
#  Upload CSV
# -------------------------------
uploaded_file = st.file_uploader("Upload Cleaned Used Cars CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(" Data loaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------
    #  Define Features & Target
    # -------------------------------
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify column types
    numeric_features = ['Year', 'Kilometers_Driven', 'Mileage', 'Engine', 'Power']
    categorical_features = ['Location', 'Fuel_Type', 'Transmission']

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Build model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -------------------------------
    #  Model Evaluation
    # -------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("###  Model Evaluation Metrics")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("Mean Absolute Error (Lakh ₹)", f"{mae:.3f}")

    # -------------------------------
    #  Visualization
    # -------------------------------
    st.write("###  Actual vs Predicted Prices")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax1)
    ax1.set_xlabel("Actual Price (Lakh)")
    ax1.set_ylabel("Predicted Price (Lakh)")
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    st.pyplot(fig1)

    # Residuals
    residuals = y_test - y_pred
    st.write("###  Residual Distribution")
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, color='purple', ax=ax2)
    ax2.set_title("Residuals (Actual - Predicted)")
    st.pyplot(fig2)

    # Feature Importance
    st.write("###  Feature Importance")
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_features = ohe.get_feature_names_out(categorical_features)
    all_features = numeric_features + list(encoded_cat_features)
    importances = model.named_steps['regressor'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': all_features, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis', ax=ax3)
    ax3.set_title("Top 15 Important Features Affecting Price")
    st.pyplot(fig3)

    # -------------------------------
    #  Prediction Form
    # -------------------------------
    st.write("###  Predict Price for a New Car")

    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox("Location", df['Location'].unique())
        fuel = st.selectbox("Fuel Type", df['Fuel_Type'].unique())
        transmission = st.selectbox("Transmission", df['Transmission'].unique())

    with col2:
        year = st.number_input("Year", min_value=2000, max_value=2025, value=2015)
        km = st.number_input("Kilometers Driven", min_value=0, value=50000)
        mileage = st.number_input("Mileage (km/l)", min_value=5.0, value=18.0)

    with col3:
        engine = st.number_input("Engine (CC)", min_value=800, value=1200)
        power = st.number_input("Power (bhp)", min_value=30.0, value=80.0)

    if st.button(" Predict Price"):
        new_data = pd.DataFrame({
            'Year': [year],
            'Kilometers_Driven': [km],
            'Mileage': [mileage],
            'Engine': [engine],
            'Power': [power],
            'Location': [location],
            'Fuel_Type': [fuel],
            'Transmission': [transmission]
        })

        predicted_price = model.predict(new_data)[0]
        st.success(f" **Predicted Price: ₹ {predicted_price:.2f} Lakh**")

else:
    st.info(" Please upload a cleaned CSV file to begin.")
