
import streamlit as st # app.py 생성
import pandas as pd
import joblib

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

url = "https://github.com/dongupak/DataML/raw/main/csv/life_expectancy.csv"

df = pd.read_csv(url)
df.columns = df.columns.str.strip()
df = df.dropna()

features = [
    'Adult mortality',
    'BMI',
    'GDP',
    'Alcohol'
]

target = 'Life expectancy'

linear_model = joblib.load("linear_model.pkl")
poly_model = joblib.load("poly_model.pkl")
ridge_model = joblib.load("ridge_model.pkl")

models = {
    "Linear": linear_model,
    "Poly": poly_model,
    "Ridge": ridge_model
}

st.title("Life Expectancy Prediction Dashboard")

st.sidebar.header("Input Features")

adult = st.sidebar.slider(
    "Adult mortality",
    float(df['Adult mortality'].min()),
    float(df['Adult mortality'].max()),
    float(df['Adult mortality'].mean())
)

bmi = st.sidebar.slider(
    "BMI",
    float(df['BMI'].min()),
    float(df['BMI'].max()),
    float(df['BMI'].mean())
)

gdp = st.sidebar.slider(
    "GDP",
    float(df['GDP'].min()),
    float(df['GDP'].max()),
    float(df['GDP'].mean())
)

alcohol = st.sidebar.slider(
    "Alcohol",
    float(df['Alcohol'].min()),
    float(df['Alcohol'].max()),
    float(df['Alcohol'].mean())
)

selected_model = st.selectbox(
    "Choose Model",
    ["Linear", "Poly", "Ridge"]
)

input_df = pd.DataFrame({
    'Adult mortality':[adult],
    'BMI':[bmi],
    'GDP':[gdp],
    'Alcohol':[alcohol]
})

prediction = models[selected_model].predict(input_df)[0]

st.metric(
    "Predicted Life Expectancy",
    f"{prediction:.2f} years"
)

X = df[features]
y = df[target]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train_full.sample(
    n=50,
    random_state=42
)

y_train = y_train_full.loc[X_train.index]

results = []

for name, model in models.items():

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    if name == "Linear":
        complexity = len(features)
    else:
        complexity = model.named_steps['poly'].fit_transform(X_train).shape[1]

    results.append([
        name,
        train_r2,
        test_r2,
        train_mse,
        test_mse,
        complexity
    ])

result_df = pd.DataFrame(
    results,
    columns=[
        'Model',
        'Train R2',
        'Test R2',
        'Train MSE',
        'Test MSE',
        'Complexity'
    ]
)

st.subheader("Model Performance Comparison")

st.dataframe(result_df)

fig, ax = plt.subplots()

ax.bar(
    result_df["Model"],
    result_df["Test R2"]
)

ax.set_ylabel("Test R2")

st.pyplot(fig)
