import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#TITLE
st.title("💼 Salary Predictor App")
st.write("Predict salary based on years of experience")

#LOAD DATA
df = pd.read_csv("employees.csv")

#MODEL
X = df[["experience"]]
y = df["salary"]

model = LinearRegression()
model.fit(X, y)

#USER INPUT
experience = st.number_input("Enter years of experience:", min_value=0.0, step=0.5)

#BUTTON
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"Estimated Salary: £{int(prediction[0])}")

#BONUS: CHART
st.subheader("📊 Salary vs Experience")

fig, ax = plt.subplots()
ax.scatter(df["experience"], df["salary"])
ax.plot(df["experience"], model.predict(X))
ax.set_xlabel("Experience")
ax.set_ylabel("Salary")

st.pyplot(fig)