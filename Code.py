import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("Crop_recommendation.csv")

data.head()

data.isnull().sum()

data.tail()

data.info()

data['label'].value_counts()

x = data.drop("label", axis=1)
y = data["label"]

x.info()

y.info()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1
)

x_train.info()

x_test.info()

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


ample = pd.DataFrame([[90, 40, 40, 25, 60, 6.5, 100]],
                      columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

prediction = model.predict(sample)
print("Recommended Crop:", prediction[0])


import pandas as pd

n = float(input("Nitrogen: "))
p = float(input("Phosphorus: "))
k = float(input("Potassium: "))
temp = float(input("Temperature: "))
humidity = float(input("Humidity: "))
ph = float(input("pH: "))
rainfall = float(input("Rainfall: "))

user_data = pd.DataFrame(
    [[n, p, k, temp, humidity, ph, rainfall]],
    columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
)

result = model.predict(user_data)

print("Best Crop:", result[0])
