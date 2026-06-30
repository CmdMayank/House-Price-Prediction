from preProsessing import *
from LinearRegression import *
from metrics import *

df = load_data("data/house_price.csv")
train, test = train_test_split(df)

X_train, Y_train = feature_train_split(train)
X_test , Y_test = feature_train_split(test)

X_train, mean, std = stardardize(X_train)
X_test = (X_test - mean) / std

model = LinearRegression(learning_rate=0.01, iterations=5000)
model.fit(X_train, Y_train)
pred = model.predict(X_test)

print(pred)

new_house = [[1700, 2, 2]]
new_house = (new_house)/std
price = model.predict(new_house)

print(price[0])