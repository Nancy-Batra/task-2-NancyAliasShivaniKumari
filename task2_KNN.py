import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

df = pd.read_excel("Dataset_for_Data_Analytics.xlsx")
print(df.head())

# col
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

df.drop(["OrderID", "CustomerID", "ShippingAddress", "TrackingNumber", "Date"], axis=1, inplace=True)

for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop("OrderStatus", axis=1)
print(X.head())
y = df["OrderStatus"]

X = StandardScaler().fit_transform(X)

# data splitting
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, shuffle=True, random_state=42 )

# KNN
best_k = 1
best_accuracy = 0

for k in range(1, 21):
    temp_model = KNeighborsClassifier(n_neighbors=k, weights= "distance")
    temp_model.fit(X_train, y_train)
    temp_pred = temp_model.predict(X_test)
    score = accuracy_score(y_test, temp_pred)

    if score > best_accuracy:
        best_accuracy = score
        best_k = k

print("\n Best K:", best_k)

# Final KNN Model
model = KNeighborsClassifier(n_neighbors=best_k, weights="distance")
model.fit(X_train, y_train)

pred = model.predict(X_test)

# evaluation and confusion matrix
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="weighted")
print("\n Accuracy:", accuracy)
print("F1 Score:", f1)

print("Confusion Matrix:")
cm = confusion_matrix(y_test, pred)
labels = ["Cancelled", "Delivered", "Pending", "Returned", "Shipped"]
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
).plot(cmap="Blues")

plt.title("Order Status Classification")
plt.show()
