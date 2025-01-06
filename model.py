import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and preprocess the dataset
df = sns.load_dataset('penguins')

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical features
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])

# Features and target
features = ["island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
target = "species"
X = pd.get_dummies(df[features], columns=["island"], drop_first=True)
y = df[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Train model
model = LogisticRegression(multi_class="multinomial", max_iter=200)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, 'penguin_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler have been saved as 'penguin_model.pkl' and 'scaler.pkl'.")
