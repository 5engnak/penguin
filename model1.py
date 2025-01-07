import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
df = sns.load_dataset('penguins')
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

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.astype('category').cat.codes.values, dtype=torch.long)

# Define the PyTorch model
class PenguinModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(PenguinModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.fc(x)

# Train the model
input_size = X_train.shape[1]
output_size = len(y.unique())  # Number of species (classes)

model = PenguinModel(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), "penguin_pytorch_model.pth")

# Evaluate the model
model.eval()
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.astype('category').cat.codes.values, dtype=torch.long)
outputs = model(X_test_tensor)
_, predicted = torch.max(outputs, 1)
accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())
print(f'Accuracy on test set: {accuracy * 100:.2f}%')

# Save the scaler
joblib.dump(scaler, 'scaler1.pkl')
