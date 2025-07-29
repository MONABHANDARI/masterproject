# train_model.py
import joblib
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load sample diabetes data
data = load_diabetes()
X = data.data
y = (data.target > 100).astype(int)  # Example: classify if target is > 100

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, 'diabetes_model.pkl')
