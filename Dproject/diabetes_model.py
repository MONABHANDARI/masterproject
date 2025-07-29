import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

class DiabetesPredictor:
    def __init__(self):
        """Initialize the diabetes prediction model."""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'gender', 'bmi', 'family_history', 'hypertension',
            'heart_disease', 'glucose_level', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'cholesterol', 'physical_activity',
            'smoking', 'alcohol_consumption'
        ]
        self.is_trained = False
        self._train_model()
    
    def _generate_synthetic_training_data(self, n_samples=1000):
        """Generate synthetic training data for the model."""
        np.random.seed(42)
        
        # Generate features
        data = {
            'age': np.random.normal(45, 15, n_samples).clip(18, 90),
            'gender': np.random.choice([0, 1], n_samples),
            'bmi': np.random.normal(26, 5, n_samples).clip(15, 50),
            'family_history': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'hypertension': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
            'heart_disease': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'glucose_level': np.random.normal(110, 30, n_samples).clip(70, 300),
            'blood_pressure_systolic': np.random.normal(130, 20, n_samples).clip(90, 200),
            'blood_pressure_diastolic': np.random.normal(85, 15, n_samples).clip(60, 120),
            'cholesterol': np.random.normal(200, 40, n_samples).clip(150, 350),
            'physical_activity': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2]),
            'smoking': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
            'alcohol_consumption': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.5, 0.1])
        }
        
        # Generate target variable based on risk factors
        risk_scores = (
            (data['age'] - 18) / 72 * 0.2 +
            data['bmi'] / 50 * 0.15 +
            data['family_history'] * 0.15 +
            data['hypertension'] * 0.1 +
            data['heart_disease'] * 0.1 +
            (data['glucose_level'] - 70) / 230 * 0.2 +
            (data['blood_pressure_systolic'] - 90) / 110 * 0.05 +
            (data['cholesterol'] - 150) / 200 * 0.05 +
            (2 - data['physical_activity']) / 2 * 0.1 +
            data['smoking'] / 2 * 0.08 +
            data['alcohol_consumption'] / 2 * 0.02
        )
        
        # Add some noise and convert to binary
        noise = np.random.normal(0, 0.1, n_samples)
        risk_scores += noise
        
        # Convert to binary target (diabetes: 1, no diabetes: 0)
        target = (risk_scores > 0.4).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['diabetes'] = target
        
        return df
    
    def _train_model(self):
        """Train the diabetes prediction model."""
        try:
            # Generate training data
            training_data = self._generate_synthetic_training_data()
            
            # Prepare features and target
            X = training_data[self.feature_names]
            y = training_data['diabetes']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            print(f"Model trained successfully with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            self.is_trained = False
    
    def predict(self, input_data):
        """
        Make diabetes risk prediction.
        
        Args:
            input_data (dict): Dictionary containing health metrics
            
        Returns:
            dict: Prediction results including probability and risk level
        """
        if not self.is_trained:
            raise ValueError("Model is not trained. Cannot make predictions.")
        
        try:
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            input_df = input_df[self.feature_names]  # Ensure correct order
            
            # Scale input data
            input_scaled = self.scaler.transform(input_df)
            
            # Make prediction
            prediction_proba = self.model.predict_proba(input_scaled)[0]
            prediction_binary = self.model.predict(input_scaled)[0]
            
            # Get prediction probability for positive class (diabetes)
            diabetes_probability = prediction_proba[1]
            
            # Determine risk level
            if diabetes_probability < 0.3:
                risk_level = "Low Risk"
            elif diabetes_probability < 0.6:
                risk_level = "Moderate Risk"
            else:
                risk_level = "High Risk"
            
            # Calculate confidence score (based on prediction certainty)
            confidence = max(prediction_proba)
            
            # Get feature importance for this prediction
            feature_importance = self._get_feature_importance(input_data)
            
            return {
                'prediction': int(prediction_binary),
                'probability': float(diabetes_probability),
                'risk_level': risk_level,
                'confidence': float(confidence),
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def _get_feature_importance(self, input_data):
        """Get feature importance for the current prediction."""
        if not self.is_trained:
            return {}
        
        feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        sorted_importance = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return dict(sorted_importance[:5])  # Return top 5 important features
    
    def get_model_info(self):
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "Model not trained"}
        
        return {
            "status": "Model trained",
            "model_type": "Random Forest Classifier",
            "n_estimators": self.model.n_estimators,
            "features": self.feature_names,
            "feature_count": len(self.feature_names)
        }
    
    def save_model(self, filepath):
        """Save the trained model to file."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a trained model from file."""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
        except Exception as e:
            raise ValueError(f"Error loading model: {str(e)}")
