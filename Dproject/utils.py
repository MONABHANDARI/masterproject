import re
import math
from datetime import datetime
import streamlit as st

def validate_inputs(input_data):
    """
    Validate health assessment input data.
    
    Args:
        input_data (dict): Dictionary containing health metrics
        
    Returns:
        list: List of validation error messages
    """
    errors = []
    
    # Age validation
    age = input_data.get('age', 0)
    if not isinstance(age, (int, float)) or age < 18 or age > 120:
        errors.append("Age must be between 18 and 120 years")
    
    # BMI validation (calculated from height and weight)
    bmi = input_data.get('bmi', 0)
    if not isinstance(bmi, (int, float)) or bmi < 10 or bmi > 70:
        errors.append("BMI appears to be outside normal range (10-70)")
    
    # Glucose level validation
    glucose = input_data.get('glucose_level', 0)
    if not isinstance(glucose, (int, float)) or glucose < 50 or glucose > 400:
        errors.append("Glucose level must be between 50 and 400 mg/dL")
    
    # Blood pressure validation
    systolic = input_data.get('blood_pressure_systolic', 0)
    diastolic = input_data.get('blood_pressure_diastolic', 0)
    
    if not isinstance(systolic, (int, float)) or systolic < 70 or systolic > 250:
        errors.append("Systolic blood pressure must be between 70 and 250 mmHg")
    
    if not isinstance(diastolic, (int, float)) or diastolic < 40 or diastolic > 150:
        errors.append("Diastolic blood pressure must be between 40 and 150 mmHg")
    
    if isinstance(systolic, (int, float)) and isinstance(diastolic, (int, float)):
        if diastolic >= systolic:
            errors.append("Diastolic blood pressure should be lower than systolic")
    
    # Cholesterol validation
    cholesterol = input_data.get('cholesterol', 0)
    if not isinstance(cholesterol, (int, float)) or cholesterol < 100 or cholesterol > 500:
        errors.append("Cholesterol level must be between 100 and 500 mg/dL")
    
    # Binary field validation
    binary_fields = ['gender', 'family_history', 'hypertension', 'heart_disease']
    for field in binary_fields:
        value = input_data.get(field)
        if value not in [0, 1]:
            errors.append(f"{field.replace('_', ' ').title()} must be specified")
    
    # Categorical field validation
    categorical_fields = {
        'physical_activity': [0, 1, 2],
        'smoking': [0, 1, 2],
        'alcohol_consumption': [0, 1, 2]
    }
    
    for field, valid_values in categorical_fields.items():
        value = input_data.get(field)
        if value not in valid_values:
            errors.append(f"{field.replace('_', ' ').title()} must be specified")
    
    return errors

def calculate_bmi(weight, height):
    """
    Calculate Body Mass Index (BMI).
    
    Args:
        weight (float): Weight in kilograms
        height (float): Height in centimeters
        
    Returns:
        float: BMI value rounded to 1 decimal place
    """
    if height <= 0 or weight <= 0:
        return 0
    
    height_m = height / 100  # Convert cm to meters
    bmi = weight / (height_m ** 2)
    return round(bmi, 1)

def get_bmi_category(bmi):
    """
    Get BMI category based on standard classifications.
    
    Args:
        bmi (float): BMI value
        
    Returns:
        str: BMI category
    """
    if bmi < 18.5:
        return "Underweight"
    elif 18.5 <= bmi < 25:
        return "Normal weight"
    elif 25 <= bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_risk_level_color(risk_level):
    """
    Get color code for risk level visualization.
    
    Args:
        risk_level (str): Risk level (Low Risk, Moderate Risk, High Risk)
        
    Returns:
        str: Color code
    """
    color_map = {
        "Low Risk": "#28a745",      # Green
        "Moderate Risk": "#ffc107", # Yellow/Orange
        "High Risk": "#dc3545"      # Red
    }
    return color_map.get(risk_level, "#6c757d")  # Default gray

def format_percentage(value):
    """
    Format a decimal value as a percentage.
    
    Args:
        value (float): Decimal value (0-1)
        
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.1f}%"

def validate_email(email):
    """
    Validate email address format.
    
    Args:
        email (str): Email address
        
    Returns:
        bool: True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def calculate_age_from_birthdate(birthdate):
    """
    Calculate age from birth date.
    
    Args:
        birthdate (datetime): Birth date
        
    Returns:
        int: Age in years
    """
    today = datetime.now()
    age = today.year - birthdate.year
    
    # Adjust if birthday hasn't occurred this year
    if today.month < birthdate.month or (today.month == birthdate.month and today.day < birthdate.day):
        age -= 1
    
    return age

def get_blood_pressure_category(systolic, diastolic):
    """
    Categorize blood pressure based on AHA guidelines.
    
    Args:
        systolic (int): Systolic blood pressure
        diastolic (int): Diastolic blood pressure
        
    Returns:
        str: Blood pressure category
    """
    if systolic < 120 and diastolic < 80:
        return "Normal"
    elif systolic < 130 and diastolic < 80:
        return "Elevated"
    elif (120 <= systolic <= 129) or (80 <= diastolic <= 89):
        return "High Blood Pressure Stage 1"
    elif systolic >= 130 or diastolic >= 90:
        return "High Blood Pressure Stage 2"
    elif systolic >= 180 or diastolic >= 120:
        return "Hypertensive Crisis"
    else:
        return "Unknown"

def get_glucose_category(glucose_level):
    """
    Categorize glucose level based on ADA guidelines.
    
    Args:
        glucose_level (float): Fasting glucose level in mg/dL
        
    Returns:
        str: Glucose category
    """
    if glucose_level < 100:
        return "Normal"
    elif 100 <= glucose_level <= 125:
        return "Pre-diabetes"
    else:
        return "Diabetes"

def calculate_diabetes_risk_score(input_data):
    """
    Calculate a simple diabetes risk score based on common risk factors.
    
    Args:
        input_data (dict): Health assessment data
        
    Returns:
        int: Risk score (0-100)
    """
    score = 0
    
    # Age factor (0-20 points)
    age = input_data.get('age', 0)
    if age >= 65:
        score += 20
    elif age >= 45:
        score += 15
    elif age >= 35:
        score += 10
    elif age >= 25:
        score += 5
    
    # BMI factor (0-25 points)
    bmi = input_data.get('bmi', 0)
    if bmi >= 35:
        score += 25
    elif bmi >= 30:
        score += 20
    elif bmi >= 25:
        score += 15
    elif bmi >= 23:
        score += 10
    
    # Family history (0-15 points)
    if input_data.get('family_history', 0) == 1:
        score += 15
    
    # Hypertension (0-10 points)
    if input_data.get('hypertension', 0) == 1:
        score += 10
    
    # Physical activity (0-10 points)
    activity = input_data.get('physical_activity', 2)
    if activity == 0:  # Low activity
        score += 10
    elif activity == 1:  # Moderate activity
        score += 5
    
    # Glucose level (0-20 points)
    glucose = input_data.get('glucose_level', 100)
    if glucose >= 126:
        score += 20
    elif glucose >= 100:
        score += 15
    elif glucose >= 90:
        score += 5
    
    return min(score, 100)  # Cap at 100

def format_health_metrics(input_data):
    """
    Format health metrics for display.
    
    Args:
        input_data (dict): Health assessment data
        
    Returns:
        dict: Formatted health metrics
    """
    formatted = {}
    
    # Basic info
    formatted['age'] = f"{input_data.get('age', 0)} years"
    formatted['gender'] = "Male" if input_data.get('gender', 0) == 1 else "Female"
    formatted['bmi'] = f"{input_data.get('bmi', 0):.1f}"
    
    # Health metrics
    formatted['glucose'] = f"{input_data.get('glucose_level', 0)} mg/dL"
    formatted['blood_pressure'] = f"{input_data.get('blood_pressure_systolic', 0)}/{input_data.get('blood_pressure_diastolic', 0)} mmHg"
    formatted['cholesterol'] = f"{input_data.get('cholesterol', 0)} mg/dL"
    
    # Lifestyle factors
    activity_map = {0: "Low", 1: "Moderate", 2: "High"}
    smoking_map = {0: "Never", 1: "Former", 2: "Current"}
    alcohol_map = {0: "None", 1: "Moderate", 2: "Heavy"}
    
    formatted['physical_activity'] = activity_map.get(input_data.get('physical_activity', 0), "Unknown")
    formatted['smoking'] = smoking_map.get(input_data.get('smoking', 0), "Unknown")
    formatted['alcohol'] = alcohol_map.get(input_data.get('alcohol_consumption', 0), "Unknown")
    
    # Medical history
    formatted['family_history'] = "Yes" if input_data.get('family_history', 0) == 1 else "No"
    formatted['hypertension'] = "Yes" if input_data.get('hypertension', 0) == 1 else "No"
    formatted['heart_disease'] = "Yes" if input_data.get('heart_disease', 0) == 1 else "No"
    
    return formatted

def display_success_message(message, icon="✅"):
    """Display a success message with custom styling."""
    st.success(f"{icon} {message}")

def display_warning_message(message, icon="⚠️"):
    """Display a warning message with custom styling."""
    st.warning(f"{icon} {message}")

def display_error_message(message, icon="❌"):
    """Display an error message with custom styling."""
    st.error(f"{icon} {message}")

def display_info_message(message, icon="ℹ️"):
    """Display an info message with custom styling."""
    st.info(f"{icon} {message}")

def create_download_link(data, filename, mime_type):
    """
    Create a download link for data.
    
    Args:
        data: Data to download
        filename (str): Name of the file
        mime_type (str): MIME type of the file
        
    Returns:
        str: Download link HTML
    """
    import base64
    
    if isinstance(data, str):
        data = data.encode()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href
