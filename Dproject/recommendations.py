import numpy as np
from datetime import datetime

class RecommendationEngine:
    def __init__(self):
        """Initialize the recommendation engine."""
        self.recommendation_templates = {
            'weight_management': {
                'category': 'Weight Management',
                'templates': [
                    {
                        'condition': 'bmi_high',
                        'priority': 'High',
                        'action': 'Weight Reduction Program',
                        'details': 'Your BMI indicates obesity. Consider consulting a nutritionist for a structured weight loss plan. Aim for 1-2 pounds weight loss per week through caloric deficit and increased physical activity.'
                    },
                    {
                        'condition': 'bmi_overweight',
                        'priority': 'Medium',
                        'action': 'Healthy Weight Management',
                        'details': 'Your BMI is in the overweight range. Focus on portion control and regular exercise. Consider a balanced diet with reduced refined carbohydrates.'
                    }
                ]
            },
            'physical_activity': {
                'category': 'Physical Activity',
                'templates': [
                    {
                        'condition': 'low_activity',
                        'priority': 'High',
                        'action': 'Increase Physical Activity',
                        'details': 'Regular exercise is crucial for diabetes prevention. Start with 30 minutes of moderate activity 5 days per week. Consider walking, swimming, or cycling.'
                    },
                    {
                        'condition': 'moderate_activity',
                        'priority': 'Medium',
                        'action': 'Enhance Exercise Routine',
                        'details': 'Continue your current activity level and consider adding strength training 2-3 times per week for better glucose metabolism.'
                    }
                ]
            },
            'nutrition': {
                'category': 'Nutrition & Diet',
                'templates': [
                    {
                        'condition': 'high_glucose',
                        'priority': 'High',
                        'action': 'Blood Sugar Management',
                        'details': 'Your glucose levels are elevated. Focus on low glycemic index foods, reduce refined sugars, and consider smaller, more frequent meals.'
                    },
                    {
                        'condition': 'general_diet',
                        'priority': 'Medium',
                        'action': 'Diabetes Prevention Diet',
                        'details': 'Adopt a Mediterranean-style diet rich in vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods and sugary beverages.'
                    }
                ]
            },
            'medical_monitoring': {
                'category': 'Medical Monitoring',
                'templates': [
                    {
                        'condition': 'high_risk',
                        'priority': 'High',
                        'action': 'Regular Medical Checkups',
                        'details': 'Given your risk profile, schedule regular checkups with your healthcare provider. Consider HbA1c testing every 3-6 months.'
                    },
                    {
                        'condition': 'hypertension',
                        'priority': 'High',
                        'action': 'Blood Pressure Management',
                        'details': 'Monitor blood pressure regularly. Consider DASH diet principles and reduce sodium intake. Consult your doctor about blood pressure management.'
                    }
                ]
            },
            'lifestyle': {
                'category': 'Lifestyle Modifications',
                'templates': [
                    {
                        'condition': 'smoking',
                        'priority': 'High',
                        'action': 'Smoking Cessation',
                        'details': 'Smoking significantly increases diabetes risk. Consider smoking cessation programs, nicotine replacement therapy, or medications under medical supervision.'
                    },
                    {
                        'condition': 'stress_management',
                        'priority': 'Medium',
                        'action': 'Stress Management',
                        'details': 'Chronic stress can affect blood sugar levels. Practice stress reduction techniques like meditation, yoga, or deep breathing exercises.'
                    }
                ]
            },
            'prevention': {
                'category': 'Prevention Strategies',
                'templates': [
                    {
                        'condition': 'family_history',
                        'priority': 'Medium',
                        'action': 'Enhanced Screening',
                        'details': 'With family history of diabetes, maintain more frequent screenings and be extra vigilant about lifestyle factors.'
                    },
                    {
                        'condition': 'general_prevention',
                        'priority': 'Low',
                        'action': 'Preventive Measures',
                        'details': 'Continue healthy habits including regular exercise, balanced diet, adequate sleep (7-9 hours), and stress management.'
                    }
                ]
            }
        }
    
    def get_recommendations(self, prediction_record):
        """
        Generate personalized recommendations based on prediction results and input data.
        
        Args:
            prediction_record (dict): Complete prediction record with input data and results
            
        Returns:
            list: List of recommendation dictionaries
        """
        recommendations = []
        input_data = prediction_record['input_data']
        risk_probability = prediction_record['risk_probability']
        risk_level = prediction_record['risk_level']
        bmi = prediction_record['bmi']
        
        # Weight management recommendations
        if bmi >= 30:
            recommendations.append(self._get_recommendation('weight_management', 'bmi_high'))
        elif bmi >= 25:
            recommendations.append(self._get_recommendation('weight_management', 'bmi_overweight'))
        
        # Physical activity recommendations
        activity_level = input_data.get('physical_activity', 0)
        if activity_level == 0:  # Low activity
            recommendations.append(self._get_recommendation('physical_activity', 'low_activity'))
        elif activity_level == 1:  # Moderate activity
            recommendations.append(self._get_recommendation('physical_activity', 'moderate_activity'))
        
        # Nutrition recommendations
        glucose_level = input_data.get('glucose_level', 100)
        if glucose_level >= 126:  # Diabetic range
            recommendations.append(self._get_recommendation('nutrition', 'high_glucose'))
        elif glucose_level >= 100:  # Pre-diabetic range
            recommendations.append(self._get_recommendation('nutrition', 'high_glucose'))
        else:
            recommendations.append(self._get_recommendation('nutrition', 'general_diet'))
        
        # Medical monitoring recommendations
        if risk_level == "High Risk":
            recommendations.append(self._get_recommendation('medical_monitoring', 'high_risk'))
        
        if input_data.get('hypertension', 0) == 1:
            recommendations.append(self._get_recommendation('medical_monitoring', 'hypertension'))
        
        # Lifestyle recommendations
        smoking_status = input_data.get('smoking', 0)
        if smoking_status == 2:  # Current smoker
            recommendations.append(self._get_recommendation('lifestyle', 'smoking'))
        
        # Always add stress management for moderate to high risk
        if risk_probability >= 0.3:
            recommendations.append(self._get_recommendation('lifestyle', 'stress_management'))
        
        # Prevention strategies
        if input_data.get('family_history', 0) == 1:
            recommendations.append(self._get_recommendation('prevention', 'family_history'))
        
        # General prevention for all users
        recommendations.append(self._get_recommendation('prevention', 'general_prevention'))
        
        # Remove duplicates and sort by priority
        unique_recommendations = self._remove_duplicate_recommendations(recommendations)
        sorted_recommendations = self._sort_by_priority(unique_recommendations)
        
        return sorted_recommendations[:6]  # Return top 6 recommendations
    
    def _get_recommendation(self, category, condition):
        """Get a specific recommendation template."""
        category_templates = self.recommendation_templates.get(category, {}).get('templates', [])
        
        for template in category_templates:
            if template['condition'] == condition:
                return {
                    'category': self.recommendation_templates[category]['category'],
                    'priority': template['priority'],
                    'action': template['action'],
                    'details': template['details'],
                    'condition': condition
                }
        
        # Return a default recommendation if specific condition not found
        return {
            'category': 'General Health',
            'priority': 'Medium',
            'action': 'Consult Healthcare Provider',
            'details': 'Consider discussing your health status and risk factors with a healthcare professional for personalized advice.',
            'condition': 'general'
        }
    
    def _remove_duplicate_recommendations(self, recommendations):
        """Remove duplicate recommendations based on action."""
        seen_actions = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec['action'] not in seen_actions:
                seen_actions.add(rec['action'])
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def _sort_by_priority(self, recommendations):
        """Sort recommendations by priority level."""
        priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
        
        return sorted(
            recommendations,
            key=lambda x: priority_order.get(x['priority'], 3)
        )
    
    def get_lifestyle_score(self, input_data):
        """Calculate a lifestyle score based on input factors."""
        score = 100  # Start with perfect score
        
        # Physical activity (0-20 points deduction)
        activity_level = input_data.get('physical_activity', 0)
        if activity_level == 0:
            score -= 20
        elif activity_level == 1:
            score -= 10
        
        # Smoking (0-25 points deduction)
        smoking_status = input_data.get('smoking', 0)
        if smoking_status == 2:  # Current smoker
            score -= 25
        elif smoking_status == 1:  # Former smoker
            score -= 10
        
        # Alcohol consumption (0-10 points deduction)
        alcohol_level = input_data.get('alcohol_consumption', 0)
        if alcohol_level == 2:  # Heavy drinking
            score -= 10
        elif alcohol_level == 1:  # Moderate drinking
            score -= 5
        
        # BMI impact (calculated separately but affects lifestyle)
        # This would be added externally based on calculated BMI
        
        return max(0, score)  # Ensure score doesn't go below 0
    
    def get_priority_actions(self, prediction_record):
        """Get top priority actions based on risk factors."""
        recommendations = self.get_recommendations(prediction_record)
        high_priority = [rec for rec in recommendations if rec['priority'] == 'High']
        
        if not high_priority:
            # If no high priority items, return top 2 medium priority
            medium_priority = [rec for rec in recommendations if rec['priority'] == 'Medium']
            return medium_priority[:2]
        
        return high_priority
    
    def generate_action_plan(self, prediction_record):
        """Generate a structured action plan with timeline."""
        recommendations = self.get_recommendations(prediction_record)
        
        action_plan = {
            'immediate_actions': [],  # Within 1 week
            'short_term_goals': [],   # 1-3 months
            'long_term_goals': []     # 3+ months
        }
        
        for rec in recommendations:
            if rec['priority'] == 'High':
                if any(keyword in rec['action'].lower() for keyword in ['smoking', 'medical', 'checkup']):
                    action_plan['immediate_actions'].append(rec)
                else:
                    action_plan['short_term_goals'].append(rec)
            else:
                action_plan['long_term_goals'].append(rec)
        
        return action_plan
    
    def get_educational_content(self, risk_level):
        """Get educational content based on risk level."""
        education_content = {
            'Low Risk': {
                'title': 'Diabetes Prevention',
                'content': [
                    'Maintain your current healthy lifestyle',
                    'Regular exercise helps improve insulin sensitivity',
                    'A balanced diet prevents blood sugar spikes',
                    'Annual health screenings are recommended'
                ]
            },
            'Moderate Risk': {
                'title': 'Pre-diabetes Management',
                'content': [
                    'You may be in the pre-diabetic stage',
                    'Lifestyle changes can prevent progression to diabetes',
                    'Weight loss of 5-10% can significantly reduce risk',
                    'Consider diabetes prevention programs'
                ]
            },
            'High Risk': {
                'title': 'Immediate Risk Management',
                'content': [
                    'Urgent lifestyle modifications are needed',
                    'Medical consultation is strongly recommended',
                    'Regular glucose monitoring may be necessary',
                    'Consider joining a diabetes prevention program'
                ]
            }
        }
        
        return education_content.get(risk_level, education_content['Moderate Risk'])
