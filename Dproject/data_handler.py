import pandas as pd
import io

class DataHandler:
    def export_to_excel(self, history, include_recommendations=True):
        df = pd.DataFrame([{
            **record['input_data'],
            'Risk Probability': record['risk_probability'],
            'Risk Level': record['risk_level'],
            'Confidence Score': record['confidence_score'],
            'BMI': record['bmi'],
            'Timestamp': record['timestamp']
        } for record in history])
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Assessments')
        output.seek(0)
        return output

    def export_to_csv(self, history):
        df = pd.DataFrame([{
            **record['input_data'],
            'Risk Probability': record['risk_probability'],
            'Risk Level': record['risk_level'],
            'Confidence Score': record['confidence_score'],
            'BMI': record['bmi'],
            'Timestamp': record['timestamp']
        } for record in history])
        return df.to_csv(index=False).encode('utf-8')
