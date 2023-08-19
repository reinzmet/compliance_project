from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import torch
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = ''
app.secret_key = 'some_secret_key'

# Load pre-trained BERT model and tokenizer
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
bert_model = BertForSequenceClassification.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained(model_name)

def check_compliance_with_llm(log_entry, rule):
    """Check a log entry's compliance with a given rule using rule-based checks and the LLM."""
    
    # Rule-based check for DELETE
    if "DELETE" in log_entry:
        return True
    
    # Check using BERT
    prompt = f"Review the log entry: '{log_entry}'. If it violates the rule: '{rule}', state 'VIOLATION'. Otherwise, state 'NO VIOLATION'."
    inputs = bert_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = bert_model(**inputs)
    logits = outputs.logits
    predictions = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class = torch.argmax(predictions[0]).item()

    # If BERT's predicted class indicates a violation, it's considered a breach
    if predicted_class == 1:  # Assuming class 1 indicates a violation
        return True
    else:
        return False



def analyze_log_with_bert(file_path):
    detected_breaches = []
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            log_data = ' '.join(row.values.astype(str))
            rule = "No user should delete files."  # Example rule
            if check_compliance_with_llm(log_data, rule):
                detected_breaches.append(log_data)
    return detected_breaches

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the uploaded file for breaches using BERT
            breaches = analyze_log_with_bert(filepath)
            
            # Return the detected breaches to the user
            return render_template('view_breaches.html', breaches=breaches)
            
    return render_template('upload.html')

@app.route('/view_breaches')
def view_breaches():
    breaches = analyze_log_with_bert(os.path.join(app.config['UPLOAD_FOLDER'], 'sample_log.csv'))
    return render_template('view_breaches.html', breaches=breaches)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
