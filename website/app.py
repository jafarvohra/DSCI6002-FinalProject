import base64
import io
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

#model code
data = pd.read_csv("heart_attack_prediction_dataset.csv")

data[["Systolic Blood Pressure","Diastolic Blood Pressure"]] = data["Blood Pressure"].str.split("/", expand=True).astype(int)
data[["Blood Pressure","Systolic Blood Pressure","Diastolic Blood Pressure"]].head()

data = data.drop(columns="Blood Pressure")

data['Sex'] = LabelEncoder().fit_transform(data['Sex']) # Male = 1, Female = 0
data['Diet'] = LabelEncoder().fit_transform(data['Diet']) # Average = 0, Healthy = 1, Unhealthy = 2

X = data.drop(columns = ['Patient ID', 'Country', 'Continent', 'Hemisphere','Heart Attack Risk'])
y = data['Heart Attack Risk']

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25 , random_state = 11)

tree_model = DecisionTreeClassifier()

# Define the parameter grid to search
param_grid = {
    'criterion': ['gini'],  # The function to measure the quality of a split
    'max_depth': [None, 50],  # Maximum depth of the tree
    'min_samples_split': [2],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1],  # Minimum number of samples required to be at a leaf node
    'min_impurity_decrease': [0.0, 0.2],  # Minimum impurity decrease for a split
    'max_leaf_nodes': [None, 10],  # Maximum number of leaf nodes in the tree
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(tree_model, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(train_X, train_y)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
best_pred = best_model.predict(test_X)


app = Flask(__name__)

@app.route('/')
def home():
    fpr, tpr, thresholds = roc_curve(test_y, best_pred)
    auc = roc_auc_score(test_y, best_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_memory = base64.b64encode(buffer.getvalue())
    img_area = image_memory.decode('utf-8')
    return render_template('index.html', img_area=img_area)

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

def predict_heart_attack_risk(model, input_data):
    input_data = np.array(input_data).reshape(1, -1)  
    prediction = model.predict(input_data)
    if prediction == 1:
        result = "High Risk of Heart Attack"
    else:
        result = "Low Risk of Heart Attack"

    return result

@app.route('/features', methods=['POST'])
def features():
    
    x1 = int(request.form.get('x1'))
    x2 = int(request.form.get('x2'))
    x3 = int(request.form.get('x3'))
    x4 = int(request.form.get('x4'))
    x5 = int(request.form.get('x5'))
    x6 = int(request.form.get('x6'))
    x7 = int(request.form.get('x7'))
    x8 = int(request.form.get('x8'))
    x9 = int(request.form.get('x9'))
    x10 = int(request.form.get('x10'))
    x11 = int(request.form.get('x11'))
    x12 = int(request.form.get('x12'))
    x13 = int(request.form.get('x13'))
    x14 = int(request.form.get('x14'))
    x15 = int(request.form.get('x15'))
    x16 = int(request.form.get('x16'))
    x17 = int(request.form.get('x17'))
    x18 = int(request.form.get('x18'))
    x19 = int(request.form.get('x19'))
    x20 = int(request.form.get('x20'))
    x21 = int(request.form.get('x21'))
    x22 = int(request.form.get('x22'))

    new_data_point = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22]
    new_data_point_df = pd.DataFrame([new_data_point], columns=data.drop(columns=['Patient ID', 'Country', 'Continent', 'Hemisphere', 'Heart Attack Risk']).columns)
    y = predict_heart_attack_risk(best_model, new_data_point_df)
   
    return render_template('monitor.html', y=y)

    



@app.route('/tips')
def tips():
    return render_template('tips.html')



if __name__ == '__main__':
    app.run(debug=True)
