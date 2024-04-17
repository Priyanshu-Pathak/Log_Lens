import numpy as np
from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.platypus import Paragraph
from io import BytesIO
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
matplotlib.use('Agg')
from flask import send_file
from io import BytesIO
from flask import Flask, render_template, request, send_file
from io import BytesIO
import pdfkit
from flask import Flask, render_template, request, send_file
from io import BytesIO
from flask import render_template, send_file, current_app
from io import BytesIO
from xhtml2pdf import pisa

from flask import send_from_directory
from mlxtend.frequent_patterns import apriori, association_rules


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join('uploads', filename)
            file.save(file_path)
            return redirect(url_for('results', filename=filename))
    return redirect(url_for('index'))

def bar_chart(df):
    df['RequestType'] = df['Request'].apply(lambda x: x.split()[0])
    request_counts = df['RequestType'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    request_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title('Number of Requests by Type')
    ax.set_xlabel('Request Type')
    ax.set_ylabel('Number of Requests')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('static/bar_chart.png')


def heatmap(df):
    df['ResponseTime'] = pd.to_numeric(df['ResponseTime'], errors='coerce')
    df = df.dropna(subset=['ResponseTime'])
    df['ResponseTimeBin'] = pd.cut(df['ResponseTime'], bins=5, labels=['Very Fast', 'Fast', 'Moderate', 'Slow', 'Very Slow'])
    df['RequestType'] = df['Request'].apply(lambda x: x.split()[0])
    df = df[['RequestType', 'ResponseTimeBin']]
    one_hot_encoded = pd.get_dummies(df)
    frequent_itemsets = apriori(one_hot_encoded, min_support=0.05, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules['Antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['Consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    plt.subplot(2, 1, 1)
    numeric_columns = ['support', 'confidence', 'lift']
    sns.heatmap(rules[numeric_columns], annot=True, fmt=".2f")
    plt.title('Association Rules')

    # Print rules in text format
    plt.subplot(2, 1, 2)
    plt.text(0, 1, rules[['Antecedents', 'Consequents', 'support', 'confidence', 'lift']].to_string(), va='top')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('static/heatmap.png')



def kmeans_clusters(df):
    X = df[['StatusCode', 'ResponseTime']]
    insights = []
    plt.figure(figsize=(12,10 ))
    for request_type in df['RequestType'].unique():
        X_subset = X[df['RequestType'] == request_type]
        kmeans = KMeans(n_clusters=3, random_state=42)  
        kmeans.fit(X_subset)
        plt.scatter(X_subset['StatusCode'], X_subset['ResponseTime'], c=kmeans.labels_, label=request_type)
        num_clusters = len(set(kmeans.labels_))
        avg_response_time = X_subset['ResponseTime'].mean()
        insights.append(f"Request Type: {request_type}, Number of Clusters: {num_clusters}, Avg. Response Time: {avg_response_time:.2f}")
    # Add labels and legend
    plt.title('Clusters for Different Request Types')
    plt.xlabel('StatusCode')
    plt.ylabel('ResponseTime')
    plt.legend()
    for i, text in enumerate(insights):
        plt.text(0, -0.07 - i * 0.02, text, transform=plt.gca().transAxes, fontsize=10)
    plt.savefig('static/kmeans_clusters.png')
    plt.close()


def decision_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True, cmap="Blues")
    plt.title('Classification Report')
    plt.savefig('static/decision_tree_classification_report.png') 
    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/decision_tree_confusion_matrix.png')  
    plt.close()

    return accuracy, classification_rep, conf_matrix


def knn(X_train, X_test, y_train, y_test):
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train, y_train)
    y_pred = knn_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plot classification report
    plt.figure(figsize=(8, 6))
    sns.heatmap(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).iloc[:-1, :].T, annot=True, cmap="Blues")
    plt.title('Classification Report')
    plt.savefig('static/knn_classification_report.png')  # Update the file path here
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('static/knn_confusion_matrix.png')  # Update the file path here
    plt.close()

    return accuracy, classification_rep, conf_matrix


def isolation_forest(df):
    df['ResponseTime'] = pd.to_numeric(df['ResponseTime'], errors='coerce')
    df = df.dropna(subset=['ResponseTime'])
    X = df[['StatusCode', 'ResponseTime']]
    clf = IsolationForest(contamination=0.1, random_state=42)  
    clf.fit(X)
    y_pred = clf.predict(X)
    df['Anomaly'] = y_pred
    df.to_csv('static/anomaly_predictions.csv', index=False)

    normal_points = df[df['Anomaly'] == 1]
    anomaly_points = df[df['Anomaly'] == -1]

    # Plot the data points
    plt.figure(figsize=(10, 8))
    plt.scatter(normal_points['StatusCode'], normal_points['ResponseTime'], color='blue', label='Normal')
    plt.scatter(anomaly_points['StatusCode'], anomaly_points['ResponseTime'], color='red', label='Anomaly')
    plt.title('Anomalies Detected by Isolation Forest')
    plt.xlabel('StatusCode')
    plt.ylabel('ResponseTime')
    plt.legend()

    plt.text(0.05, -0.1, f"Number of anomalies: {len(anomaly_points)}", transform=plt.gca().transAxes, fontsize=10)
    plt.text(0.05, -0.12, f"Anomaly rate: {len(anomaly_points) / len(df) * 100:.2f}%", transform=plt.gca().transAxes, fontsize=10)

    # Save the plot as PNG file
    plt.savefig('static/isolation_forest.png')

    # Close the plot to release resources
    plt.close()


@app.route('/results')
def results():
    filename = request.args.get('filename')
    if filename:
        file_path = os.path.join('uploads', filename)
        df = pd.read_csv(file_path, header=None, names=["IP", "Request", "StatusCode", "Column8", "UserAgent", "ResponseTime"])
        df['ResponseTime'] = pd.to_numeric(df['ResponseTime'], errors='coerce')
        df = df.dropna(subset=['ResponseTime'])
        df['RequestType'] = df['Request'].apply(lambda x: x.split()[0])
        bar_chart(df)
        heatmap(df)
        kmeans_clusters(df)
        X = df[['StatusCode', 'ResponseTime']]
        y = df['Request'].apply(lambda x: x.split()[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy, classification_rep, conf_matrix = decision_tree(X_train, X_test, y_train, y_test)
        
        # Call knn function
        knn_accuracy, knn_classification_report, knn_conf_matrix = knn(X_train, X_test, y_train, y_test)
        
        # Call isolation_forest function
        isolation_forest(df)
        
        # Get paths for the classification report and confusion matrix images generated by decision_tree function
        dt_classification_report_img = 'static/decision_tree_classification_report.png'
        dt_confusion_matrix_img = 'static/decision_tree_confusion_matrix.png'
        
        return render_template('results.html',
                               kmean='static/kmeans_clusters.png',
                               hmap='static/heatmap.png',
                               bar='static/bar_chart.png', 
                               accuracy=accuracy, 
                               classification_report=classification_rep, 
                               confusion_matrix=conf_matrix,
                               knn_accuracy=knn_accuracy,
                               knn_classification_report=knn_classification_report,
                               knn_confusion_matrix=knn_conf_matrix,
                               knn_classification_report_img='static/knn_classification_report.png',  # Update file path
                               knn_confusion_matrix_img='static/knn_confusion_matrix.png',  # Update file path
                               isolation_forest_img='static/isolation_forest.png',  # Add this line
                               decision_tree_classification_report_img='static/decision_tree_classification_report.png',
                               decision_tree_confusion_matrix_img='static/decision_tree_confusion_matrix.png')
    return redirect(url_for('index'))


@app.route('/download_report', methods=['GET'])
def download_report():
    filename = 'LOG_FILES.csv'
    file_path = os.path.join('uploads', filename)
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, header=None, names=["IP", "Request", "StatusCode", "Column8", "UserAgent", "ResponseTime"])
        df['ResponseTime'] = pd.to_numeric(df['ResponseTime'], errors='coerce')
        df = df.dropna(subset=['ResponseTime'])
        df['RequestType'] = df['Request'].apply(lambda x: x.split()[0])
        bar_chart(df)
        heatmap(df)
        kmeans_clusters(df)
        X = df[['StatusCode', 'ResponseTime']]
        y = df['Request'].apply(lambda x: x.split()[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        accuracy, classification_rep, conf_matrix = decision_tree(X_train, X_test, y_train, y_test)
        
        # Call knn function
        knn_accuracy, knn_classification_report, knn_conf_matrix = knn(X_train, X_test, y_train, y_test)
        
        # Call isolation_forest function
        isolation_forest(df)
        
        # Get paths for the classification report and confusion matrix images generated by decision_tree function
        decision_tree_classification_report_img='static/decision_tree_classification_report.png'
        decision_tree_confusion_matrix_img='static/decision_tree_confusion_matrix.png'
        knn_class = 'static/knn_classification_report.png'
        knn_conf = 'static/knn_confusion_matrix.png'
        iso_fo = 'static/isolation_forest.png'
        kmean='static/kmeans_clusters.png'
        hmap='static/heatmap.png'
        bar='static/bar_chart.png'
        # Check if the conf_matrix variable contains data
        render_conf_matrix = bool(conf_matrix.any()) if isinstance(conf_matrix, np.ndarray) else False
        
        # Render the results.html template to HTML string
        html_content = render_template('results.html',
                                       bar=bar,
                                       hmap=hmap,
                                       kmean=kmean,
                                       accuracy=accuracy,
                                       classification_report=classification_rep,
                                       confusion_matrix=conf_matrix,
                                       knn_accuracy=knn_accuracy,
                                       knn_classification_report=knn_classification_report,
                                       knn_confusion_matrix=knn_conf_matrix,
                                       knn_classification_report_img=knn_class,
                                       knn_confusion_matrix_img=knn_conf,
                                       isolation_forest_img=iso_fo,
                                       decision_tree_classification_report_img=decision_tree_classification_report_img,
                                       decision_tree_confusion_matrix_imgg=decision_tree_confusion_matrix_img,
                                       render_conf_matrix=render_conf_matrix)
        
        # Convert HTML string to PDF
        pdf = BytesIO()
        pisa_status = pisa.CreatePDF(html_content, dest=pdf)
        if pisa_status.err:
            return "PDF creation failed", 500
        
        # Set the filename for the downloaded PDF
        pdf.seek(0)
        return send_file(pdf, as_attachment=True, mimetype='application/pdf', download_name='report.pdf')
    else:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True)
