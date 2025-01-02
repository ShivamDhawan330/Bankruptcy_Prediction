from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or you can specify specific origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

def train_and_evaluate_model(x_train, y_train, x_test, y_test, class_weights=None):
    sc = StandardScaler()
    x_train2 = sc.fit_transform(x_train)
    x_test2 = sc.transform(x_test)
    model = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=1000, random_state=42)
    param_grid = {'C': [0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8, 1.0], 'class_weight': [None, 'balanced']}
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3)
    grid_search.fit(x_train2, y_train)
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test2)
    performance_lr = {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return best_model, performance_lr

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Company Bankruptcy Prediction</title>
    </head>
    <body>
        <style>
            body {{ background-color:#DAF7A6; font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{text-align:center; color:#008080}}
            h2 {{ color: #333; }}
        </style>
            
        <h1>COMPANY BANKRUPTCY PREDICTION</h1>
        
        <i>This application is to test the bankruptcy of the Company using <b>Logistic Regression Model</b> regularized with <b>ElasticNet</b> and hypertuned with <b>GridSearch</b> for the dataset present at : https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction/ </i>
        
        <h2>Upload your .csv data file</h2>
        <input type="file" id="fileInput">
        
        <button onclick="uploadFile()">Submit</button>
        <div id="output"></div>
        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append("file", file);

                const response = await fetch("/get_results/", {
                    method: "POST",
                    body: formData,
                });

                const output = document.getElementById('output');
                if (response.ok) {
                    const result = await response.text();  // Changed to .text() to handle HTML response
                    output.innerHTML = result;  // Inject HTML content directly into the page
                } else {
                    output.textContent = "Error: " + response.statusText;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/get_results/")
async def get_results(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        if 'Bankrupt?' not in df.columns:
            raise ValueError("Column 'Bankrupt?' not found in the uploaded file.")

        df2 = df.rename(columns={'Bankrupt?': 'Y'})
        y = df2['Y']
        X = df2.drop(columns=['Y'])

        if df2[df2['Y'] == 0].shape[0] == 0:
            raise ValueError("No Records for the class 0")
        elif df2[df2['Y'] == 1].shape[0] == 0:
            raise ValueError("No Records for the class 1")
        else:
            class_distribution = df2['Y'].value_counts().to_dict()

        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        best_model, performance_lr = train_and_evaluate_model(x_train, y_train, x_test, y_test)

        if hasattr(best_model, "coef_"):
            coefficients = best_model.coef_[0]
            column_names = df.columns[1:].to_list()
            variable_importance = {
                "positive Correlated variables": {col: coef for col, coef in zip(column_names, coefficients) if coef > 0.5},
                "negative Correlated variables": {col: coef for col, coef in zip(column_names, coefficients) if coef < -0.5},
            }
        else:
            variable_importance = {"positive Correlated variables": {}, "negative Correlated variables": {}}

        # Prepare HTML response with the results
        html_response = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Results</title>
            <style>
                body {{ background-color:#DAF7A6; font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1 {{text-align:center; color:#008080}}
                h2 {{ color: #333; }}
                pre {{ background: #f4f4f4; padding: 10px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Results</h1>

            <h2><strong>Class Distribution</strong></h2>
            <pre>{class_distribution}</pre>

            <h2><strong>GridSearch Performance</strong></h2>
            <h3><strong>Confusion Matrix</strong></h3>
            <table>
                {"".join([f"<tr>{''.join([f'<td>{cell}</td>' for cell in row])}</tr>" for row in performance_lr['confusion_matrix']])}
            </table>

            <h3><strong>Classification Report</strong></h3>
            <pre>{performance_lr["classification_report"]}</pre>

            <h2><strong>Important Variables</strong></h2>
            <h3><strong>Positive Correlated Variables</strong></h3>
            <pre>{variable_importance["positive Correlated variables"]}</pre>
            <h3><strong>Negative Correlated Variables</strong></h3>
            <pre>{variable_importance["negative Correlated variables"]}</pre>
        </body>
        </html>
        """

        return HTMLResponse(content=html_response)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Use the PORT environment variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
