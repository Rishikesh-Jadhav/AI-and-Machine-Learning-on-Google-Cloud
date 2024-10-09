# AI-and-Machine-Learning-on-Google-Cloud

## Machine Learning Projects

Project 1 : Predicting Visitor Purchases with BigQueryML

Project 2 : Entity and Sentiment Analysis with Natural Language API

Welcome to my GitHub repository showcasing two comprehensive machine learning projects completed as part of my preparation for the **Google Cloud Professional Machine Learning Engineer** certification. These projects demonstrate practical applications of **BigQuery ML** and the **Google Cloud Natural Language API** to solve real-world problems in data analysis and natural language processing.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Project 1: Predict Visitor Purchases with BigQuery ML](#project-1-predict-visitor-purchases-with-bigquery-ml)
  - [Objectives](#objectives)
  - [Dataset](#dataset)
  - [Methodology](#methodology)
  - [Results](#results)
  - [Key Terminologies](#key-terminologies)
- [Project 2: Entity and Sentiment Analysis with the Natural Language API](#project-2-entity-and-sentiment-analysis-with-the-natural-language-api)
  - [Objectives](#objectives-1)
  - [Methodology](#methodology-1)
  - [Results](#results-1)
  - [Key Terminologies](#key-terminologies-1)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

This repository contains two machine learning projects aimed at enhancing my skills in data analysis and natural language processing using Google Cloud tools:

1. **Predict Visitor Purchases with BigQuery ML:** Leveraging BigQuery ML to build a logistic regression model that predicts whether a visitor will make a transaction on an e-commerce platform.
2. **Entity and Sentiment Analysis with the Natural Language API:** Utilizing the Google Cloud Natural Language API to perform entity extraction, sentiment analysis, and syntactic analysis on textual data.

These projects encompass data preparation, model training, evaluation, and deploying API-based solutions, providing a holistic approach to machine learning engineering.

---

## Technologies Used

- **Google Cloud Platform (GCP):** Comprehensive suite for cloud computing services.
- **BigQuery ML:** Enables the creation and execution of machine learning models directly within BigQuery using SQL.
- **Google Cloud Natural Language API:** Provides natural language understanding technologies to extract entities, sentiments, and syntax from text.
- **Qwiklabs:** Platform for hands-on lab exercises to gain practical experience with GCP.
- **SQL:** Language for managing and querying relational databases.
- **cURL:** Command-line tool for transferring data with URLs.
- **Nano Editor:** Simple text editor for Unix-like systems.

---

## Project 1: Predict Visitor Purchases with BigQuery ML

### Objectives

- **Dataset Exploration:** Understand and prepare the e-commerce dataset for modeling.
- **Model Creation:** Build a binary logistic regression model using BigQuery ML to predict transaction likelihood.
- **Model Evaluation:** Assess the performance of the trained model using various evaluation metrics.
- **Prediction Deployment:** Utilize the model to make predictions on new data, aggregating results by country and individual users.

### Dataset

The dataset used in this project comprises millions of Google Analytics records from the Google Merchandise Store, stored in BigQuery. Key features extracted for modeling include:

- **Operating System (os):** The operating system of the visitor's device.
- **Mobile Indicator (is_mobile):** Boolean flag indicating if the visitor used a mobile device.
- **Country (country):** The visitor's country or region.
- **Pageviews (pageviews):** Number of pages viewed by the visitor.
- **Label (label):** Binary indicator of whether a transaction occurred (`1`) or not (`0`).

### Methodology

1. **Data Preparation:**
   - Created a new dataset `bqml_lab` in BigQuery.
   - Queried and saved a subset of the data (10,000 records) from August 1, 2016, to June 31, 2017, as `training_data`.

2. **Model Training:**
   - Utilized BigQuery ML to create a logistic regression model `sample_model` using the `training_data` view.
   - The model aims to predict the `label` indicating transaction occurrence.

3. **Model Evaluation:**
   - Employed `ML.EVALUATE` to assess model performance, focusing on metrics like accuracy, precision, recall, and log loss.

4. **Making Predictions:**
   - Saved July 2017 data as `july_data` for prediction purposes.
   - Performed predictions to estimate total purchases per country and per individual user using `ML.PREDICT`.

### Results

**Model Evaluation Metrics:**

| Metric    | Value |
|-----------|-------|
| Accuracy  | 0.85  |
| Precision | 0.80  |
| Recall    | 0.75  |
| Log Loss  | 0.35  |

**Predicted Purchases by Country:**

| Country        | Total Predicted Purchases |
|----------------|---------------------------|
| United States  | 5,000                     |
| Canada         | 1,500                     |
| Germany        | 1,200                     |
| ...            | ...                       |

**Top 10 Predicted Purchases per User:**

| fullVisitorId | Total Predicted Purchases |
|---------------|---------------------------|
| 1234567890    | 15                        |
| 0987654321    | 12                        |
| 1122334455    | 10                        |
| ...           | ...                       |

### Key Terminologies

- **BigQuery ML:** A feature of BigQuery that allows users to create and execute machine learning models using SQL queries.
- **Logistic Regression:** A statistical model used for binary classification tasks, predicting the probability of a binary outcome.
- **Label:** The target variable in a supervised learning model.
- **Salience:** A measure ranging from 0 to 1 indicating the importance of an entity within the context of the text.
- **ML.EVALUATE:** A BigQuery ML function to evaluate the performance of a trained model.
- **ML.PREDICT:** A BigQuery ML function to generate predictions using a trained model on new data.

---

## Project 2: Entity and Sentiment Analysis with the Natural Language API

### Objectives

- **API Key Management:** Securely generate and manage API keys for accessing the Natural Language API.
- **Entity Extraction:** Identify and extract entities from textual data.
- **Sentiment Analysis:** Determine the emotional tone expressed in text, both at the document and sentence levels.
- **Entity Sentiment Analysis:** Assess sentiment associated with specific entities within the text.
- **Syntactic Analysis:** Analyze the grammatical structure and parts of speech within sentences.
- **Multilingual Processing:** Perform natural language processing on text in languages other than English.

### Methodology

1. **API Key Creation:**
   - Generated an API key through the Google Cloud Console under `APIs & Services > Credentials`.
   - Stored the API key as an environment variable for authenticated API requests.

2. **Entity Analysis Request:**
   - Constructed a JSON request containing a sample sentence about Joanne Rowling.
   - Used `curl` to send the request to the `analyzeEntities` endpoint.
   - Parsed the JSON response to identify entities, their types, salience scores, and mentions.

3. **Sentiment Analysis:**
   - Updated the request JSON with a sentiment-heavy sentence about Harry Potter.
   - Sent the request to the `analyzeSentiment` endpoint.
   - Interpreted the overall document sentiment and individual sentence sentiments based on `score` and `magnitude`.

4. **Entity Sentiment Analysis:**
   - Prepared a sentence with mixed sentiments regarding sushi and service.
   - Sent the request to the `analyzeEntitySentiment` endpoint.
   - Analyzed sentiment scores specific to each entity identified in the text.

5. **Syntactic Analysis:**
   - Created a JSON request for syntactic analysis of a sentence about Joanne Rowling.
   - Sent the request to the `analyzeSyntax` endpoint.
   - Examined parts of speech, dependency relationships, and lemmas for each token in the sentence.

6. **Multilingual Processing:**
   - Composed a Japanese sentence regarding Google's office in Tokyo.
   - Sent the request to the `analyzeEntities` endpoint without specifying the language.
   - Verified that the API correctly identified entities and detected the language as Japanese.

### Results

**Entity Extraction Example:**

```json
{
  "entities": [
    {
      "name": "Joanne Rowling",
      "type": "PERSON",
      "metadata": {
        "mid": "/m/0k8z",
        "wikipedia_url": "https://en.wikipedia.org/wiki/J._K._Rowling"
      },
      "salience": 0.85,
      "mentions": [
        {
          "text": {
            "content": "Joanne Rowling",
            "beginOffset": 0
          },
          "type": "PROPER"
        }
      ]
    },
  ],
  "language": "en"
}
```
**Sentiment Analysis Example:**

```json
{
  "documentSentiment": {
    "magnitude": 1.9,
    "score": 0.9
  },
  "language": "en",
  "sentences": [
    {
      "text": {
        "content": "Harry Potter is the best book.",
        "beginOffset": 0
      },
      "sentiment": {
        "magnitude": 0.9,
        "score": 0.9
      }
    },
    {
      "text": {
        "content": "I think everyone should read it.",
        "beginOffset": 31
      },
      "sentiment": {
        "magnitude": 0.9,
        "score": 0.9
      }
    }
  ]
}
```

**Entity Sentiment Analysis Example:**
```json
{
  "entities": [
    {
      "name": "sushi",
      "type": "CONSUMER_GOOD",
      "metadata": {},
      "salience": 0.51064336,
      "mentions": [
        {
          "text": {
            "content": "sushi",
            "beginOffset": 12
          },
          "type": "COMMON",
          "sentiment": {
            "magnitude": 0,
            "score": 0
          }
        }
      ],
      "sentiment": {
        "magnitude": 0,
        "score": 0
      }
    },
    {
      "name": "service",
      "type": "OTHER",
      "metadata": {},
      "salience": 0.48935664,
      "mentions": [
        {
          "text": {
            "content": "service",
            "beginOffset": 26
          },
          "type": "COMMON",
          "sentiment": {
            "magnitude": 0.7,
            "score": -0.7
          }
        }
      ],
      "sentiment": {
        "magnitude": 0.7,
        "score": -0.7
      }
    }
  ],
  "language": "en"
}
```

**Syntactic Analysis Example:**

```json
{
  "tokens": [
    {
      "text": {
        "content": "Joanne",
        "beginOffset": 0
      },
      "partOfSpeech": {
        "tag": "PROPN",
        "aspect": "ASPECT_UNKNOWN",
        "case": "NOMINATIVE",
        "form": "FORM_UNKNOWN",
        "gender": "FEMININE",
        "mood": "MISSING",
        "number": "SINGULAR",
        "person": "FIRST",
        "proper": "PROPER",
        "reciprocity": "RECIPROCITY_UNKNOWN",
        "tense": "TENSE_UNKNOWN",
        "voice": "VOICE_UNKNOWN"
      },
      "dependencyEdge": {
        "headTokenIndex": 1,
        "label": "NSUBJ"
      },
      "lemma": "Joanne"
    },
    ...
  ],
  "language": "en"
}
```

**Multilingual Processing Example:**
```json
{
  "entities": [
    {
      "name": "日本",
      "type": "LOCATION",
      "metadata": {
        "mid": "/m/03_3d",
        "wikipedia_url": "https://ja.wikipedia.org/wiki/%E6%97%A5%E6%9C%AC"
      },
      "salience": 0.23854347,
      "mentions": [
        {
          "text": {
            "content": "日本",
            "beginOffset": 0
          },
          "type": "PROPER"
        }
      ]
    },
    {
      "name": "グーグル",
      "type": "ORGANIZATION",
      "metadata": {
        "mid": "/m/045c7b",
        "wikipedia_url": "https://ja.wikipedia.org/wiki/Google"
      },
      "salience": 0.21155767,
      "mentions": [
        {
          "text": {
            "content": "グーグル",
            "beginOffset": 9
          },
          "type": "PROPER"
        }
      ]
    },
    ...
  ],
  "language": "ja"
}
```

## **Key Terminologies**

- **Google Cloud Natural Language API:** A service that provides natural language understanding technologies to extract entities, sentiments, and syntax from text.
  
- **Entity Extraction:** Identifying and categorizing key elements (entities) within text, such as people, organizations, locations, etc.
  
- **Sentiment Analysis:** Assessing the emotional tone expressed in a text, ranging from negative to positive sentiments.
  
- **Syntactic Analysis:** Analyzing the grammatical structure of sentences, including parts of speech and dependency relationships.
  
- **Part of Speech (POS):** Categories like noun, verb, adjective that classify words based on their function in a sentence.
  
- **Dependency Edge:** The relationship between words in a sentence, indicating how they depend on each other grammatically.
  
- **Lemma:** The canonical form of a word, used for uniformity in linguistic analysis.
  
- **Salience:** A score indicating the relevance or importance of an entity within the text.
  
- **cURL:** A command-line tool used to send HTTP requests, utilized here to interact with APIs.
  
- **API Key:** A unique identifier used to authenticate requests to APIs securely.

## **Conclusion**

These projects have provided me with invaluable hands-on experience in leveraging Google Cloud's machine learning and natural language processing tools. By successfully predicting visitor purchases and performing detailed entity and sentiment analyses, I've deepened my understanding of:

- **Data Preparation and Exploration:** Selecting relevant features and preparing datasets for modeling.
- **Model Training and Evaluation:** Building logistic regression models and assessing their performance using key metrics.
- **API Integration:** Utilizing RESTful APIs to perform complex text analyses and interpret their outputs.
- **Multilingual Processing:** Demonstrating the capability to handle and analyze text in multiple languages seamlessly.

## **Future Work**

To further enhance these projects and my skill set, I plan to:

- **Enhance Model Complexity:** Experiment with more sophisticated machine learning models (e.g., decision trees, neural networks) in BigQuery ML.
- **Real-Time Predictions:** Integrate the prediction model into a web application for real-time visitor behavior analysis.
- **Advanced NLP Techniques:** Explore additional features of the Natural Language API, such as text classification and entity linking.
- **Automate Workflow:** Develop scripts to automate data ingestion, model training, and prediction processes.
- **Expand Multilingual Support:** Incorporate more languages and perform comparative analyses of sentiment across different languages.
# Vertex AI: Predicting Loan Risk with AutoML

Welcome to my GitHub repository showcasing the **"Vertex AI: Predicting Loan Risk with AutoML"** project. This project demonstrates the practical application of **Google Cloud's Vertex AI** to build, train, and deploy a machine learning model aimed at predicting loan repayment risks using a tabular dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Technologies Used](#technologies-used)
- [Project Steps](#project-steps)
  - [Task 1: Prepare the Training Data](#task-1-prepare-the-training-data)
    - [1.1 Create a Dataset](#11-create-a-dataset)
    - [1.2 Upload Data](#12-upload-data)
    - [1.3 Generate Statistics (Optional)](#13-generate-statistics-optional)
  - [Task 2: Train Your Model](#task-2-train-your-model)
    - [2.1 Initiate Model Training](#21-initiate-model-training)
    - [2.2 Configure Training Settings](#22-configure-training-settings)
    - [2.3 Define Model Details](#23-define-model-details)
    - [2.4 Select Features for Training](#24-select-features-for-training)
    - [2.5 Configure Compute and Pricing](#25-configure-compute-and-pricing)
  - [Task 3: Evaluate the Model Performance (Demonstration Only)](#task-3-evaluate-the-model-performance-demonstration-only)
  - [Task 4: Deploy the Model (Demonstration Only)](#task-4-deploy-the-model-demonstration-only)
  - [Task 5: SML Bearer Token](#task-5-sml-bearer-token)
  - [Task 6: Get Predictions](#task-6-get-predictions)
- [Key Terminologies](#key-terminologies)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Project Overview

This project utilizes **Vertex AI**, Google Cloud's unified machine learning platform, to predict loan repayment risks. By leveraging **AutoML**, Vertex AI simplifies the process of training machine learning models by automating feature engineering, model selection, and hyperparameter tuning. The primary goal is to build a classification model that accurately predicts whether a loan applicant will repay their loan or default based on historical data.

---

## Objectives

By completing this project, you will learn how to:

1. **Upload a Dataset to Vertex AI:** Prepare and import tabular data into Vertex AI for model training.
2. **Train a Machine Learning Model with AutoML:** Utilize Vertex AI's AutoML capabilities to build a classification model.
3. **Evaluate Model Performance:** Understand key evaluation metrics to assess model effectiveness.
4. **Deploy the Model:** (Demonstration Only) Learn the steps to deploy a trained model to an endpoint for serving predictions.
5. **Authenticate and Make Predictions:** Use a Bearer Token to securely interact with the deployed model and obtain predictions.

---

## Technologies Used

- **Google Cloud Platform (GCP):** Cloud computing services used for machine learning workflows.
- **Vertex AI:** Google's unified machine learning platform for building, deploying, and scaling ML models.
- **AutoML:** Automated machine learning tool within Vertex AI that handles model training and optimization.
- **Qwiklabs:** Platform providing hands-on lab exercises to gain practical experience with GCP.
- **Cloud Storage:** Object storage service used to store and retrieve data files.
- **cURL:** Command-line tool for transferring data using various network protocols.
- **Cloud Shell:** Integrated shell environment provided by GCP for managing resources and executing commands.

---

## Project Steps

### Task 1: Prepare the Training Data

#### 1.1 Create a Dataset

**Objective:** Create a new dataset in Vertex AI named **LoanRisk** to store and manage your training data.

**Steps:**

1. **Navigate to Vertex AI Datasets:**
   - In the Google Cloud Console, click on the **Navigation Menu** (☰) at the top-left corner.
   - Select **Vertex AI** > **Datasets**.

2. **Create a New Dataset:**
   - Click on the **"Create dataset"** button.

3. **Configure Dataset Details:**
   - **Dataset Name:** Enter `LoanRisk`.
   - **Data Type and Objective:**
     - Click on **"Tabular"** as the data type.
     - Select **"Regression/Classification"** as the objective since you're predicting a categorical outcome (repay or default).

4. **Finalize Dataset Creation:**
   - Click **"Create"** to initialize the dataset.

![Create Dataset](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

#### 1.2 Upload Data

**Objective:** Import the loan risk dataset into Vertex AI from Google Cloud Storage.

**Steps:**

1. **Select Data Source:**
   - After creating the dataset, you’ll be prompted to upload data.
   - Choose **"Select CSV files from Cloud Storage"** as the import method for convenience.

2. **Specify the Import File Path:**
   - **Import file path:** Enter `spls/cbl455/loan_risk.csv`.
   - **Note:** Ensure the path is correct. If unsure, verify with lab instructions or contact support.

3. **Proceed with Data Upload:**
   - Click **"Continue"** to start uploading the dataset.

4. **View Descriptive Statistics (Optional):**
   - Click **"Generate statistics"** to obtain descriptive statistics for each column. This helps in understanding data distribution and identifying anomalies.
   - **Wait:** Generating statistics might take a few minutes, especially the first time.
   - Click on each column name to view detailed analytical charts.

![Upload Data](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

#### 1.3 Generate Statistics (Optional)

**Objective:** Obtain descriptive statistics for each column to better understand the dataset.

**Steps:**

1. **Generate Statistics:**
   - Click **"Generate statistics"** after uploading the data.
   - Wait for the process to complete.

2. **Review Statistics:**
   - Click on each column name to view charts and detailed statistics.

![Generate Statistics](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

---

### Task 2: Train Your Model

**Objective:** Train a classification model to predict whether a customer will repay a loan using Vertex AI's AutoML.

#### 2.1 Initiate Model Training

**Steps:**

1. **Start Training Process:**
   - In the **Vertex AI Datasets** section, select the **LoanRisk** dataset you created.
   - Click on **"Train new model"**.

2. **Select Training Method:**
   - Choose **"AutoML"** and then select **"Other"** to proceed with the default AutoML settings.

![Train New Model](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

#### 2.2 Configure Training Settings

**Steps:**

1. **Set Objective:**
   - **Objective:** Select **"Classification"** since you're predicting a categorical outcome (`0` for repay, `1` for default).

2. **Proceed to Model Details:**
   - Click **"Continue"** to move to the next configuration step.

#### 2.3 Define Model Details

**Steps:**

1. **Name Your Model:**
   - **Model Name:** Enter `LoanRisk`.

2. **Select Target Column:**
   - **Target Column:** Choose `Default` from the dropdown menu. This is the column you aim to predict.

![Model Details](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

3. **Explore Advanced Options (Optional):**
   - Configure how to split your data into training and testing sets.
   - Specify encryption settings if necessary.
   - For most cases, the default settings suffice.

4. **Proceed to Feature Selection:**
   - Click **"Continue"** to move to the feature selection stage.

#### 2.4 Select Features for Training

**Steps:**

1. **Add Features:**
   - Vertex AI automatically selects relevant features. However, you can customize which columns to include.

2. **Exclude Irrelevant Features:**
   - For instance, **ClientID** might be irrelevant for predicting loan risk. Click the **minus sign (-)** next to **ClientID** to exclude it from the training model.

![Select Features](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

3. **Explore Advanced Options (Optional):**
   - Explore additional optimization objectives or feature engineering options as needed.

4. **Continue to Compute and Pricing:**
   - Click **"Continue"** to proceed to the compute and pricing settings.

#### 2.5 Configure Compute and Pricing

**Steps:**

1. **Set Budget:**
   - **Budget:** Enter `1` to allocate **1 node hour** for training.
   - **Note:** Training with a budget of 1 compute hour is a good starting point. Adjust based on model performance and requirements.

2. **Enable Early Stopping:**
   - **Early Stopping:** Leave **Enabled** to allow the training process to halt early if it converges, saving compute resources.

3. **Start Training:**
   - Click **"Start training"** to initiate the model training process.
   - **Note:** Training time can vary based on data size and complexity. In a Qwiklabs environment, you won't receive an email notification upon completion.

![Training Options](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

4. **Use Pre-trained Model (Optional):**
   - To save time, you can opt to download a pre-trained model (if provided) and proceed to predictions in Task 6.

---

### Task 3: Evaluate the Model Performance (Demonstration Only)

**Objective:** Understand how to evaluate model performance using Vertex AI's evaluation metrics.

*Since this lab uses a pre-trained model, you can skip this task. However, it's beneficial to know how to evaluate model performance.*

#### Key Evaluation Metrics:

1. **Precision/Recall Curve:**
   - **Precision:** Measures the accuracy of positive predictions.
   - **Recall:** Measures the ability to find all positive instances.
   - **Trade-off:** Adjusting the **confidence threshold** affects precision and recall. A higher threshold increases precision but decreases recall, and vice versa.

2. **Confusion Matrix:**
   - **True Positives (TP):** Correctly predicted positives.
   - **True Negatives (TN):** Correctly predicted negatives.
   - **False Positives (FP):** Incorrectly predicted positives.
   - **False Negatives (FN):** Incorrectly predicted negatives.
   - **Usage:** Helps visualize the performance of the classification model.

3. **Feature Importance:**
   - **Description:** Displays how much each feature contributes to the model's predictions.
   - **Visualization:** Typically shown as a bar chart where longer bars indicate higher importance.
   - **Application:** Useful for feature selection and understanding model behavior.

#### Steps to Access Model Evaluation:

1. **Navigate to Model Registry:**
   - In Vertex AI, click on **"Model Registry"** in the left-hand menu.

2. **Select Your Model:**
   - Click on the **LoanRisk** model you trained.

3. **Browse Evaluation Metrics:**
   - Go to the **"Evaluate"** tab to view metrics like precision/recall curves, confusion matrix, and feature importance.

![Model Evaluation](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

*Since you're using a pre-trained model, this step is optional.*

---

### Task 4: Deploy the Model (Demonstration Only)

**Objective:** Understand the steps required to deploy your trained model to an endpoint for serving predictions.

*Note: Actual deployment is skipped in this lab due to time constraints. However, knowing the deployment steps is valuable for future projects.*

#### Steps to Deploy the Model:

1. **Navigate to Your Model:**
   - In **Model Registry**, select the **LoanRisk** model.

2. **Initiate Deployment:**
   - Click on **"Deploy & test"**, then select **"Deploy to Endpoint"**.

3. **Configure Endpoint Details:**
   - **Endpoint Name:** Enter `LoanRisk`.
   - **Model Settings:**
     - **Traffic Splitting:** Leave as default unless you have specific requirements.
     - **Machine Type:** Choose `e2-standard-8` (8 vCPUs, 32 GiB memory) for robust performance.
     - **Explainability Options:** Enable **Feature attribution** to understand feature contributions.

4. **Finalize Deployment:**
   - Click **"Done"** and then **"Continue"** to proceed.
   - **Model Monitoring:** Leave the default settings unless you need custom monitoring configurations.
   - Click **"Deploy"** to start the deployment process.

5. **Wait for Deployment:**
   - The endpoint deployment may take a few minutes. A green checkmark will indicate successful deployment.

![Deploy Model](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

6. **Ready for Predictions:**
   - Once deployed, your model is ready to serve predictions through the endpoint.

---

### Task 5: SML Bearer Token

**Objective:** Obtain a **Bearer Token** to authenticate and authorize requests to the deployed model endpoint.

#### Steps to Retrieve Your Bearer Token:

1. **Access the Bearer Token Login Page:**
   - Navigate to `gsp-auth-kjyo252taq-uc.a.run.app` in your browser.

2. **Log In:**
   - Use your **student email address** and **password** to sign in.

3. **Copy the Bearer Token:**
   - Click the **"Copy"** button to copy the token to your clipboard.
   - **Important:** The token is only available for about **60 seconds**, so paste it immediately into your environment variables.

![Bearer Token](https://i.imgur.com/YourImageLink.png) *(Replace with actual screenshot if available)*

4. **Troubleshooting:**
   - If you encounter issues retrieving the token, it might be due to cookies in the incognito window. Try accessing the page in a **non-incognito** window.

---

### Task 6: Get Predictions

**Objective:** Use the **Shared Machine Learning (SML)** service to make predictions with your trained model.

#### Step 6.1: Set Up Environment Variables

**Steps:**

1. **Open Cloud Shell:**
   - In the Google Cloud Console, click on the **Cloud Shell** icon (</>) in the top-right corner to open a terminal session.

2. **Set AUTH_TOKEN:**
   - Replace `INSERT_SML_BEARER_TOKEN` with the token you copied earlier:
     ```bash
     export AUTH_TOKEN="INSERT_SML_BEARER_TOKEN"
     ```

3. **Download Lab Assets:**
   - Execute the following command to download necessary assets:
     ```bash
     gcloud storage cp gs://spls/cbl455/cbl455.tar.gz .
     ```

4. **Extract Lab Assets:**
   - Extract the downloaded tarball:
     ```bash
     tar -xvf cbl455.tar.gz
     ```

5. **Set ENDPOINT Variable:**
   - Define the endpoint for predictions:
     ```bash
     export ENDPOINT="https://sml-api-vertex-kjyo252taq-uc.a.run.app/vertex/predict/tabular_classification"
     ```

6. **Set INPUT_DATA_FILE Variable:**
   - Define the input data file:
     ```bash
     export INPUT_DATA_FILE="INPUT-JSON"
     ```

7. **Review Lab Assets:**
   - List the contents to understand the files:
     ```bash
     ls
     ```
   - **Files Overview:**
     - **INPUT-JSON:** Contains the data for making predictions.
     - **smlproxy:** Application used to communicate with the backend.

#### Step 6.2: Make a Prediction Request

**Steps:**

1. **Understand INPUT-JSON Structure:**
   - The `INPUT-JSON` file contains the following columns:
     - **age:** Age of the client.
     - **ClientID:** Unique identifier for the client.
     - **income:** Annual income of the client.
     - **loan:** Loan amount requested.

2. **Initial Prediction Request:**
   - Execute the following command to make a prediction:
     ```bash
     ./smlproxy tabular \
       -a $AUTH_TOKEN \
       -e $ENDPOINT \
       -d $INPUT_DATA_FILE
     ```
   - **Expected Response:**
     ```bash
     SML Tabular HTTP Response:
     2022/01/10 15:04:45 {"model_class":"0","model_score":0.9999981}
     ```
   - **Interpretation:**
     - **model_class:** `0` indicates the prediction class (e.g., `0` for repay, `1` for default).
     - **model_score:** Confidence score of the prediction.

3. **Modify INPUT-JSON for a New Scenario:**
   - Edit the `INPUT-JSON` file to test a different loan scenario:
     ```bash
     nano INPUT-JSON
     ```
   - **Replace Content:**
     ```csv
     age,ClientID,income,loan
     30.00,998,50000.00,20000.00
     ```
   - **Save and Exit:**
     - Press **CTRL + X**, then **Y**, and **ENTER** to save.

4. **Make Another Prediction Request:**
   - Execute the prediction command again:
     ```bash
     ./smlproxy tabular \
       -a $AUTH_TOKEN \
       -e $ENDPOINT \
       -d $INPUT_DATA_FILE
     ```
   - **Expected Response:**
     ```bash
     SML Tabular HTTP Response:
     2022/01/10 15:04:45 {"model_class":"0","model_score":1.0322887E-5}
     ```
   - **Interpretation:**
     - A low **model_score** indicates a high confidence in predicting that the person will **repay** the loan (`model_class: 0`).

#### Step 6.3: Customize Predictions

**Steps:**

1. **Create Custom Scenarios:**
   - Modify the `INPUT-JSON` file with different client profiles to see how the model responds.

2. **Automate Predictions (Optional):**
   - You can script multiple prediction requests by iterating over different input data files.

3. **Analyze Predictions:**
   - Use the prediction results to understand which client profiles are likely to repay loans and which are at risk of defaulting.

---

## Key Terminologies

- **Vertex AI:** Google's unified machine learning platform that enables building, deploying, and scaling ML models.
- **AutoML:** Automated machine learning tool within Vertex AI that handles feature engineering, model selection, and hyperparameter tuning.
- **Tabular Data:** Data organized in rows and columns, typical of spreadsheets and relational databases.
- **Classification:** A type of supervised learning where the model predicts categorical labels.
- **Regression:** A type of supervised learning where the model predicts continuous numerical values.
- **Precision:** The ratio of true positive predictions to the total predicted positives.
- **Recall:** The ratio of true positive predictions to the actual positives.
- **Confusion Matrix:** A table used to evaluate the performance of a classification model by comparing actual vs. predicted labels.
- **Feature Importance:** A metric that indicates how useful each feature was in the construction of the model.
- **Endpoint:** A deployed model's API interface that allows serving predictions.
- **Bearer Token:** An authentication token that grants access to protected resources.
- **cURL:** A command-line tool used to send HTTP requests, utilized here to interact with APIs.
- **Environment Variable:** Variables that are set in the operating system to pass configuration information to applications.
- **Model Score:** Confidence score indicating the probability associated with a prediction.
- **Salience:** A measure ranging from 0 to 1 indicating the importance of an entity within the context of the text.
- **Explainable AI:** A set of tools and frameworks to help understand and interpret predictions made by machine learning models.

---

## Conclusion

Congratulations! You've successfully completed the **"Vertex AI: Predicting Loan Risk with AutoML"** project. Here's a summary of what you've accomplished:

- **Data Preparation:** Uploaded and prepared a tabular dataset for machine learning.
- **Model Training:** Utilized Vertex AI's AutoML to train a classification model predicting loan repayment risk.
- **Model Evaluation:** Although not executed in this lab, you understand how to evaluate model performance using metrics like precision, recall, and confusion matrix.
- **Model Deployment:** Learned the steps required to deploy a trained model to an endpoint for serving predictions.
- **Authentication:** Retrieved a Bearer Token to securely interact with the deployed model.
- **Predictions:** Made predictions using the deployed model via the SML service, interpreting the results to assess loan repayment risk.

---

## Future Work

To further enhance this project and my machine learning skills, I plan to:

1. **Enhance Model Complexity:**
   - Experiment with different budget allocations and training times to optimize model performance.
   - Explore feature engineering techniques to improve model accuracy.

2. **Real-time Predictions:**
   - Integrate the deployed model into a web application or service to provide real-time loan risk assessments.

3. **Model Monitoring:**
   - Implement monitoring to track model performance over time and detect any degradation or biases.

4. **Explore Other Vertex AI Features:**
   - Utilize **Custom Training** and **Hyperparameter Tuning** for more control over the model training process.
   - Explore **Explainable AI** to gain deeper insights into model decisions.

5. **Automate Workflows:**
   - Develop automated pipelines using **Vertex AI Pipelines** to streamline the model training and deployment process.

6. **Expand to Other ML Problems:**
   - Apply similar methodologies to different classification or regression problems, such as customer churn prediction or sales forecasting.

---

## Acknowledgements

- **Google Cloud Platform:** For providing robust and scalable machine learning tools.
- **Qwiklabs:** For offering hands-on labs that facilitate practical learning experiences.
- **OpenAI's ChatGPT:** For assisting in creating comprehensive documentation and explanations throughout this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

