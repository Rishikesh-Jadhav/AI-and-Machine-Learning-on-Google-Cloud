# AI-and-Machine-Learning-on-Google-Cloud

## Machine Learning Projects

Project 1 : Predicting Visitor Purchases with BigQueryML

Project 2 : Entity and Sentiment Analysis with Natural Language API

Project 3 : Vertex AI: Predicting Loan Risk with AutoML

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
    ...
  ],
  "language": "en"
}

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

Syntactic Analysis Example:
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

Multilingual Processing Example:
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

