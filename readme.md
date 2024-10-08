# Data Augmentation for Aspect Sentiment Classification

## Project Overview
This project focuses on Aspect Sentiment Classification using the SemEval 2016 Task 5 Subtask 2 dataset. The objective is to determine the sentiment associated with specific aspects at the review level.

## Data
The project utilizes the SemEval 2016 dataset, which provides labeled data for aspect-based sentiment analysis, enabling effective training and evaluation of sentiment classification models.

## Data Augmentation Techniques
To enhance the model's performance and robustness, the following data augmentation techniques have been employed:

- **Back Translation (BT)**: Translating the text into another language and then back to the original language to generate paraphrases.
- **Keyboard Augmentation (KA)**: Introducing variations by simulating common typing errors.
- **Easy Data Augmentation (EDA)**: Utilizing techniques such as synonym replacement and random insertion to create diverse training examples.
- **Mixup**: Creating synthetic examples by blending two instances together.

## Model Training
For the classification task, I have fine-tuned both **BERT** and **RoBERTa** models to achieve accurate sentiment predictions for each aspect in the reviews.

