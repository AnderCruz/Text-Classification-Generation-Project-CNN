# Text Classification and Generation Project

## Project Description

This project explores natural language processing techniques using TensorFlow and Keras, focusing on challenges faced by a **news portal**. The portal requires solutions for **classifying articles** based on their titles and descriptions, and for **assisting writers** in maintaining a consistent style by suggesting subsequent words.

The project addresses these needs through two main tasks:

1.  **Text Classification:** Classifying the portal's news articles (based on title and description) into one of four categories using various neural network architectures (Dense, CNN, LSTM). It includes hyperparameter tuning and cross-validation to find the optimal LSTM model for categorization.
2.  **Text Generation:** Training a model based on Bidirectional LSTMs to predict the next word in a sequence, enabling **style-consistent word suggestions** for article writers.


## Table of Contents

* [Project Description](#project-description)
* [Features](#features)
* [Dataset](#dataset)
* [Methodology](#methodology)
    * [Code Highlights](#code-highlights)
    * [Text Classification](#text-classification-article-categorization)
    * [Text Generation](#text-generation-next-word-suggestion)
* [Technologies Used](#technologies-used)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Results](#results)
* [GitHub Topics (Tags)](#github-topics-tags)


## Features

* Data loading and preprocessing using Pandas.
* Text vectorization using `tf.keras.layers.TextVectorization`.
* Implementation and comparison of text classification models for article categorization:
    * Simple Dense Network
    * Convolutional Neural Network (1D CNN)
    * Bidirectional Long Short-Term Memory (LSTM) Network
* Hyperparameter optimization using Keras Tuner (`Hyperband`) with K-Fold Cross-Validation.
* Model evaluation using accuracy/loss curves and confusion matrix.
* N-gram sequence generation for language modeling.
* Training a Bidirectional LSTM model for next-word prediction (word suggestion).
* Text generation function with temperature control for sampling diversity.
* Saving and loading the vectorizer (`pickle`) and trained models (`.keras`).


## Dataset

* The data is loaded from a local zipped CSV file: `data/train.zip`.
* The CSV contains three columns: `ClassIndex`, `Título` (Title), and `Descrição` (Description), representing news article data.
* The `Título` and `Descrição` columns are concatenated into a single `Texto` (Text) column for processing.
* There are 4 distinct classes (categories), indexed 0 to 3 (originally 1 to 4 in the CSV).
* *(Note: The specific dataset source, like AG News, is not mentioned in the code but the structure is consistent with such datasets).*


## Methodology

### Code Highlights

* **Data Loading & Preprocessing:** Uses `pandas.read_csv` to load the dataset. The 'Title' (`Título`) and 'Description' (`Descrição`) are combined into a single 'Text' (`Texto`) feature (`df['Texto'] = df['Título'] + ' ' + df['Descrição']`) which serves as the input for the models. Class indices are adjusted to start from 0.
* **Text Vectorization:** `tf.keras.layers.TextVectorization` is crucial for converting raw text strings into sequences of integer token indices that neural networks can process. Key parameters include `max_tokens` (vocabulary size) and `output_sequence_length` (fixed length for sequences, padding if necessary). The `adapt()` method builds the vocabulary from the training corpus.
* **Model Architectures:**
    * `Embedding` Layer: Learns dense vector representations for each word in the vocabulary. `mask_zero=True` handles variable sequence lengths efficiently.
    * `Dense` Layer: Standard fully connected layer used for basic classification or feature transformation.
    * `Conv1D` Layer: Applies 1D convolutions, useful for extracting local patterns (like n-grams) from sequences.
    * `LSTM` / `Bidirectional(LSTM)`: Captures sequential dependencies and context in the text, crucial for understanding meaning and generating coherent sequences. Bidirectional LSTMs process the sequence in both forward and backward directions.
    * `GlobalAveragePooling1D`: Reduces the sequence dimension by averaging across the sequence length, often used after recurrent or convolutional layers.
    * `Dropout`: Regularization technique to prevent overfitting by randomly setting a fraction of input units to 0 during training.
* **Hyperparameter Tuning:** `keras_tuner.Hyperband` automates the search for optimal hyperparameters (like embedding dimension, LSTM units, dense units, dropout rate). It efficiently explores different combinations. `sklearn.model_selection.KFold` is used within the tuning process (`run_tuner` function) to ensure the selected hyperparameters generalize well across different subsets of the data (5-fold cross-validation).
* **N-gram Generation (Text Generation):** Input sequences for the generation task are created by taking sliding windows (n-grams) from the tokenized text. For a sentence `[10, 25, 5, 30]`, the sequences generated would be `[10, 25]`, `[10, 25, 5]`, and `[10, 25, 5, 30]`. `pad_sequences` ensures all input sequences have the same length before feeding them to the model.
* **Text Generation Logic:** The `predict_next_words` function takes the trained model and vectorizer, processes the input `text`, predicts probabilities for all words in the vocabulary as the next word. `Temperature` scaling is applied to the predicted probabilities (`preds = np.log(preds + 1e-9) / temperature; exp_preds = np.exp(preds); preds = exp_preds / np.sum(exp_preds)`) to control randomness: lower temperature makes predictions more deterministic (picks the most likely word), higher temperature increases diversity. The `generate_text` function iteratively calls this prediction function to build a sequence of generated words.

### Text Classification (Article Categorization)

1.  **Preprocessing:** Load data, combine text fields, adjust class index, split into train/test sets.
2.  **Encoding:** Create and adapt `TextVectorization` (vocab size 2500) on training data.
3.  **Model Training & Comparison:** Train baseline Dense, CNN, and Bidirectional LSTM models.
4.  **Hyperparameter Tuning (LSTM):** Use `keras_tuner.Hyperband` with `KFold` to find the best hyperparameters based on validation accuracy.
5.  **Final Model Training & Evaluation:** Build and train the final LSTM model using optimal hyperparameters on the full training set. Evaluate on the test set using accuracy/loss plots and confusion matrix.

### Text Generation (Next-Word Suggestion)

1.  **Preprocessing:** Sample data (4000 texts), create and adapt a new `TextVectorization` (vocab size 4000, seq length 50), generate and pad N-gram sequences, split into X (inputs) and y (next word targets).
2.  **Model Training:** Build and train a Bidirectional LSTM model for next-word prediction using callbacks (`EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau`).
3.  **Generation:** Use the trained model and helper functions (`predict_next_words`, `generate_text`) with temperature sampling to generate text sequences. Save the final model.


## Technologies Used

* Python 3
* TensorFlow
* Keras
* Keras Tuner
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Pickle


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    * *(Create a `requirements.txt` file based on the notebook imports)*
    ```bash
    pip install -r requirements.txt
    ```
    *(Example content for requirements.txt):*
    ```
    tensorflow
    pandas
    numpy
    scikit-learn
    matplotlib
    keras-tuner
    ```
4.  **Data:**
    * Ensure the `train.zip` file is placed inside a `data/` directory in the project root (`data/train.zip`).


## Usage

1.  Activate the virtual environment.
2.  Start Jupyter Notebook or Jupyter Lab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Open the `Project.ipynb` notebook.
4.  Run the cells sequentially to execute the data processing, model training, evaluation, and text generation steps.
5.  Saved artifacts include:
    * `vectorizer.pkl` (for text generation)
    * Models saved by `ModelCheckpoint` during generation training (e.g., `best_model.keras`)
    * Final generation models (e.g., `seer_model.keras`, `futuristic_seer_model.keras`)
    * Keras Tuner results directory (`my_dir/`)


## Results

* **Classification:** The performance of different models in categorizing news articles is evaluated using accuracy and loss plots. The final optimized LSTM model's categorization performance on the test set is visualized with a confusion matrix. *(Refer to the plots and matrix in the notebook output for specific metrics).*
* **Generation:** The project demonstrates the ability to generate style-consistent next-word suggestions based on a seed phrase using the trained LSTM language model. *(See notebook output for examples of generated text).*
