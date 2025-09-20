Project: PIRvision Occupancy Detection Using LSTM with 5-Fold Cross-Validation

This project demonstrates the development of an LSTM-based model for occupancy detection using the PIRvision dataset. The code includes datapreprocessing, exploratory analysis, model training via 5-fold cross-validation, checkpoints creation, and an evaluation function that loads a saved checkpoint to compute accuracy on test data. Detailed performance metrics 
(including mean accuracy, standard deviation, macro F1-score, per-class precision,
recall, F1, and a confusion matrix) are computed across all folds. Additionally, 
training and validation loss/accuracy curves are visualized.

Libraries Used:

1. Python 3.x
2. Pandas – for data manipulation and analysis.
3. Numpy – for numerical computations.
4. PyTorch – for building and training deep learning models.
5. scikit-learn – for data preprocessing, cross-validation, and evaluation metrics.
6. Matplotlib – for plotting training/validation curves and data visualizations.

Installation of Required Libraries:

You can install the necessary libraries using pip. For example, run:

    pip install pandas numpy torch scikit-learn matplotlib

Alternatively, if you use Anaconda, you may install them with:

    conda install pandas numpy scikit-learn matplotlib
    conda install pytorch torchvision torchaudio -c pytorch

Project Structure:

The project is organized in a Jupyter Notebook with the following main sections:
  • Imports & Global Settings: Loads libraries and sets up device configuration.
  • Data Loading & Preprocessing: Loads the dataset, checks for missing values, 
    computes summary statistics for sensor features, and normalizes the data.
  • Model Definition: Defines the PIRvisionLSTM model architecture and custom 
    weight initialization.
  • 5-Fold Cross-Validation Training: Trains the model using 5-fold cross-validation, 
    saves a checkpoint from the 3rd fold, and visualizes training/validation loss 
    and accuracy curves.
  • Evaluation Function: Implements the 'evaluate_model' function to load an 
    evaluation dataset and a model checkpoint, then computes and returns accuracy.
    
Running the Code:

1.. Execute Cells in Order:
   - Start by running the cell with the library imports and global settings.
   - Continue through each cell in the suggested order:
       a. Data Loading & Preprocessing
       b. Helper Functions and Model Definition
       c. 5-Fold Cross-Validation Training (this will also save checkpoints)
       d. Evaluation Function and its usage
   - Make sure that you run all cells to avoid missing dependencies or definitions.

2. Evaluating the Model:
   - After the training is completed and the checkpoint is saved (e.g., 'fold3_checkpoint.pth'),
     run the evaluation cell that calls the `evaluate_model` function.
   - The function will output the accuracy of the saved model on the specified dataset.

Additional Notes:

• Make sure that the dataset file 'pirvision_office_dataset1.csv' is located in the same
  directory as your notebook or provide the correct path.
• If you modify any parameters (such as hyperparameters or file paths), update them consistently 
  throughout the notebook.
• The evaluation function currently re-fits the scalers on the evaluation data. For a 
  more consistent evaluation, you may want to save and reuse the scalers from the training phase.