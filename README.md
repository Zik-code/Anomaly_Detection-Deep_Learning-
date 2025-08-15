
### 一、Project Introduction
This is a project that integrates several deep learning models based on the TranAD and DTAAD baseline models for anomaly detection in time-series data.

### 二、运行项目
Install Python >= 3.7 and PyTorch<br>
 
Please execute the following command to install PyTorch:

```python
pip install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```
Please execute the following command to install the required libraries:<br>

```python
pip install -r requirements.txt
```
The datasets used in this project have been preprocessed and placed in the processed_data folder.<br>
To run the model on the dataset and reproduce the results, please execute the following commands:<>

```python
python main.py --model <model> --dataset <dataset> # 加载预训练模型进行训练
```
```python
python main.py --model <model> --dataset <dataset> --retrain # 重新训练模型
```
where the model can be any one of 'TranAD', 'OmniAnomaly', or 'DTAAD', and the dataset can be any one of 'SMAP', 'SMD', or 'NAB'.

### 三、Overview of the Overall Process for Reconstruction-Based Anomaly Detection (Important)
    
Taking the SMD dataset's machine-1-1 as an example, the dataset is divided into a training set train, a test set test, and a label set label (which identifies anomalies in the test set). All these training, test, and label sets are two-dimensional arrays (28479×38), where rows represent time steps (28479 detection time points) and columns represent feature dimensions (38). A single data point in the two-dimensional array represents the data of one feature at one time point<br>
    
Taking TranAD as an example, both the training set and the test set are divided into windowed data with a window size of 10 (10×38). When extracting data for each window, it slides one step at a time, and these windows are stacked layer by layer to form a three-dimensional array. The training set is input into the model, and the data at the last time step in the window (i.e., the data at the current time step) is taken as the reconstruction target. The mean squared error (loss function/anomaly score) between the data reconstructed by the model and the original data is calculated, and the goal of training the model is to minimize this loss function.<br>

All anomaly scores of the training set are used to calculate the threshold for determining anomalies through the pot algorithm. Specifically, the principle of the pot algorithm is to first select the 98th percentile of the anomaly scores as the initial threshold. Anomaly scores exceeding 98% of the data (i.e., values with large reconstruction errors) are a small number of peaks.<br>

These peaks approximately follow a Pareto distribution (a statistical model similar to the Gaussian distribution). The Pareto curve is used to fit these peak data (calculating the two parameters of the GDF), and the final threshold for determining anomalies is determined through a threshold calculation formula. This threshold is used to judge the anomaly scores of the test dataset. Time points with anomaly scores greater than the threshold are marked as anomalies, forming an anomaly  pred. Then, pred is compared with label to calculate TP and TN (where TP refers to time points where both pred and label are 1, i.e., actual anomalies that are detected as anomalies), and further calculate the F1 score.

**POT Function in Time-Series Anomaly Detection:**
To detect time-series anomalies, it is necessary to further correct the anomaly marker sequence pred obtained above. This is specifically achieved through a function method (adjust_predicts()). If the data at the current time point of the test set is determined to be an anomaly (marked as 1) and the corresponding real label is also an anomaly (marked as 1) but has not been marked as an anomaly segment, we trace back to the real label position and modify the corresponding judgment position in pred to True (indicating an anomaly). Finally, the function returns the corrected pred (the specific reason for doing this will be explained later), and then the general F1 calculation is performed.<br>


### 四、 Project Construction 
The core modules consist of three files:<br>
  
- `models.py`： Contains the definitions of multiple core deep learning models, which are the basis for algorithm implementation.<br>
- `backprop.py`：Adopts a "base class + subclass" strategy pattern design. The base class defines general interfaces for training and testing, and each model subclass inherits and implements its own unique training and testing logic.<br>
- `main.py`：The program entry, which coordinates the complete execution logic of model loading, data processing, training and testing processes, and result evaluation.<br>
- 'pot.py':The module for anomaly detection and evaluation index determination.<br>
Parameter constant modules include:<br>
- `constants.py`:  The quantile level for initially determining the threshold for specific datasets, and the scaling factor for finally scaling the threshold calculated by SPOT.<br>
- `folderconstants.py`: The file names where the loaded data is stored<br>
- `parser.py`: Command-line parameters.<br>

Auxiliary tool modules include:
- `Tranutils.py`: Stores Transformer components and other exclusive architecture implementations required by models such as TranAD and DTAAD.
- `utils.py`: Provides support for general tool functions.
- `plotting`.py: Responsible for visualization functions, generating training curves, prediction results, and other charts.
- `preprocessd.py`: The data preprocessing module.

### 五、 Core Modules and Processes

#### 1.  Data Layer: Loading and Preprocessing
- **Data Source**： The dataset is loaded through load_dataset(args.dataset), including the training set (train_loader), test set (test_loader), and anomaly labels (labels). Raw data is stored in the data folder, and preprocessed data is stored in the processed_data folder.<br>
    - Detailed data loading process (SMD)：<br>
      Load the dataset, train each machine's data separately, uniformly load the training set, test set, and label set into a loader according to the dataset name passed through the command line, and then load train_loader and test_loader from the loader. For example, for the first machine, load machine-1-1-train.npy, machine-1-1.test.npy, and machine-1-1-labels into a data loader. loader[0], loader[1], and loader[2] correspond to machine-1-1-train.npy, machine-1-1.test.npy, and machine-1-1-labels respectively. Here, the batch_size for loading is specified as all data in one .npy file. For the SMD dataset, this means loading all measurement data of one machine at a time.<br>
      
- **Windowing Processing (Key Point):**:
For models that require input data processed through windowing:
Taking TranAD as an example, when testing and training on the SMD dataset's machine-1-1 (28479×38), set the window size to 10. The data in each window is (10×38), and the sliding window logic is "taking the current index i as the end, and intercepting w_size time steps forward". For the first w_size time steps, the missing data in the window is filled with the first element uniformly (this approach is also based on the idea of causal convolution, where the data at the last time step of the current window can only see historical data information).<br>

Original data: 28479 time steps, with 38 features per time step → shape (28479, 38)
Window: After sliding window processing, each window is a segment of 10 consecutive time steps → shape (10, 38).
Total number of samples: 28479 windows are generated through sliding the window.
Batch: During training, 128 windows are taken from 28479 samples each time → shape (128, 10, 38) (128 samples, each with 10 time steps).<br>


  The convert_to_windows function implements windowing processing:<br>
  
  For the first w_size time steps, fill the missing parts with the first element.
For TranAD or Attention models, maintain the two-dimensional shape of (10, 38) (retaining the structure of time steps and features) for stacking. For other models (if the window size is 5): The window is flattened into a one-dimensional tensor w.view(-1) with a shape of (5×38=190, concatenating 38 features of 5 time steps into 190 features) and then stacked. The purpose of splitting into such window data is for parallel computing. The following figure demonstrates this process and the differences. <br>
Note: The windows here are not directly divided as shown in the figure, but are split by sliding one step each time.<br>
<img width="1457" height="563" alt="image" src="https://github.com/user-attachments/assets/9745e440-7dcc-4482-909c-297d400220e2" />


#### 2. Model Layer: Support for Multiple Deep Learning Models
  The deep learning models currently integrated in the project are as follows. For detailed model introductions, please refer to the document Introduction to Time-Series Anomaly Detection Models.pdf.<br>

<<<<<<< HEAD
| 模型类型             | 核心逻辑                                                                                                                                      | 损失函数特点                                                                      |
=======
| Model             | Core Logic                                                                                                                                     | Loss Function                                                                     |
>>>>>>> 8825f4817ea27e3850b70a239c69495e8b33a896
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Autoencoder（Basic） | The simplest autoencoder model                                                                                                                | Mean Squared Error (MSE), directly measuring reconstruction error                 |
| TranAD               | Transformer-based anomaly detection model, capturing long time-series dependencies, outputting dual prediction results (dynamically weighted) |
| OmniAnomaly          | A variant of Variational Autoencoder (VAE), introducing KL divergence regularization                                                          | MSE (reconstruction loss) + β×KL divergence (regularization to avoid overfitting) |
| DTAAD                | Dual autoencoder model, with two autoencoders constraining each other, dual TCN architecture                                                  | with two autoencoders constraining each other, dual TCN architecture              |
#### 3.  Loading the Model
  If a pre-trained model is loaded, for example, if it was trained for 10 epochs last time, this time it will continue training from epoch 11.<br>
<<<<<<< HEAD

#### 4. 训练与推理流程
A base class for training and testing is designed, and different models inherit and rewrite the training and testing methods. As shown in the following figure, the base class.<br>

=======

#### 4. Train And Test
A base class for training and testing is designed, and different models inherit and rewrite the training and testing methods. As shown in the following figure, the base class.<br>

>>>>>>> 8825f4817ea27e3850b70a239c69495e8b33a896
    
<img width="1034" height="396" alt="image" src="https://github.com/user-attachments/assets/8549b5e0-2075-4005-9804-a78eb9ec1b16" /> 
    
TranAD subclass inheritance is as shown in the following figure:<br>
    
<img width="963" height="125" alt="image" src="https://github.com/user-attachments/assets/3cf7629a-ff35-4efb-be27-f2128ea81cc3" />

    
##### Training Mode (training=True):
- **Input**: Windowed training data, model, optimizer (AdamW with weight decay to prevent overfitting), learning rate scheduler (StepLR, decaying to 0.9 times every 5 epochs).
- **Process**: Calculate the loss according to the model type (such as MSE), update model parameters through backpropagation, and record the loss and learning rate.
- **Output**: The average loss of the current epoch and the current learning rate, used to monitor the training process.
Testing Mode (training=False):
- **Input**: Windowed test data, trained model.
- **Process**: Turn off gradient calculation, the model outputs prediction/reconstruction results, 循环计算每个特征的损失（38 times for the SMD dataset）, calculate the anomaly threshold using the POT algorithm, and evaluate. Average the losses of all features in the training set and test set for evaluation, and combine anomaly labels (mark as abnormal if any feature is abnormal).
Print the evaluation results of each feature and the comprehensive evaluation results.

##### 5. Model Saving and Loading
- **Saving (save_model)**: Save model parameters, optimizer state, scheduler state, training epochs, and loss records to the path checkpoints/model name_dataset name/, facilitating resuming training after interruption or direct use for testing.
- **Loading (load_model)**: Dynamically load the model class from src.models according to the model name. If a pre-trained model exists, load the parameters; otherwise, initialize a new model (supporting the retrain parameter to control whether to retrain).

#### 6.Anomaly Determination
**Specific POT Algorithm Process:：**
  For the detailed principle of the POT algorithm, please refer to
 
   [pot Original Paper](https://dl.acm.org/doi/10.1145/3097983.3098144)<br>

 The purpose of the pot (Peak Over Threshold) algorithm is to automatically select the threshold for judging anomalies through all anomaly scores of the training set without manually specifying the threshold. The specific steps are: first, select the 98th percentile of the anomaly scores as the initial threshold. Anomaly scores exceeding 98% of the data (i.e., values with large reconstruction errors) are a small number of peaks. These peaks approximately follow a Generalized Pareto Distribution (the distribution of extreme values is almost independent of the distribution of the original data, similar to the Central Limit Theorem). The Pareto curve is used to fit these peak data (calculating the two parameters of the GDF). Maximum Likelihood Estimation (MLE) is used to estimate parameters gamma and beta, and the final threshold for judging anomalies is determined through a threshold calculation formula.<br>
  <img width="352" height="162" alt="5baa76da8e085671f1ea8d397d1f9d6" src="https://github.com/user-attachments/assets/5c021367-f3e5-4806-8d22-acda6042544f" />
<br>
The following figure shows the adjustment of the pot algorithm, referring to the paper<br>
Abdulaal, Ahmed et al. "Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization.",ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2021)
  <img width="1314" height="782" alt="image" src="https://github.com/user-attachments/assets/4cb61443-e8f9-43be-8494-475ebd28ace0" />

**SPOT Algorithm**<br>
The spot algorithm applies the pot algorithm to streaming data. It first estimates the first n values using the pot algorithm to obtain the initial anomaly threshold. For subsequent data, it can perform anomaly labeling or update the threshold. If the observed data exceeds the threshold, it is regarded as an anomaly, and the anomaly value is not used to update the threshold. However, the spot algorithm requires the data to be relatively stable, with no obvious shift in the value distribution.

**Judging Anomalies** 

<<<<<<< HEAD
  This threshold is used to judge the anomaly scores of the test dataset. Time points with anomaly scores greater than the threshold are marked as anomalies, forming an anomaly  pred. Then, pred is compared with label to calculate TP and TN (where TP refers to time points where both pred and label are 1, i.e., actual anomalies that are detected as anomalies), and further calculate the F1 score. Detecting time-series anomalies requires correcting the anomaly 标记序列 pred through a function method (adjust_predicts()). If the data at the current time point of the test set is determined to be an anomaly (True) and the corresponding real label is also an anomaly (True) but has not been marked as an anomaly segment, trace back the real label position and modify the corresponding judgment position in pred to True (anomaly). Finally, the function returns the corrected pred, and then the general F1 calculation is performed. The number of anomaly segments is calculated to calculate the average detection delay (total delay / number of anomaly segments), and the delay is counted on a per-anomaly-segment basis. Delay refers to the number of time steps from the start of the anomaly to when the algorithm detects the anomaly.<br>
=======
  This threshold is used to judge the anomaly scores of the test dataset. Time points with anomaly scores greater than the threshold are marked as anomalies, forming an anomaly  pred. Then, pred is compared with label to calculate TP and TN (where TP refers to time points where both pred and label are 1, i.e., actual anomalies that are detected as anomalies), and further calculate the F1 score. Detecting time-series anomalies requires correcting the anomaly  pred through a function method (adjust_predicts()). If the data at the current time point of the test set is determined to be an anomaly (True) and the corresponding real label is also an anomaly (True) but has not been marked as an anomaly segment, trace back the real label position and modify the corresponding judgment position in pred to True (anomaly). Finally, the function returns the corrected pred, and then the general F1 calculation is performed. The number of anomaly segments is calculated to calculate the average detection delay (total delay / number of anomaly segments), and the delay is counted on a per-anomaly-segment basis. Delay refers to the number of time steps from the start of the anomaly to when the algorithm detects the anomaly.<br>
>>>>>>> 8825f4817ea27e3850b70a239c69495e8b33a896
  
he following figure demonstrates the process of correcting predict. Note that in the actual actual and pred lists, False and True are stored instead of 0 and 1. The following figure is only for ease of representation.<br>
- <img width="1043" height="727" alt="image" src="https://github.com/user-attachments/assets/fd2beddf-6e4e-4350-a4b5-56834ab24253" />


#### 7.Detection Indicators
Explanation of the detection indicators used in the project


| Metric Name | Metric Meaning                                                                                                                    | Simplified Calculation Formula                                                                                        | Core Function                                                                                                                                                                             |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TP          | Number of samples predicted as "anomalous" by the model and actually "anomalous" (true anomalies correctly identified)            | $TP = $ Number of samples in the intersection of predicted anomalies and actual anomalies                             | Measures the model's basic ability to "catch anomalies" and is the foundation for metrics like recall and F1                                                                              |
| TN          | Number of samples predicted as "normal" by the model and actually "normal" (true normals correctly identified)                    | $TN = $ Number of samples in the intersection of predicted normals and actual normals                                 | Measures the model's basic ability to "recognize normals" and reflects the accuracy in distinguishing normal samples                                                                      |
| FP          | Number of samples predicted as "anomalous" by the model but actually "normal" (normal samples falsely judged as anomalies)        | $FP = $ Number of samples in the intersection of predicted anomalies and actual normals                               | Reflects the "false positive" situation; more false positives indicate the model is more sensitive to interference from normal samples                                                    |
| FN          | Number of samples predicted as "normal" by the model but actually "anomalous" (anomalous samples missed as normal)                | $FN = $ Number of samples in the intersection of predicted normals and actual anomalies                               | Reflects the "missed detection" situation; more missed detections indicate the model has poorer coverage of anomalous samples                                                             |
| precision   | Proportion of samples predicted as "anomalous" that are actually "anomalous"                                                      | $precision = \frac{TP}{TP + FP}$                                                                                      | Measures "how many of the predicted anomalous samples are truly anomalous," focusing on **reducing false positives**                                                                      |
| recall      | Proportion of actually "anomalous" samples that are predicted as "anomalous" by the model                                         | $recall = \frac{TP}{TP + FN}$                                                                                         | Measures "how many of the truly anomalous samples are found by the model," focusing on **reducing missed detections**                                                                     |
| F1          | Harmonic mean of precision and recall, balancing the comprehensive performance of "no false positives" and "no missed detections" | $f1 = \frac{2 \times precision \times recall}{precision + recall}$                                                    | Comprehensively evaluates the model's ability to balance "false positives" and "missed detections," suitable for overall performance comparison                                           |
<<<<<<< HEAD
| ROC/AUC     | Area under the ROC curve (ROC curve is plotted based on TPR and FPR under different thresholds)                                   | Calculated based on the area of the curve of TPR (recall) and FPR (false positive rate) under all possible thresholds | Measures the model's **ability to distinguish between anomalous and normal samples**, independent of the threshold; the closer the value is to 1, the stronger the discrimination ability |
=======
| ROC/AUC     | Area under the ROC curve (ROC curve is plotted based on TPR and FPR under different thresholds)                                   | Calculated based on the area of the curve of TPR (recall) and FPR (false positive rate) under all possible thresholds | Measures the model's **ability to distinguish between anomalous and normal samples**, independent of the threshold; the closer the value is to 1, the stronger the discrimination ability |
>>>>>>> 8825f4817ea27e3850b70a239c69495e8b33a896
