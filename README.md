# Precision Oncology Detection System: Tumor Imaging with Deep Learning

## Project Overview

The **Precision Oncology Detection System** uses deep learning to classify brain tumors from MRI scans into one of four categories: **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**. By leveraging advanced convolutional neural networks (CNNs), specifically the **Xception** architecture, this system aims to provide automated tumor detection that can assist radiologists in making quicker, more accurate diagnoses.

The model is trained on a publicly available **Brain Tumor MRI Dataset** from Kaggle, and has been optimized for high accuracy and robust performance using techniques such as **early stopping**, **learning rate adjustment**, and **data augmentation**.

## Features

- **Brain Tumor Classification**: The system classifies MRI images into four classes: glioma, meningioma, pituitary tumor, and no tumor.
- **Xception Model**: The pre-trained Xception model is used as a base for feature extraction, enabling better transfer learning for MRI image classification.
- **Custom CNN Layers**: The architecture includes custom CNN layers, such as dropout and fully connected layers, to fine-tune the model.
- **Training and Evaluation**: The system includes comprehensive metrics to evaluate the model's performance, including accuracy, precision, recall, loss, and confusion matrix.
- **Image Augmentation**: Techniques such as brightness variation and rescaling improve model robustness by simulating real-world variances in MRI images.
- **Model Inference**: The trained model can predict tumor types for unseen MRI scans, with probability values indicating the confidence of the prediction.
- **Visualization**: Training/validation metrics are visualized over epochs, and confusion matrices are plotted to understand model performance across different tumor types.

## Technical Details

### Data Preprocessing

1. **Dataset**: The model is trained using the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, which contains labeled MRI images of various brain tumor types.
2. **Image Rescaling**: Each image is resized to 299x299 pixels to match the input size expected by the Xception model.
3. **Data Augmentation**: Images are rescaled using a factor of 1/255 to normalize pixel values. Brightness range is varied between 0.8 and 1.2 to simulate real-world inconsistencies.
4. **Image Generators**: The **ImageDataGenerator** is used to generate batches of augmented images for training, validation, and testing.

### Model Architecture

1. **Xception Base Model**: 
   - The Xception model is a deep convolutional network based on depthwise separable convolutions, which significantly reduces computational complexity and model size compared to traditional convolutional layers.
   - We use the pre-trained **Xception model** (from ImageNet) without the top classification layer (`include_top=False`) and with **max pooling** to reduce feature map dimensions.

2. **Custom Layers**:
   - **Flatten Layer**: Converts the output of the convolutional base into a 1D array, which can be passed to the fully connected layers.
   - **Dropout**: Dropout layers are added after fully connected layers with rates of 0.3 and 0.25 to prevent overfitting during training.
   - **Dense Layer**: A fully connected layer with 128 neurons and ReLU activation function to learn more complex features.
   - **Softmax Output Layer**: The final output layer uses **softmax activation** for multi-class classification, outputting the probability distribution over the 4 possible classes.

3. **Optimizer**: 
   - The model uses **Adamax**, a variant of the Adam optimizer, which is known to work well with sparse gradients and large datasets.
   - A learning rate of **0.001** is used to balance convergence speed and stability during training.

4. **Loss Function**: 
   - The model uses **categorical crossentropy** for multi-class classification, which is standard for multi-class problems where each sample belongs to one of several classes.

5. **Evaluation Metrics**:
   - **Accuracy**: Measures how many of the total predictions are correct.
   - **Precision**: The percentage of positive predictions that are actually correct, which helps to minimize false positives.
   - **Recall**: The percentage of actual positive instances that were correctly identified by the model, which helps to minimize false negatives.

### Model Training and Optimization

1. **Training**:
   - The model is trained for up to **30 epochs** with a batch size of 32 using the **ImageDataGenerator**. Training utilizes a **validation split** (50% of the testing data) to monitor overfitting.
   - **Early Stopping**: Stops training if the validation loss stops improving, which prevents overfitting.
   - **ReduceLROnPlateau**: Dynamically reduces the learning rate if the model’s performance plateaus, enabling finer adjustments during training.

2. **Visualizing Training**:
   - The model’s performance (accuracy, precision, recall, loss) is visualized over epochs. Best performing epochs for each metric are highlighted.

### Evaluation and Inference

- **Confusion Matrix**: A confusion matrix is generated to visualize the true vs. predicted labels, helping to identify misclassifications and assess the model's strengths and weaknesses.
- **Model Inference**: The system is capable of predicting the tumor type in unseen MRI images, displaying the predicted class alongside the probabilities for each class.
  
### Final Model Evaluation

1. **Training Accuracy**: 99.28%
2. **Validation Accuracy**: 98.02%
3. **Test Accuracy**: 97.71%

### Example Inference

To predict tumor types for new MRI images:

```python
predict("/path/to/mri/image.jpg")
```

This will display the input MRI image, the predicted tumor type, and the corresponding prediction probabilities.

### Model Saving and Loading

Once training is complete, the model's weights are saved to disk using:

```python
model.save_weights("xception_model.weights.h5")
```

This allows you to load the model later for inference or further fine-tuning.

## Conclusion

This system provides an automated approach to brain tumor classification, which can be leveraged by medical professionals for faster and more accurate diagnoses. With the use of advanced deep learning models like Xception, combined with effective data preprocessing and model optimization techniques, the system achieves high performance in detecting various brain tumor types from MRI scans.

