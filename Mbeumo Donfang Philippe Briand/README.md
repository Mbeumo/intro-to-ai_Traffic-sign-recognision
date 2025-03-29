# Traffic Sign Classification Experimentation

## Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify traffic signs into their respective categories. The goal was to achieve the highest possible accuracy while ensuring the model's training process was efficient and reliable.

## Experimentation Process

### 1. Initial Approach: Basic CNN Model
- **What I Tried**: 
  - Started with a simple CNN architecture with a few convolutional layers, max-pooling, and fully connected layers.
  - Trained the model directly on the original dataset without any data augmentation.
- **What Worked Well**:
  - The model was able to learn basic patterns in the dataset.
  - Training was relatively fast, and the loss decreased steadily.
- **What Didn’t Work Well**:
  - The accuracy plateaued at a relatively low value, indicating the model lacked the capacity to learn complex features.

---

### 2. Adding More Layers to the CNN
- **What I Tried**:
  - Increased the depth of the CNN by adding more convolutional layers and fully connected layers.
  - Used dropout and batch normalization to prevent overfitting and stabilize training.
- **What Worked Well**:
  - The deeper model showed better feature extraction capabilities.
  - Validation accuracy improved slightly compared to the initial model.
- **What Didn’t Work Well**:
  - The model started overfitting after a few epochs, as the dataset was relatively small.
  - Training time increased significantly.

---

### 3. Experimenting with Data Augmentation
- **What I Tried**:
  - Used `ImageDataGenerator` to augment the dataset with transformations such as shear, zoom, brightness adjustments, and horizontal flips.
  - Trained the model on the augmented dataset.
- **What I Noticed**:
  - The augmented dataset introduced impurities, making the data less representative of the original dataset.
  - The model struggled to learn from the augmented data, resulting in high loss values (e.g., 3, 5, 7) and very low accuracy.
  - The accuracy increased very slowly and inconsistently, indicating that the augmented data was hindering the learning process.
- **What Didn’t Work Well**:
  - Augmentation caused the model to learn less effectively, as the transformations distorted the data too much.

---

### 4. Final Approach: Training Without Data Augmentation
- **What I Tried**:
  - Removed data augmentation entirely and trained the model directly on the original dataset.
  - Optimized the CNN architecture by:
    - Adding more convolutional layers with smaller filters (`3x3`).
    - Using `GlobalAveragePooling2D` instead of `Flatten` to reduce overfitting.
    - Adjusting dropout rates and adding L2 regularization to prevent overfitting.
    - Using a learning rate scheduler (`ReduceLROnPlateau`) to dynamically adjust the learning rate.
- **What I Noticed**:
  - When training with 30 epochs, the validation loss started increasing after a certain point, even though the training loss continued to decrease. This indicated **overfitting**.
  - To address this, I implemented **early stopping** to halt training when the validation loss stopped improving. This reduced the risk of overfitting and saved training time.

- **Early Stopping Implementation**:
  ```python
    early_stopping = EarlyStopping(
        monitor="val_loss",  # Monitor validation loss
        patience=5,          # Stop if no improvement after 5 epochs
        restore_best_weights=True,  # Restore the best weights
        verbose=1
    )
- **What Worked Well**:
  - The model achieved significantly higher accuracy compared to previous attempts.
  - The loss decreased steadily, and the accuracy increased exponentially during training.
  - Early stopping ensured the model stopped training as soon as it stopped improving, preventing overfitting
  - The training process was faster and more stable.
- **What Didn’t Work Well**:
  - None. This approach yielded the best results.

---

## Key Findings
1. **Data Augmentation**: While data augmentation can be useful in some cases, it introduced impurities in this dataset, making it harder for the model to learn effectively.
2. **Model Depth**: Increasing the depth of the CNN improved feature extraction but required careful regularization to prevent overfitting.
3. **Learning Rate Scheduler**: Dynamically adjusting the learning rate helped the model converge more effectively.
4. **Simpler Data**: Training on the original dataset without augmentation yielded the best results, as the data was clean and representative of the problem.
5. **Early Stopping**:  Adding early stopping prevented overfitting by halting training when validation loss stopped improving.

---

## Final Model
The final model achieved the highest accuracy with the following characteristics:
- **Architecture**:
  - 3 convolutional blocks with batch normalization and dropout.
  - Global average pooling instead of flattening.
  - 2 Fully connected layers with L2 regularization.
- **Training**:
  - Used a learning rate scheduler to optimize convergence.
  - Added early stopping to prevent overfitting.
  - Trained directly on the original dataset without augmentation.
- **Performance**:
  - Rapid accuracy improvement during training.
  - Stable and consistent loss reduction.

---

## Conclusion
Through experimentation, I discovered that data augmentation was detrimental to this specific dataset, as it introduced impurities that hindered learning. By focusing on optimizing the CNN architecture and training directly on the original dataset, I achieved the best results. This highlights the importance of understanding the dataset and tailoring the approach accordingly.