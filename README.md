# 🌾 Rice Leaf Disease Detection using Deep Learning

> An intelligent image classification system to automatically detect and classify rice leaf diseases using CNN and Transfer Learning.

---

## 📌 Project Overview

Rice crops are highly vulnerable to leaf diseases that significantly reduce yield and farmer income. Early detection is critical, but manual inspection is time-consuming and prone to human error.

This project builds a Deep Learning-based image classification model to automatically detect and classify rice leaf diseases from leaf images. Multiple architectures were implemented and compared to determine the most effective approach for small agricultural datasets.

---

## 🎯 Objectives

- Automate rice leaf disease detection  
- Compare Custom CNN and Transfer Learning approaches  
- Improve generalization on a small dataset  
- Reduce overfitting using augmentation and regularization  

---

## 📂 Dataset Details

- Total Images: 120  
- Image Size: 224 × 224  
- Classes: 3 disease categories  
- Train–Validation Split: 80% / 20%  

### Key Challenges
- Limited dataset size  
- Risk of overfitting  
- Similar visual patterns across disease classes  

---

## 🛠️ Tech Stack

- Python  
- TensorFlow  
- Keras  
- NumPy  
- Pandas  
- Matplotlib  

---

## 🧠 Model Architectures

### 1️⃣ Custom CNN (Baseline Model)

- Convolutional Layers with ReLU activation  
- MaxPooling layers  
- Dropout for regularization  
- Fully connected Dense layers  
- Softmax output layer (3 classes)  

This model served as the baseline for performance comparison.

---

### 2️⃣ CNN with Data Augmentation

To improve generalization and reduce overfitting, the following augmentation techniques were applied:

- Rotation  
- Zoom  
- Horizontal Flip  
- Rescaling  

This improved validation stability compared to the baseline CNN.

---

### 3️⃣ Transfer Learning using VGG16 (Best Performing Model)

- Pretrained VGG16 with ImageNet weights  
- Frozen convolutional base  
- Custom classification head  
- Dropout for regularization  

Transfer Learning achieved the best validation performance, demonstrating the effectiveness of pretrained models on small datasets.

---

## 📊 Key Insights

- Data augmentation improves robustness  
- Transfer Learning significantly boosts performance  
- Regularization is essential for small datasets  
- Model comparison helps identify the optimal architecture  

---

## 📈 Skills Demonstrated

- Deep Learning model development  
- CNN architecture design  
- Transfer Learning implementation  
- Overfitting handling techniques  
- Model evaluation and comparison  
- End-to-end ML workflow  

---

## 🚀 Future Improvements

- Increase dataset size  
- Fine-tune pretrained layers  
- Experiment with advanced architectures (ResNet, EfficientNet)  
- Deploy as a web or mobile application  

---

## 📌 Conclusion

This project demonstrates the practical application of Deep Learning in agriculture. By comparing multiple architectures and applying proper regularization strategies, the model achieves strong performance even with limited data.

It reflects hands-on experience in building, evaluating, and optimizing deep learning models for real-world problems.

---

## 🔖 Tags

`#DeepLearning` `#ComputerVision` `#CNN` `#TransferLearning`  
`#TensorFlow` `#Keras` `#ImageClassification`  
`#AgricultureAI` `#MachineLearningProject`
