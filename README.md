# Crop-Disease-Prediction-InceptionV3-and-MobilNetV2

Dataset Description
In this world of modern agriculture, early and accurate detection of crop diseases can make the difference
between a thriving yield and a failed harvest. Our task dives deep into this critical challenge through a rich and
diverse plant disease classification dataset spanning five major crops—Corn, Potato, Rice, Wheat, and
Sugarcane.
The dataset comprises 13,324 high-resolution images categorized into 17 distinct classes, where each
class represents either a healthy crop condition or a specific disease. Henceforth, this dataset captures the
real-world complexities of plant pathology. Some examples include:
● Corn: Common Rust, Gray Leaf Spot, Northern Leaf Blight, and Healthy.
● Potato: Early Blight, Late Blight, and Healthy.
● Rice: Brown Spot, Leaf Blast, Neck Blast, and Healthy.
● Wheat: Brown Rust, Yellow Rust, and Healthy.
● Sugarcane: Red Rot, Bacterial Blight, and Healthy
The dataset comprises leaf images of five major crops—corn, potato, rice, wheat, and sugarcane—collected
from reputable open-source datasets like PlantVillage, Dhan-Shomadhan, and Kaggle. It captures both healthy
and diseased conditions. With real-world variations in lighting, disease severity, and crop stages, the dataset
presents a challenging and realistic classification problem that closely resembles what farmers or field experts
encounter on-site.
Modeling the Problem
This problem is framed as a multi-class image classification task, where the input is a single RGB image of
a leaf, and the output is a prediction over one of the 17 possible classes.
What makes this task challenging is:
● High inter-class similarity: Different diseases across crops can exhibit visually similar symptoms.
● Intra-class variation: The same disease can appear different due to stage of infection, lighting, or leaf
orientation.
● Class imbalance: Particularly in Sugarcane classes, where image count is relatively low.
To tackle this, we primarily focused on InceptionV3, a deep and powerful CNN known for its high accuracy
and ability to capture fine-grained visual details—making it ideal for our complex, multi-class dataset.
Alongside it, we also explored MobileNetV2 as a lightweight alternative, valued for its speed and efficiency.
While InceptionV3 remains our top choice for deployment due to its superior performance, MobileNetV2 serves
as a strong backup option, especially for resource-constrained scenarios or mobile applications.
Introduction to Inception and MobileNet Models
Inception (GoogLeNet)
Inception, also known as GoogLeNet, is a deep convolutional neural network architecture developed by
Google. It is famous for its inception modules, which are designed to capture information at various scales and
improve computational efficiency. Inception was developed to address the challenge of training very deep
neural networks while maintaining computational efficiency and interpretability of feature maps.
Architecture:
● Inception Modules: Inception networks use multiple parallel convolutional layers of different filter sizes
(1x1, 3x3, 5x5) to capture features at different scales within the same layer.
● Reduction Units: These units are used to reduce the dimensionality of the input volume before feeding
it into the deeper layers, thus reducing computational complexity.
● Global Average Pooling: Instead of fully connected layers at the end, Inception networks typically use
global average pooling, which reduces overfitting and improves efficiency.
Motivation for Usage: Inception's architecture allows it to capture intricate details and features across
different scales, which is beneficial for complex datasets like ours with diverse plant disease images. Its ability
to efficiently handle large volumes of data and maintain strong performance makes it a suitable choice for
accurate classification tasks.
MobileNet
MobileNet is a lightweight convolutional neural network (CNN) designed for mobile and embedded vision
applications. It was developed by Google researchers to provide efficient and low-latency solutions for tasks
like image classification, object detection, and more. MobileNet was developed with the goal of enabling
efficient deep learning models on mobile and embedded devices, where computational resources such as
memory and processing power are limited.
Architecture:
● Depthwise Separable Convolution: MobileNet utilizes depthwise separable convolutions, which
separate the spatial filtering and the channel-wise filtering into two distinct operations. This significantly
reduces computational cost and model size compared to traditional convolutions.
● Efficiency: By using fewer parameters and computations, MobileNet achieves a good balance
between accuracy and efficiency, making it ideal for deployment on resource-constrained devices.
Motivation for Usage: MobileNet's architecture is tailored to be lightweight and fast, making it suitable for
real-time applications and scenarios where model size and inference speed are critical factors. For your
dataset with over 13,000 images across 17 classes, MobileNet's efficiency ensures that inference can be
performed swiftly even on devices with limited hardware capabilities.

Inception Model Summary:
InceptionV3 is a deeper and more complex model with over 25 million parameters. It starts with multiple
convolutional layers and pooling, followed by Inception modules (A to E) that perform multiple convolutions in
parallel (1x1, 3x3, 5x5) to capture features at various scales.
The model ends with global average pooling and a dense layer to classify images. Despite its large size
(~231MB), InceptionV3 is highly effective at handling diverse and complex datasets like ours due to its ability to
learn rich hierarchical features.
Model Performance and Evaluation:
Training Accuracy : 0.9724 Validation Accuracy : 0.8062 Train Loss : 0.0885


MobileNet Model Summary:
The MobileNetV2 model we used consists of around 2.24 million parameters and follows a streamlined
architecture optimized for speed and efficiency. It begins with a basic convolutional block, followed by a series
of Inverted Residual blocks—a key feature of MobileNetV2. These blocks use depthwise separable
convolutions, which reduce computation while preserving accuracy.
The model progressively increases channel depth while reducing spatial resolution, enabling it to learn rich
features in a compact form. At the end, it applies global average pooling, dropout, and a final dense layer to
classify the image into one of 17 plant disease categories. The total size of the model is just ~116MB, making it
ideal for low-resource environments.
Model Performance and Evaluation:
Training Accuracy : 0.9820 Validation Accuracy : 0.8007 Train Loss : 0.0505

Challenges Faced During Model Training and Evaluation
1. High Computational Overhead from Data Transformations
○ A significant portion of GPU time was underutilized.
○ Most of the training time was consumed by image transformations and data preprocessing
(especially class-wise transformations and augmentations).
○ This led to inefficiencies in model training, as GPU utilization remained low during these phases.
2. Class Imbalance Handling Was Resource-Intensive
○ To balance the dataset, multiple augmentation techniques were used on underrepresented
classes.
○ This balancing process increased memory usage and added overhead during batch processing
and data loading.
3. Underperformance in Certain Classes
○ As seen in the classification report and confusion matrix, a few classes (e.g.,
Wheat___Yellow_Rust, Corn___Gray_Leaf_Spot) had very low or zero recall.
○ These classes were among those that required extensive transformation, possibly leading to
distortions or inadequate feature learning.
○ The similarity in visual patterns among some disease classes could also have confused the
model (e.g., between different types of leaf spots or blights).
4. Limited Training Epochs Due to Hardware Constraints
○ The model (InceptionV3) is computationally heavy and could not be trained beyond a few
epochs due to device (GPU/VRAM) limitations.
○ As a result, early stopping was applied and the best-performing model was selected from the
first three epochs.
5. General Class-Wise Performance Variability
○ While many classes showed high precision and recall, others had significant misclassification
rates.
○ The confusion matrix highlights that misclassified samples often fall into visually or contextually
similar classes.
○ This suggests a need for more discriminative features or better augmentation strategies for
those specific classes.
Optimization Techniques in Model Training
To improve model performance and ensure fair learning across all 17 classes, we applied several optimization
techniques during training. These addressed key challenges such as class imbalance, dataset diversity, and
training efficiency.
1. Data Balancing
The dataset exhibited significant class imbalance, with some classes containing over 1,000 images and others
as few as 100. To address this:
● We oversampled and undersampled classes to reach a uniform target size.
● A custom BalancedPlantDataset class was implemented, incorporating class-specific
augmentation strategies to boost the minority classes effectively without relying solely on duplication.
2. Class-Dependent Data Augmentation
Data augmentation was applied to enhance the dataset's variability and help the model generalize better. Our
approach included:
● Applying stronger augmentations (e.g., rotation, color jitter, affine transformations, and perspective
distortion) to underrepresented classes.
● Applying lighter augmentations (e.g., horizontal and vertical flips) to well-represented classes.
● Implementing a custom augmentation pipeline using torchvision transforms and PIL to control the
augmentation intensity based on class size.
This targeted augmentation strategy ensured that minority classes were sufficiently diversified during training,
reducing overfitting and improving overall model balance.
3. DataLoader Optimization
To enhance training efficiency, the data pipeline was optimized by:
● Setting pin_memory=True and num_workers=4 in the PyTorch DataLoader to enable faster batch
loading and better use of system resources.
● Enabling data shuffling during training to randomize input order and prevent the model from learning
sequence-based patterns.
These techniques collectively contributed to a more balanced and robust model capable of performing well
across all crop types and disease classes.
