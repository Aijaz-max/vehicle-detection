# vehicle-detection


**Project Objective:**  
Develop a machine learning-based system to accurately detect, count, and classify vehicles in real time, aiding in traffic management, congestion reduction, and urban planning.  

**Key Contributions:**  
- **Data Analysis and Preprocessing:**  
   - Collected and preprocessed traffic video datasets, including annotations for vehicles such as cars, trucks, buses, and motorcycles.  
   - Handled challenges like occlusions, varying lighting conditions, and weather effects using augmentation techniques.  
   - Performed feature extraction using convolutional layers to capture spatial and object-specific details.  

- **Machine Learning Algorithms:**  
   - **YOLO (You Only Look Once):** Used for real-time object detection and classification due to its balance between speed and accuracy.  
   - **Support Vector Machine (SVM):** Integrated for post-classification refinement in scenarios with overlapping vehicle types.  
   - **XGBoost:** Utilized to analyze and validate traffic data, ensuring robust predictions under varied conditions.  

- **Model Optimization:**  
   - Fine-tuned hyperparameters of YOLO to improve performance on live video streams.  
   - Reduced computational overhead through edge device optimizations, enabling real-time inference.  

**Achievements:**  
- Achieved high detection accuracy by combining YOLO’s speed with feature selection techniques.  
- Identified critical traffic metrics like vehicle density, classification counts, and peak congestion times.  
- Enhanced system robustness for diverse traffic scenarios, including urban intersections and highways.  

**Tools & Technologies:**  
- **Programming:** Python  
- **Libraries:** Darknet, OpenCV, TensorFlow, Pandas  
- **Platforms:** NVIDIA Jetson Nano, Google Colab, Jupyter Notebook  

**Future Scope:**  
- Real-time integration with IoT sensors for adaptive traffic signal control.  
- Expansion to multi-camera systems for wider area coverage.  
- Scalability to handle autonomous vehicle data streams for smart city infrastructure.
- 
- ## Features
- Real-time vehicle detection and classification.
- Capable of detecting and classifying multiple vehicle types.
- Counts the vehicles passing a predefined line in the video.
- Outputs a new video with annotated bounding boxes and classification labels.
**Real-Time Vehicle Counting and Classification Using Machine Learning**  


This system showcases practical expertise in machine learning, computer vision, and real-time system deployment to address modern traffic challenges.  
