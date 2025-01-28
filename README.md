**Vehicle Detection and Classification System**  

**Project Objective:**  
Design and implement a machine learning-based system to detect, count, and classify vehicles in real time, aiding in effective traffic management, congestion control, and urban planning.  

**Key Contributions:**  
- **Data Analysis and Preprocessing:**  
   - Collected and annotated traffic video datasets with labels for cars, trucks, buses, and motorcycles.  
   - Addressed challenges like occlusions, poor lighting, and adverse weather conditions using augmentation techniques.  
   - Extracted relevant spatial and object-specific features using convolutional layers.  

- **Machine Learning Models:**  
   - **YOLO (You Only Look Once):** Applied for fast, accurate real-time vehicle detection and classification.  
   - **Support Vector Machine (SVM):** Used for refining classification accuracy in cases of overlapping vehicle types.  
   - **XGBoost:** Implemented for analysis and validation of traffic data to enhance model reliability.  

- **Model Optimization:**  
   - Tuned YOLO hyperparameters to boost detection speed and accuracy on real-time feeds.  
   - Optimized for edge devices like NVIDIA Jetson Nano to enable lightweight and real-time performance.  

**Achievements:**  
- Attained high detection accuracy with optimized YOLO and advanced feature selection techniques.  
- Extracted critical traffic metrics such as vehicle density, type-wise counts, and peak traffic timings.  
- Demonstrated robustness in diverse traffic conditions, including highways and intersections.  

**Tools & Technologies:**  
- **Programming:** Python  
- **Libraries:** Darknet, OpenCV, TensorFlow, Pandas  
- **Platforms:** Google Colab, Jupyter Notebook, NVIDIA Jetson Nano  

**Future Scope:**  
- Integrate with IoT for adaptive traffic control systems.  
- Expand to multi-camera configurations for broader coverage.  
- Scale to handle autonomous vehicle data for smart city applications.  

**Features:**  
- Real-time detection and classification of vehicles.  
- Multi-type vehicle classification with annotation output.  
- Generates an output video with bounding boxes and labels for each vehicle.  

This system demonstrates expertise in real-time machine learning applications to solve traffic challenges effectively.  
