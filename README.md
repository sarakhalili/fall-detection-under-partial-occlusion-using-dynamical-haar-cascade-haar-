# Fall Detection under partial occlusion Using Dynamical haar Cascade 
One of the most possible dangers that older people face in their daily lives is falling. A fall is an accident that results in disability and crippling. If the person is not able to inform others about their condition after a fall, this dilemma gets more critical, and the person may even lose their life; therefore, the existence of intelligent and efficient systems for detecting falls in the elderly is essential. As deep learning and computer vision have developed in recent years, researchers have used these techniques to detect falls. Among the computer vision challenges, occlusion is one of the biggest challenges of these systems and degrades their performance to a considerable extent. Here, we provide an effective solution for occlusion handling in vision-based fall detection systems. We propose a fall detection algorithm under partial occlusion using the dynamical haar cascade presented in [Fall-Detection_Dynamical_haar_Cascade](https://github.com/amrn9674/Fall-Detection_Dynamical_haar_Cascade) to tackle this problem.
## Main Idea
Occlusion can lead a model to learn unrealistic features or force the model to ignore the blocked spatial area in un-occluded samples. In other words, body characteristics and sources of occlusion simultaneously affect the distribution of features. As a result, occluded data inevitably has a negative effect on the extraction of normal data features; therefore, it is necessary to design an effective learning strategy to optimize the model. To this end, we present weighted model training by defining a new cost function. The new cost function is defined by L=Ln+(λ*(n/o)*Lo), where Ln is the classification error of normal samples and Lo is the classification error of occluded samples[1]. This framework can be applied to various fall detection systems. Here we use dynamical haar cascade for feature extraction.
The dynamical haar algorithm is based on Viola, Jones, and Snow's research[2] written by Mohammadzadeh et al. [3]. This study provides a temporal extension of [2] based on the differences between pairs of consecutive images to identify pedestrians. Hence this research focuses on activity classification; time is a more important factor in feature extraction. As a result, instead of using two consecutive frames, four consecutive frames are used.

## Pre-processing
The Whole number of features that can be extracted from a video grows exponentially as the size of its frames increases, so it is important to resize the videos to have a reasonable and sufficient number of features. Following Viola, Jones, and Snow's work, the proper size is adjusted to 20*15 Pixels. This task is done by “resize” code.
## Training Phase
After extracting the features by all possible motion and appearance filters, 700272 features are generated for every 4 consecutive frames. This part is done by "all feature extraction" code. The motion and appearance filters are defined in “haar features” code. In addition, the motion and appearance filters with their specific labels are defined in “haarfeature_with_clue” code. These codes are available in [Fall-Detection_Dynamical_haar_Cascade](https://github.com/amrn9674/Fall-Detection_Dynamical_haar_Cascade). The specific labels can be generated using the “make files with clue” code.
These 700272 features are entered into the AdaBoost algorithm, and the top 300 features that best separate falling from other activities are selected. Here we train the AdaBoost classifier with the weighted lost function. For finding the best features (=filters), we used “find_feature” code available at [Fall-Detection_Dynamical_haar_Cascade](https://github.com/amrn9674/Fall-Detection_Dynamical_haar_Cascade). After choosing the top features, a new model named “final_model” is defined which only contains the top filters. This model would be generated automatically in the training phase.  Then we extract features with the new model in “haar extractor” code. 
## Detection phase 
Finally, we use the SVM classifier to classify the extracted features. SVM is a supervised machine learning algorithm that is very popular due to its high accuracy. The classification task is done in “cross sample” code and again SVM is trained with weighted lost function.
### Any Questions?
If you had any questions about using this code, Please contact [Sara Khalili](sarahkhalili89@gmail.com)

### Refrences
[1]	C. Shao et al., "Biased Feature Learning for Occlusion Invariant Face Recognition," in IJCAI, 2020, pp. 666-672.

[2]	P. Viola, M. J. Jones, and D. Snow, "Detecting pedestrians using patterns of motion and appearance," International Journal of Computer Vision, vol. 63, no. 2, pp. 153-161, 2005.

[3]	H. M. Amirreza Razmjoo. (2018). Fall-Detection_Dynamical_haar_Cascade. Available: https://github.com/amrn9674/Fall-Detection_Dynamical_haar_Cascade/tree/master




