# Prediction of Joint Angles for Human Elbow Motion Based on sEMG

1<sup>st</sup> Jinqiang Wang School of Engineering Qufu Normal University Rizhao, China wang jinqiang@163.com

2<sup>nd</sup> Dianguo Cao School of Engineering Qufu Normal University Rizhao, China caodg@qfnu.edu.cn

3<sup>rd</sup> Hongyu Liu School of Engineering Qufu Normal University Rizhao, China 1521129553@qq.com

4<sup>th</sup> Yuqiang Wu\* School of Engineering Qufu Normal University Rizhao, China wyq@qfnu.edu.cn

Abstract—In the field of intelligent rehabilitation, the primary reliance is on rehabilitation robots, which often struggle to continuously respond to a patient’s movement intentions in real-time and with accuracy. To improve the level of humanrobot interaction (HRI) and provide personalized rehabilitation exercises with active patient participation, this study developed a joint angle prediction model using surface electromyography (sEMG). Initially, six healthy subjects were selected, and devices for sEMG and angle acquisition were employed to synchronize the collection of sEMG and elbow joint angle data for each participant. Subsequently, the preprocessed data were input into the designed Transformer-based joint angle prediction model. Finally, the performance of the Transformer-based model was compared with that of Recurrent Neural Networks (RNN) and Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM) models. Experimental results indicate that the Transformer-based joint angle prediction model outperforms its counterparts, achieving a Mean Absolute Error (MAE) of less than 0.065, thereby suggesting its potential for future applications.

Index Terms—human-robot interaction (HRI), joint angle, surface Electromyography (sEMG), Transformer

## I. INTRODUCTION

With the rapid advancements in rehabilitation robotics, relying solely on either human or robotic rehabilitation has become insufficient to meet the growing demands of the healthcare industry. Human-robot interaction (HRI) and collaboration have thus emerged as critical directions in the development of rehabilitation robotics [1], [2].

Surface Electromyography (sEMG) is a physiological signal closely linked to human movement, capturing the electrical ac tivity of muscles in a non-invasive manner [3]. This technique eliminates the need for needle insertion or surgical procedures, making it easier to record electrical signals produced during muscle contraction and relaxation.

sEMG signals are ideally suited for use as control and feedback signals in rehabilitation exoskeleton robots [4]. In the context of upper limb exoskeleton systems, accurately predict ing joint angles from sEMG signals is essential. This capability allows exoskeleton robots to adapt their movements based on the user’s muscle activity, enabling more personalized and natural motion assistance. The objective of this study is to explore the prediction of joint angles for continuous elbow motion in the upper limb.

Currently, the primary techniques for estimating continuous joint motion intention include the biomechanical modeling approach and the regression modeling approach. The biomechanical modeling approach explains the mechanism of motion generation from a microscopic perspective. For instance, some researchers have utilized Hill-based muscle modeling (HMM) [5] combined with biomechanics to estimate the states of continuous motion joints. Jianda Han et al. [6] developed a continuous motion estimation method based on a muscle model that incorporates a variety of intricate physiological parameters, showcasing a notable parameter fitting methodology. On the other hand, machine learning approaches have gained attention. He Li et al. [7] introduced a feature selection method based on a two-stage genetic algorithm (GA) to develop a multi-user shared model. Experimental results for this shared model show that the prediction error remains within 10°. In a recent study, Feiyun Xiao and colleagues proposed that the multitemporal delayed feature (MTDF) of sEMG signals can effectively predict joint angle information [8]. When combined with a random forest (RF) model, this system can rapidly estimate joint states during continuous motion, demonstrating high robustness and real-time performance.

With the advancement of deep learning technology, the Transformer model has been widely adopted in natural language processing (NLP) tasks, such as machine translation, text generation, and text categorization [9]. In this study, the Transformer model is introduced for the first time into myoelectric joint angle estimation, leading to the development of a Transformer-based model for the continuous estimation of joint angles using myoelectric signals.

The primary contributions of this study are outlined as follows:

A Transformer-based model for joint angle prediction was developed utilizing multichannel surface electromyography (sEMG) signals and historical joint angle data as input features.

Two machine learning models, Recurrent Neural Network (RNN) [10] and Convolutional Neural Network-Long Short-Term Memory (CNN-LSTM) [11], were established and compared with the Transformer prediction model. The experimental results demonstrate that the Transformer model exhibits superior performance in prediction.

The remainder of this paper is structured as follows. Section II details the data acquisition process, data preprocessing methods, prediction model architecture, and model performance evaluation criteria. Section III presents the joint angle prediction experiments. Section IV provides the conclusion.

## II. METHODOLOGY

## A. Data Acquisition

In the study, six healthy individuals (four males and two females) aged 22 to 29 were recruited. All participants were confirmed to be in good health. Before initiating the experiment, the procedures were comprehensively explained to the participants, and informed consent was obtained from each individual.

In joint angle prediction, the precise selection of motionrelated muscles can mitigate informational redundancy, thereby enhancing prediction accuracy. This experiment fo cused on elbow flexion and extension to assess upper limb movements. The primary muscles implicated in elbow flexion are the biceps brachii and brachioradialis, whereas the triceps brachii and anconeus are involved in elbow extension. The muscle selection process is illustrated in Fig.1, panel a.

The sEMG signals from four channels for each subject were collected using a Delsys wireless sEMG device with a sampling rate of 2000 Hz. Joint angle data was acquired using an angular sensor (JY-ME02-CAN) with a sampling rate of 200 Hz. Before electrode attachment, the skin was shaved to minimize the potential for interference with the sEMG signal.

Before each data collection session, sufficient rest periods were provided to prevent abnormal sEMG signals caused by muscle fatigue. Each session lasted 18 seconds, during which the subjects performed as many elbow movements as possible without experiencing fatigue. Each subject completed 20 data collection sessions. To ensure that the upper limb movements were restricted to a single degree of freedom in the vertical plane, subjects were instructed to keep their wrists aligned with their forearms throughout the experiment.

During the experiment, the sEMG signals were transmitted directly to the local script via the socket protocol, while the angle information was transferred to the same script via the serial port. This approach enabled the simultaneous acquisition of sEMG signals and angle information. The script was developed in Visual Studio 2021 using C#. The experimental setup for acquiring sEMG signals and angular data is shown in Fig. 1, panel b.

## B. Data Preprocessing

The raw sEMG signal is a particularly delicate and susceptible biological signal, making data preprocessing a crucial step in obtaining high-quality signals. Initially, the acquired sEMG signals were filtered to remove industrial frequency interference using a 60 Hz notch filter. Subsequently, the data underwent band-pass filtering with a frequency range of 20 to 450 Hz, considering the effective bandwidth of sEMG signals lies between 0 and 500 Hz, with most of the spectrum distributed between 10 and 250 Hz [12].

![](images/f51b3490da89c5ed443380473f9cece12ea6f3ec0000bc8d15554842c3f33070.jpg)  
Fig. 1. Upper Limb Muscle Selection and Data Acquisition Experimental Site.

To align the angle data with the sEMG signal, the sEMG data was downsampled to 200 Hz by calculating the rootmean-square (RMS). The RMS for each sEMG channel is computed as follows:

$$
R M S (t) = \sqrt {\frac {1}{N} \sum_ {n = t - N + 1} ^ {t} x [ n ] ^ {2}}\tag{1}
$$

where N = 10 represents the number of samples within the time window, and x[n] is the sEMG signal at sample point n.

During the training phase, the sEMG signal features from the four channels, along with the corresponding angle data, were used as inputs to train the model. The dataset samples are structured as follows:

$$
I n p u t (t) = [ R M S _ {1} (t), R M S _ {2} (t), R M S _ {3} (t), R M S _ {4} (t), \theta (t) ]\tag{2}
$$

In the testing phase, the system’s input consists solely of the sEMG signal features from the four channels.

## C. Prediction Model Structure

The Transformer neural network architecture holds significant importance, particularly in the fields of computer vision (CV) and natural language processing (NLP) [13]. The model notably enhances training efficiency and effectiveness by processing data in parallel. As depicted in Fig.2, it comprises the following four fundamental components:

Embedding layer [14]: The input embedding layer at the beginning of the Transformer model converts the input sequence into a high-dimensional vector space, allowing the model to better capture key information from the sequence. Typically, the Transformer adds positional encodings, generated using sine and cosine functions, directly to the input embeddings. This enables the model to effectively pinpoint the location of crucial information within the sequence, overcoming the limitations of traditional neural networks in processing time series data. In this study, both the Input embedding layer and the Position encoding layer have dimensions of 5 for the input and 512 for the output.

![](images/d94131da8c8ebe64f94f3639928bc092c63e4b43842c8e65f0e740b44f035cdf.jpg)  
Fig. 2. Transformer Model Architecture.

Encoder network: The Transformer encoder network comprises two sub-layers: a multi-head attention layer and a feed forward layer [15]. Each sub-layer produces its output via a feedforward network followed by layer normalization. Typically, multiple encoder network are utilized in series within the Transformer architecture, in this study, three encoder network are employed. In the initial encoding layer of the Transformer model, the Query, Key, and Value vectors necessary for the multi-head attention mechanism are derived from the Positional Embedding Layer. At this stage, the input sequences are first transformed into word embedding vectors, followed by the addition of positional information via positional encoding [16]. In subsequent encoder network, the input to the multihead attention sublayer is sourced from the output produced by the preceding encoder network, rather than directly from the embedding Layer. Consequently, each encoder network recomputes the query, key, and value vectors based on the previous layer’s output, thereby progressively refining and abstracting the input information.

Decoder network: Parallel to the encoder network, the decoder network have an equivalent number of layers. Each decoder network comprises three sub-layers: a masked multihead attention sub-layer, an encoder-decoder multi-head attention sub-layer, and a feedforward sub-layer. Decoder network also uses feedforward and layer normalization techniques in each sublayer to produce the output, which is similar to the encoder network’s processing. However, the decoder network adds a masking step to the processing of the self-attention mechanism. In the multi-headed self-attention sublayer of this masking, the Transformer masks future input embeddings, preventing the model from accessing future information in advance. The purpose of this is to ensure that the model, at each current location, makes decisions and predictions based only on the output information it has previously generated, and does not see future information in advance [17].

![](images/b9d9becf5613b165e21ac17c9367c62e0c74a37e8c09f78d7dd54121673b4b0f.jpg)  
Fig. 3. Attention Mechanism in Transformer

Softmax layer: In alignment with other neural machine translation frameworks, the decoder network’s output is subjected to a linear transformation prior to being fed into the Softmax layer. This layer then transforms the logistic scores into a probability distribution [18].

The self-attention mechanism is a pivotal element of the Transformer architecture, utilized in both the encoder network and decoder network [13]. Its main role is to learn the global temporal dependencies of the sEMG features, helping to compensate for the global information that the convolutional module is unable to capture due to its limited sensory field. Specifically, in the Transformer model, the mechanism known as multi-head attention is used, which is actually an extension of the scaled dot-product attention mechanism. The detailed operational principles of scaled dot-product attention and multi-head attention are illustrated in Fig.3.

Scaled dot-product attention involves linearly transforming the output of the embedding or encoding layer into three identically shaped matrices: query (Q), key (K) and value (V), each with a dimension $d _ { k }$ . These matrices are calculated as follows:

$$
Q = X W _ {Q}, \quad K = X W _ {K}, \quad V = X W _ {V}\tag{3}
$$

where X is the output from the previous layer, with dimensions batch size sequence length $\times \ d _ { k }$ , which in this study is $3 2 \times 1 0 \times 5 1 2$ . The matrices $W _ { Q } , \ W _ { K }$ , and $W _ { V }$ are the linear transformation matrices for the query, key, and value, respectively.

Next, the dot product of the query Q and key K is calculated and then divided by $d _ { k }$ to prevent the values from becoming too large. The attention weights are finally obtained using the softmax function:

$$
\text { Attention } (Q, K, V) = \operatorname{softmax} \left(\frac {Q K ^ {\mathrm{T}}}{\sqrt {d _ {k}}}\right) V\tag{4}
$$

Here, $Q K ^ { T }$ generates an attention matrix, which measures the similarity between each query and key. The softmax function normalizes this matrix, and the resulting weight matrix is applied to the value V to produce the final output. This mecha nism enables the model to capture long-distance dependencies by considering all positions in the input sequence when computing the output at each position [19]. This capability is a key reason why the Transformer model can effectively manage time series tasks.

The multi-head attention mechanism is a core innovation of the Transformer architecture, enhancing the model’s ability to capture features in different subspaces. By incorporating multiple attention heads, this mechanism performs attention computations in parallel across various representation spaces, capturing more complex dependencies and contextual information in sequences [20]. The computational process of the multi-head attention mechanism can be expressed as follows:

$$
\operatorname{MultiHead} (Q, K, V) = \operatorname{Concat} \left(\operatorname{head} _ {1}, \dots , \operatorname{head} _ {h}\right) W ^ {O}\tag{5}
$$

where each head is calculated as:

$$
\mathrm{head} _ {i} = \mathrm{Attention} (Q W _ {i} ^ {Q}, K W _ {i} ^ {K}, V W _ {i} ^ {V})\tag{6}
$$

Here $W _ { i } ^ { Q } , W _ { i } ^ { K }$ and $W _ { i } ^ { V }$ are the linear transformation matrices for the query, key and value corresponding to the i th header, respectively. $W ^ { \dot { O } }$ is the linear transformation matrix used to project the spliced result back to the desired dimension [21].

## D. Model Performance Measurement

In the experiments, two evaluation metrics, Mean Absolute Error (MAE) and Standard Deviation (STD), were used to assess the effectiveness of the prediction system [22]. During the model training phase, five-fold cross-validation was performed to determine the optimal model parameters. In this process, 15 data sets were used for training with cross-validation, while the remaining 5 data sets were reserved for testing. Subsequently, the MAE between the predicted and actual elbow joint angles was calculated to evaluate the model’s accuracy. Finally, the STD was computed for each subject across multiple trials.

The formula for MAE is:

$$
\mathrm{MAE} = \frac {1}{M} \sum_ {i = 1} ^ {M} | y _ {i} - \hat {y} _ {i} |\tag{7}
$$

where $M$ is the number of downsampled points, $y _ { i }$ is the actual value, and $\hat { y } _ { i }$ is the predicted value. The formula for Standard Deviation (STD) is:

$$
\mathrm{STD} = \sqrt {\frac {1}{S} \sum_ {i = 1} ^ {S} (m _ {i} - \mu) ^ {2}}\tag{8}
$$

Here, $m _ { i }$ represents the MAE value for each trial of a subject, $\mu$ is the average MAE across 5 test trials for each subject, and $S$ is the total number of test trials, which is 5.

## III. EXPERIMENT RESULTS

The model training for the experiment was conducted on a local server equipped with an RTX3090 graphics processor, using Python 3.7 and PyTorch 1.12. The primary training parameters were set as follows: epoch=300, batch size=64, and learning rate=0.0001. In the Transformer prediction model, the number of Encoder and Decoder layers was set to 3 each, and the number of heads in the multi-head attention mechanism was set to 8. For each subject, 70% of the data was used as the training and validation dataset, while the remaining 30% was used as the testing dataset. The prediction results for each subject are illustrated in Fig. 4, and the MAEs for all prediction results are presented in Table I.

This work also establishes two additional prediction models: an RNN and a CNN-LSTM. These models are evaluated in comparison to the Transformer model. The comparative analysis of the three models predicting the joint angle is presented in Fig. 5, with subject A serving as an illustrative example. The MAE between the true and predicted values for all subjects is statistically depicted in Table I.

As shown in Fig. 4, the Transformer model demonstrates a high level of accuracy in predicting the elbow angles of the six subjects. Additionally, Fig. 5 highlights the Transformer model’s superior accuracy in predicting continuous joint angles compared to the RNN and CNN-LSTM models, a conclusion further supported by the data in Table I. Moreover, a preliminary comparison of runtime and resource consumption reveals that while the RNN model has the longest runtime, it consumes fewer resources, making it more suitable for shortsequence tasks. CNN-LSTM offers faster runtime but at the cost of higher resource consumption, making it more effective for complex time-series tasks. The Transformer model, which processes entire sequences in parallel, achieves the shortest runtime and excels in long-sequence tasks, though it has the highest resource consumption.

## IV. CONCLUSION

To achieve the goal of predicting the elbow joint angle, this study initially designed sEMG and joint angle data acquisition experiments and completed the processes of data acquisition and preprocessing. Subsequently, a joint angle prediction model based on the Transformer architecture was developed, utilizing multi-channel sEMG signals and time series angles as inputs. Experiments comparing the joint angle prediction performance of the Transformer’s model with that of RNN and CNN-LSTM models were conducted. The experimental results indicate that the MAE between the predicted and true values of the Transformer-based model is less than 0.065, which is lower than the errors observed in the RNN and CNN-LSTM models, thereby demonstrating the efficacy of our approach.

The proposed model for predicting elbow joint motion angles shows improved accuracy in forecasting the user’s joint motion, offering new insights for control algorithms and rehabilitation robot control. However, further research is required to explore several promising directions. Future efforts could focus on optimizing the model architecture to enhance prediction accuracy and efficiency, expanding its application to other joints, and integrating additional biosignals for mul timodal data fusion. These advancements would broaden the scope of research and contribute to more advanced control algorithms for rehabilitation devices.

MEAN ABSOLUTE ERROR (MAE) AND STANDARD DEVIATION (STD) BETWEEN PREDICTED AND ACTUAL ELBOW ANGLES FOR THREE MODELS ACROSS SIX SUBJECTS  
TABLE I

<table><tr><td>model</td><td>evaluation metrics</td><td>Subject A</td><td>Subject B</td><td>Subject C</td><td>Subject D</td><td>Subject E</td><td>Subject F</td></tr><tr><td rowspan="2">RNN</td><td>MAE</td><td>0.0795</td><td>0.1018</td><td>0.0678</td><td>0.0975</td><td>0.0594</td><td>0.0922</td></tr><tr><td>STD</td><td>0.0080</td><td>0.0110</td><td>0.0090</td><td>0.0070</td><td>0.0081</td><td>0.0102</td></tr><tr><td rowspan="2">CNN-LSTM</td><td>MAE</td><td>0.0635</td><td>0.0863</td><td>0.0408</td><td>0.0729</td><td>0.0502</td><td>0.0750</td></tr><tr><td>STD</td><td>0.0075</td><td>0.0090</td><td>0.0107</td><td>0.0140</td><td>0.0095</td><td>0.0131</td></tr><tr><td rowspan="2">Transformer</td><td>MAE</td><td>0.0403</td><td>0.0600</td><td>0.0377</td><td>0.0539</td><td>0.0274</td><td>0.0484</td></tr><tr><td>STD</td><td>0.0045</td><td>0.0052</td><td>0.0060</td><td>0.0088</td><td>0.0090</td><td>0.0080</td></tr></table>

## REFERENCES

[1] K. Li, J. Zhang, L. Wang, M. Zhang, J. Li, and S. Bao, “A review of the key technologies for semg-based human-robot interaction systems,” Biomedical Signal Processing and Control, vol. 62, p. 102074, 2020. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1746809420302305

[2] A. Mohebbi, “Human-robot interaction in rehabilitation and assistance: a review,” Current Robotics Reports, vol. 1, no. 3, pp. 131–144, 2020.

[3] A. Goen and D. Tiwari, “Review of surface electromyogram signals: its analysis and applications,” International Journal of Electrical and Computer Engineering, vol. 7, no. 11, pp. 1429–1437, 2013.

[4] T. du Plessis, K. Djouani, and C. Oosthuizen, “A review of active hand exoskeletons for rehabilitation and assistance,” Robotics, vol. 10, no. 1, 2021. [Online]. Available: https://www.mdpi.com/2218-6581/10/1/40

[5] S.-H. Yeo, J. Verheul, W. Herzog, and S. Sueda, “Numerical instability of hill-type muscle models,” Journal of the Royal Society Interface, vol. 20, no. 199, p. 20220430, 2023.

[6] J. Han, Q. Ding, A. Xiong, and X. Zhao, “A state-space emg model for the estimation of continuous joint movements,” IEEE Transactions on Industrial Electronics, vol. 62, no. 7, pp. 4267–4275, 2015.

[7] H. Li, S. Guo, D. Bu, and H. Wang, “A two-stage ga-based semg feature selection method for user-independent continuous estimation of elbow angles,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1–9, 2023.

[8] F. Xiao, Y. Wang, Y. Gao, Y. Zhu, and J. Zhao, “Continuous estimation of joint angle from electromyography using multiple timedelayed features and random forests,” Biomedical Signal Processing and Control, vol. 39, pp. 303–311, 2018. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1746809417301775

[9] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, Z. Yang, Y. Zhang, and D. Tao, “A survey on vision transformer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 87–110, 2023.

[10] R. B. Azhiri, M. Esmaeili, and M. Nourani, “Real-time emg signal classification via recurrent neural networks,” in 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2021, pp. 2628– 2635.

[11] T. Kim and S. Cho, “Predicting residential energy consumption using cnn-lstm neural networks,” Energy, vol. 182, pp. 72–81, 2019. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0360544219311223

[12] E. A. Clancy, E. L. Morin, G. Hajian, and R. Merletti, “Tutorial. surface electromyogram (semg) amplitude estimation: Best practices,” Journal of Electromyography and Kinesiology, vol. 72, p. 102807, 2023. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S1050641123000664

[13] Y. Zhao, J. Zhang, and C. Zong, “Transformer: A general framework from machine translation to others,” Machine Intelligence Research, vol. 20, no. 4, pp. 514–538, 2023.

[14] T. Mickus, D. Paperno, and M. Constant, “How to Dissect a Muppet: The Structure of Transformer Embedding Spaces,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 981–996, 09 2022. [Online]. Available: https://doi.org/10.1162/tacl a 00501

[15] E. Voita, D. Talbot, F. Moiseev, R. Sennrich, and I. Titov, “Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned,” arXiv preprint arXiv:1905.09418, 2019.

[16] D. Qiu and B. Yang, “Text summarization based on multi-head selfattention mechanism and pointer network,” Complex & Intelligent Sys tems, pp. 1–13, 2022.

[17] L. Gong, Y. Li, J. Guo, Z. Yu, and S. Gao, “Enhancing low-resource neural machine translation with syntax-graph guided self-attention,” Knowledge-Based Systems, vol. 246, p. 108615, 2022. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S0950705122002763

[18] F. Stahlberg, “Neural machine translation: A review,” Journal of Artifi cial Intelligence Research, vol. 69, pp. 343–418, 2020.

[19] S. Ahmed, I. E. Nielsen, A. Tripathi, S. Siddiqui, R. P. Ramachandran, and G. Rasool, “Transformers in time-series analysis: A tutorial,” Circuits, Systems, and Signal Processing, vol. 42, no. 12, pp. 7433– 7466, 2023.

[20] Z. Niu, G. Zhong, and H. Yu, “A review on the attention mechanism of deep learning,” Neurocomputing, vol. 452, pp. 48–62, 2021. [Online]. Available: https://www.sciencedirect.com/science/article/pii/S092523122100477X

[21] T. Fang, J. Wang, W. Mu, Z. Song, X. Zhang, G. Zhan, P. Wang, J. Bin, L. Niu, L. Zhang, and X. Kang, “Noninvasive neuroimaging and spatial filter transform enable ultra low delay motor imagery eeg decoding,” Journal of Neural Engineering, vol. 19, no. 6, p. 066034, dec 2022. [Online]. Available: https://dx.doi.org/10.1088/1741-2552/aca82d

[22] D. Chicco, M. J. Warrens, and G. Jurman, “The coefficient of determination r-squared is more informative than smape, mae, mape, mse and rmse in regression analysis evaluation,” Peerj computer science, vol. 7, p. e623, 2021.

![](images/15d62f3c9b08f11785ef14c7b2c6edc2a991514d90d905c4f2f305b3a250ad31.jpg)

(a) Predicted vs. True Joint Angles for Subject A  
![](images/a8c50ed0ecae5a8fce88d629b51ce1a9d1344e3b2a6ebb636ec90ce62f4d7799.jpg)

(b) Predicted vs. True Joint Angles for Subject B  
![](images/439bd7a7049bf4efb7d8d12d6796b96c1384015e9103eff414192ecedb8784cb.jpg)

(c) Predicted vs. True Joint Angles for Subject C  
![](images/75b03bec1df001f28da27eb8d9c258db2f3e4f7d82ef43b82204bf375efa2497.jpg)

(d) Predicted vs. True Joint Angles for Subject D  
![](images/665a03a803b2eda796890c7446e645a2972ca82df2c22d05c7c09bcd630fe2dd.jpg)  
(e) Predicted vs. True Joint Angles for Subject E

![](images/c7417873e8323e124af40b8aa90905704046e7d1b954cb848a3e185a06995353.jpg)  
(b) CNN-LSTM Model Predicts Elbow Joint Angle vs. Actual Angle

![](images/45a29a42254d34ec43ea6f4422533a9107d15bcc984661cf6a4a7b043ba728d8.jpg)

(a) Transformer Model Predicted Elbow Joint Angle vs. Actual Angle  
![](images/19379b919cbae28ea6341346d6167bb863a3fd90da151c07f34b8e94c3bed648.jpg)  
(f) Predicted vs. True Joint Angles for Subject F  
(c) RNN Model Predicts Elbow Joint Angle vs. Actual Angle

![](images/9039c36b43e408dfc1249c58aef800ac921f976b08931b6b147b9575f49a5173.jpg)  
Fig. 4. Comparison of Transformer Model Predictions and Actual Elbow Angles for Six Subjects  
Fig. 5. Comparison of Elbow Joint Angle Predictions: Transformer Model vs. RNN Model and CNN-LSTM Model