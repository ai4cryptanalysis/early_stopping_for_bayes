# Early_Stopping_for_BayesianKeySearch

The code is related to 'Accelerating Machine Learning-Aided Key Recovery with Early Stopping Technique'.

**`Title`**: Accelerating Machine Learning-Aided Key Recovery with Early Stopping Technique<br> 

**`Author`**: <br> 

**`Abstract`**: In CRYPTO 2019, Gohr demonstrated the fusion of machine learning and differential cryptanalysis, revealing that differential-neural distinguishers (NDs) outperform traditional differential techniques in distinguishing attacks. Additionally, a novel key-recovery strategy incorporating Bayesian optimization was introduced to enhance the key recovery of Speck32/64. The subsequent works have provided insights into the analysis and improvement of NDs. However, the investigation of this innovative key-recovery strategy remains underexplored in current research literature.

In this paper, we focus on speeding up the machine learning-aided key-recovery strategy. By analyzing the performance difference when performing the Bayesian optimization on the correct and wrong ciphertext structures, we introduce an early stopping technique to optimize the BayesianKeySearch algorithm. By training neural networks to predict the likelihood of a ciphertext structure being right based on score trends during iterations of Bayesian optimization, we allocate more iterations to promising structures while terminating unpromising ones early, thereby, reducing the computational overhead and improving time complexity.

Applying our early stopping techniques, we enhance the key recovery attacks on 16-round Simon32/64 and 11-round Speck32/64, resulting in higher success rates and lower time complexities compared to previous works. Our results validate the practical effectiveness of the early stopping technique and are expected to contribute to the advancement of machine learning-aided cryptanalysis.<br><br>

**`Tested configuration`**<br>
Ubuntu20.04<br>
python == 3.8.19<br>
tensorflow-gpu == 2.5.0<br>
h5py == 3.1.0<br>
numpy == 1.19.5<br>
keras-nightly == 2.5.0.dev2021032900<br>
cudatoolkit == 11.2.2<br>
cudnn == 8.1.0.77 
