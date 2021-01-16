# Prediction of DNA binding proteins using local features and long-term dependencies with primary sequences based on deep learning 


Guobin Li 1, Xiuquan Du 2, Xinlu Li 1, Le Zou 1, Guanhong Zhang 1 and Zhize Wu 1

1 School of Artificial Intelligence and Big Data, Hefei University, Hefei 230601, China
2 School of Computer Science and Technology, Anhui University, Hefei 230601, China

**Abstract**

DNA-binding proteins (DBPs) play pivotal roles in many biological functions such as alternative splicing, RNA editing, and methylation. Many researchers have proposed traditional computational methods to predict DBPs. However, these methods either rely on manual feature extraction or fail to capture long-term dependencies in the amino acid sequence. This paper proposes the PDBP-Fusion method based on deep learning (DL) to identify DBPs only from primary sequences. The proposed approach predicts DBPs based on the fusion of local characteristics and long-term dependencies. A convolutional neural network (CNN) is used to complete local feature learning, and a bi-directional long-short term memory network (Bi-LSTM) is used to capture critical long-term dependencies in context.  Furthermore, feature extraction, model training, and model prediction are achieved simultaneously. The proposed PDBP-Fusion approach can predict DBPs with 86.90% sensitivity, 78.20% specificity, 82.57% accuracy, and 0.656 MCC on the PDB14189 benchmark dataset. The MCC of the authors' proposed methods was increased by at least 8% compared with other advanced prediction models in the literature. Moreover, the proposed method outperforms most existing classifiers on the PDB2272 independent dataset.  The proposed PDBP-Fusion approach can be used to predict DBPs from sequences effectively; the online server is at http://119.45.144.26:8080/PDBP-Fusion/.

Dataset:



Sourecode usage:

