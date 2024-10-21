# ATP-Pred: An End-to-End Prediction Method for Protein-ATP-binding Residues Using Pretrained Protein Language Models

This repository contains the code for the ATP-Pred framework, which is used for predicting protein-ATP binding residues. 

PDNAPred relies on two large-scale pre-trained protein language models: ESM-2 and ProtT5. These models are implemented using Hugging Face's Transformers library and PyTorch. Please make sure to install the required dependencies beforehand.

- ESM-2: [https://huggingface.co/facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_35M_UR50D)
- ProtT5: https://huggingface.co/Rostlab/prot_t5_xl_uniref50

# Usage

We have placed the complete test code in the inferrence.py file, the only thing that needs to be done is to download the weights and modify the path.

Our model can be download from :https://drive.google.com/drive/folders/1SEXdnmO78jTQg8bYR_Atum47AtVQIZBx?usp=drive_link

So, you can test only one sequence.

# Contact

If you have any questions regarding the code, paper, or trained model, please feel free to contact Lingrong Zhang at [zlr_zmm@163.com](mailto:zlr_zmm@163.com).
