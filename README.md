# Continual-Learning-with-Meta-Learning-and-Experience-Replay-
Meta_ER PictoBERT: Enhancing Continual Learning with Meta-Learning and Experience Replay for Personalized Pictogram Recommendations 

Augmentative and Alternative Communication (AAC) boards are Assistive Technology tools that try to compensate for the difficulties faced by people with Complex Communication Needs (CCN), such as people with down’s syndrome, autism spectrum disorder, intellectual disability, cerebral palsy, developmental apraxia of speech, or aphasia
A pictogram is a picture with a label that denotes an action, object, person, animal, or place. Predicting the next pictogram to be set in a sentence in construction is an essential feature for AAC boards to facilitate communication. These tools allow individuals with CCN to communicate themselves by selecting and arranging pictograms in sequence to make up a sentence, as shown in the example illustrated in Fig. 1.
<p align="center">
    <img src="imgs/efr-eg-2.png", style="width: 600px; height: 300px;"/></center>
</p>

[PictoBERT](https://github.com/jayralencar/pictoBERT), an adaptation of BERT for the next pictogram prediction task, with changed input embeddings to allow word-sense usage instead of words, considering that a word-sense represents a pictogram.

Continual learning (CL) aims to adaptively learn across time by leveraging previously learned data to improve generalization for future data.

* Model 1 : Experience Replay
   The model, ER_PictoBERT, is built using PyTorch Lightning and utilizes a pretrained BERT-based architecture (BertForMaskedLM: PictoBERT) for next word prediction tasks.
   It includes an external memory buffer mechanism to store and sample data for continual learning.
   - The `memory buffer` is designed to store past training samples to facilitate experience replay (ER), a technique commonly used in continual learning.
   - ER helps `mitigate catastrophic forgetting`, a common problem in neural networks when they are trained incrementally on new tasks. 
   - The buffer allows the model to remember and rehearse past data, effectively balancing learning between new and old data.

* Model 2 : Meta PictoBERT
+ <b>Base Model</b>: Utilizes the pre-trained BERT model (pictoBERT) designed for masked language modeling, which serves as the foundation for downstream tasks such as understanding and generating natural language text.

+ <b>Meta-Learning Framework</b>: The model leverages meta-learning techniques to adapt quickly to new tasks by simulating a scenario where it learns from a smaller dataset (trajectory data) and then evaluates on another set (meta-test data). This approach is designed to improve generalization to new, unseen data.
  - using a meta-learning approach, the model can quickly adapt to new tasks, demonstrating strong generalization capabilities
