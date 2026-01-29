# Instructions
- Unzip the **Data** file, which contains sample maps.   
- The **Models** folder contains the trained skelUnet model for the three cases with the best results described in the paper.   
- **Deep Denoising Auto-Encoder.ipynb**, is a notebook that runs the selected model in feedforward mode for a map of your choice. 
- **graphics.ipynb**, reproduces the training loss function results from the data in the **tables_training** folder.
- The **main_v2.py** file contains the coded architecture of the proposed skelUnet neural network. In addition function to the training code, to run a new training session, you must change the path of the dataset files and run in a PyTorch with CUDA environment or change the device in the code.
  
