# ProteinCoronaPredict_PayneLab
 Predicting Protein Abundance through NP and Protein Features and Experimental Conditions.

**Running/Using this Code**
This code is structured to be run on Google Colab (free version). 

- You will need to ensure that the data_prep_functions.py is linked to the correct directory, or else it cannot be accessed (necessary for RFC.ipynb and RFR.ipynb).
- #### Depending on how you title your code, all of the linked file directories will need to be adjusted. If using Google Coloab, simply right-click on the file and select 'copy path' to ensure correct path is selected and then pasted in the code where applicable.

**Important:** If you choose to rename the parent folder or any subdirectories, please ensure that you update the file paths in the code accordingly. This will allow the code to locate the necessary files without any issues.

**Instructions for Running the Code**

1) Upload Raw Mass Spectrometry Data:
   Open the Proteomic_Data_Perseus_to_df.ipynb notebook in Google Colab.
   Upload your raw mass spectrometry data (.txt files) within the notebook.
   Follow the steps to convert the .txt file into a readable input format for further analysis.

2) Merge Relevant Data:
   Open the Data_Consolidation.ipynb notebook in Google Colab.
   Use this notebook to consolidate and merge all relevant data files.
   Detailed instructions regarding the necessary input information can be found at the top of the Colab notebook.
   Data file/DataFrame used for the publication: df_Bov Swiss Intensity _v1.xlsx

4) Run the Prediction Models:
   Depending on your analysis goal, you can choose to run one of the following notebooks:
   RFC.ipynb: Use this notebook to predict categorical protein enrichment/depletion values using a Random Forest Classifier.
   RFR.ipynb: Use this notebook to predict continuous protein abundance values using a Random Forest Regressor.
