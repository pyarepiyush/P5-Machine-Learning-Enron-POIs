# P5 - ML Final Project
## Enron Persons Of Interest (POIs)
Udacity Data Analyst NanoDegree Project5 submission

The objective of this Project is to use Machine Learning to identify persons of interest related to collapse of Enron in 2002.

This project uses various classifiers from python scikit learn package.  All project related files are kept in 'final_project' folder.  Here is the content and its description:

**poi_id.py** --> Python script all steps from reading the data, to choosing the final classifier. It reads, cleans, created new features, compares different classifiers, tunes the parameters for chosen classifier, and finally outputs the selected classifier into .pkl files.

**ML Final Project Answers to questions.pdf** --> .pdf file with answers to all questions asked as part of the final project.

**final_project_dataset.pkl** --> Dictionary with input data for Enron employees. It contains financial and email information for those employees. It also contains labels for POIs vs. non-POIs.  

**my_classifier.pkl** --> Dump of final classifier with appropriate parameters chosen

**my_dataset.pkl** --> Dump of the dataset to be used with the selected classifier. Outliers and cleaned, and new features are added.

**my_feature_list** --> Dump of all features used in the selected classifier (that aligns with my_dataset.pkl) 

