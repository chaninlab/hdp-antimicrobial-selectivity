# Importing libraries
import os
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, recall_score
from sklearn.inspection import permutation_importance

#@ Ignoring the Warnings:
import warnings
warnings.filterwarnings(action="ignore", message="^ internal")

####################
# Data processing
####################

# Getting the Data:
github_url = "Saureus_desc_from_1st_run_master.csv"

def load_data(PATH=github_url):
  csv_path = os.path.join(PATH)
  return pd.read_csv(csv_path)

# Preparing the Data:
bacteria = load_data()
bacteria["Activity"] = bacteria["Activity"].map({"High": 1, "Low": 0}) # Mapping the binary class labels
data = bacteria.drop("Activity", axis=1) # Dropping the Labels from Training set.
target = bacteria["Activity"].copy() # Target Attributes.

####################
# Model building
####################

# Creating placeholders for various metrics that will be collected during the model building

#@ Evaluation Metrics:
oob_list = []                                                             # List for storing the OOB scores.
recall_list = []                                                          # List for storing the Recall scores.
kappa_list = []                                                           # List for storing the Kappa scores.

#@ GINI Features Importance:
gini_importances = {}                                                     # Instantiating the Empty Dictionary.
gini_importances["Columns"] = data.columns                                # Names of attributes.

#@ PERMUTATION Features Importance:
permutation_importances = {}                                              # Instantiating the Empty Dictionary.
permutation_importances["Columns"] = data.columns                         # Names of attributes.

# Building models via 25 iterations
for seed in range(123, 148):                                              # Iterating the Loop for 25 times.
  model = RandomForestClassifier(n_estimators=1000,                        # Instantiating the Classifier.
                                 n_jobs=-1, random_state=seed,            # Changing the value of seed.
                                 oob_score=True)                          # Returning the OOB score.
  model.fit(data, target)                                                 # Training the Classifier.

  #@ Evaluation Metrics:
  predictions = model.predict(data)                                       # Making the Predictions.
  oob = model.oob_score_                                                  # Calculating the OOB score.
  oob_list.append(oob)
  recall = recall_score(target, predictions)                              # Calculating the Recall score.
  recall_list.append(recall)
  kappa = cohen_kappa_score(target, predictions)                          # Calculating the Kappa score.
  kappa_list.append(kappa)

  #@ GINI Features Importance:
  gini_importances["Run" + str(seed)] = model.feature_importances_             # Features Importance: GINI.

  #@ PERMUTATION Features Importance:
  permutation_important = permutation_importance(model, data, target, n_repeats=5,    # Permutation Importance.
                                   n_jobs=-1, random_state=seed)
  permutation_importances["Run" + str(seed)] = permutation_important.importances_mean             # Features Importance: PERMUTATION.

# 1. Evaluation Metrics:
# I will calculate the Evaluation metrics such as OOB score, Recall score and Kappa score using the whole dataset.
# I will be changing the seed of Random Forest Classifier along with the loop. The values of seed starts from 123
# and goes up to 148 which will also be the range of Iteration of the loop which is 25 Iterations in total.

#@ Dictionary of Evaluation Metrics:
scores = {"OOB Score": oob_list,
          "Recall Score": recall_list,
          "Kappa Score": kappa_list}
scores = pd.DataFrame.from_dict(scores, orient="columns")                 # Creating the DataFrame.
scores.to_csv("./Saureus_evaluation_metrics_nestimator1000.csv")                                  # Saving the DataFrame into csv.

# 2. GINI: Features Importance
# I will be changing the seed of Random Forest Classifier along with the loop. The values of seed starts from 123
# and goes up to 148 which will also be the range of Iteration of the loop which is 25 Iterations in total.

#@ GINI Features Importance: Creating the DataFrame:
gini_importance = pd.DataFrame.from_dict(gini_importances, orient="columns")
gini_importance.set_index("Columns", inplace=True)                             # Reseting the Index.

# Now, I will calculate the mean of each rows which will be the mean of Importances of each Attributes or Features.
# Then I will create the new column called "Average" in the DataFrame which will store the mean of each Attributes or
# Features and I will sort the DataFrame according to this column in ascending order.

#@ Calculating the Mean of Attributes:
gini_importance["Average"] = gini_importance.mean(axis=1)
gini_importance["SD"] = gini_importance.std(axis=1)
gini_importance.sort_values(by=["Average"], ascending=False, inplace=True)          # Sorting the DataFrame.
gini_importance.to_csv("./Saureus_gini_importance_nestimator1000.csv")                                    # Saving the DataFrame.

# 3. PERMUTATION: Features Importance
# I will be changing the seed of Random Forest Classifier along with the loop. The values of seed starts from 123
# and goes up to 148 which will also be the range of Iteration of the loop which is 25 Iterations in total.

#@ Creating the DataFrame:
permutation_importance_df = pd.DataFrame.from_dict(permutation_importances, orient="columns")
permutation_importance_df.set_index("Columns", inplace=True)                             # Reseting the Index.

#@ Calculating the Mean of Attributes:
permutation_importance_df["Average"] = permutation_importance_df.mean(axis=1)
permutation_importance_df["SD"] = permutation_importance_df.std(axis=1)
permutation_importance_df.sort_values(by=["Average"], ascending=False, inplace=True)          # Sorting the DataFrame.
permutation_importance_df.to_csv("./Saureus_permutation_importance_nestimator1000.csv")                             # Saving the DataFrame.
