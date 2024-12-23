**PART 1: Model Prototyping**
The social logging is a powerful tool to ease the installation process whether you are a new user or registered user who has reinstalled the app.  
To test the asset of this tool is possible to use multiple methodologies. This study will be focused on those based on machine learning and data solutions.  
By the data solutions hand the best one could be a classic A|B test,  even if is possible to build another solution based on reinforce learning for example. In the A|B test, the control group are those who use the traditional logging and treatment group will be the social loggings users. 
In an A|B test the most important part of the design is always the metrics selection. The present case could present two questions to select the best metric: 
•	Do users register or log more when using social logging? 
•	Are the social logging users consistent or they only logging and left the app?  
•	Do the users uninstall the app more with social logging?
Based on those 3 questions is possible to design the metrics used for the A|B test. 
 The first one answer to the first question measuring the number of loggings.  Also, will check the dropout rate which is the inverse of the registered users. 
The second question will be answered with the number of active users. This metric depends of the definition of active users because the user interaction with the app could variate a lot depending of the business specifications .  
Finally, the last question would be answer by the number of users who uninstall the app in a sort time. 
Using two metrics is possible to have 4 different results: 

- Both metrics are better:
  The social logging is better
- Better in early logging but worse in consistency:
  Is the cost of social logging implementation paid by the information of the users logged?
- Worse in early logging but worse in consistency:
   Is the information and potential clients lost paid by the quality of the current users
- Both metrics are worse or the uninstall metric is greater:
   Social logging is worse

The main problem of an A|B test is always the cost of the experiment. The treatment and control group are clients with a potential benefit for the company which could be lost because of the experiment. 
Also, we must consider that the metrics are not perfect. Any metric will not cover all the possible situations and all the information. The possibility of have an increase on the loggings because of the social logging but not only in the first minutes after the app installation will be lost on this experiment.
To implement the experiment will be necessary to know the information available about any user when is downloading the app. With the information available is possible to segment the client to choose better the percentage of control group, for example, young users need a lower percentage of control. Also, is important to have a representative users portfolio sample because the transition between control and treatment group is based in sampling. It means the security of the method will be better when the sample is increased but it won’t ever be perfect. Once the segmentation and proportions of any group are chosen any new user will be randomized put into treatment or control group only paying attention to the proportion to be representative at every segment. 
Once all the data is available is mandatory to consider the time evolution. For this porpoise is possible to treat the metrics not only as a complete par but, also as deltas. To define the deltas also is important to define the success and fail registration journey based on the define metrics:

•	Success: 
1.	Download
2.	Registration completed.
3.	The user turns into an active user.
   
•	Failed: In the failed journey the funnel has not such a simplicity.
1.	Download.
2.	Registration when is not completed paying attention if that implicate an app uninstall.
3.	The user does not turn into an active user paying attention if that implicate an app uninstall.
   
So is important to analyse the time that users’ need to complete the journey successfully. For example, the first day a 75% of the users complete the journey, the 25% rest of the journey are lost in this part of the funnel. Then the analysis will be repeated in the following 5, 7, etc days until have a enough consistent analysis to know the evolution of the users portfolio.
Finally, a simple statistic test like a t-stundent test will confirm or not the statistically differences between groups paying attention to the time and the absolute metrics.

**PART 2: Model Prototyping**
The first step to start a model is always to analyse and clean the dataset. At the beginning the lack of journeys comparing to those which are noted in the enounced of the problem. No matter the way the json file is read (whether as text or a sequence of dictionaries) only 3027 routes are found.
Also, at the initial information is possible to get the first hint to clean the dataset due to the present of some routes duplicated due to the analysis of 2 or more annotators. For this labour the first step was to analyse duplicated labelled journeys, finding out a not depreciable part of those (more than the 50% of the sample) are labelled at least once with “I don’t know”. Trying to have a more specific and bigger sample only the journey labelled with a different ticket than “I don’t know”, of those duplicated, will be saved. Also, those journeys that differs and are analyse for more than 2 annotators are ordered to keep only the most frequent label. Finally, all those journeys labelled as “I don’t know” are dropped trying to avoid the “garbage in, garbage out” principle.
To archive the purpose business case in the minimum time possible as is asked, we start researching similar projects. This research is started using the main free use LLM (Gemini & Bing Copilot). The option chosen is the Gemini one which recommend using multiple to measure the distance between routes and use those distances to classify. The LLM model purpose is use the following distances:
•	Fréchet inception distance (FID): This metric is used to measure the difference between real & predicted distribution by compare the mean and covariance in distributions. This metric brings a good information for the proposed model but is only accurate for same shaped embeddings. A possible solution to calculate the FID is suppose that once the sorted route is over the car will keep always in the same place. This solution will solve the technical problem, but it could confuse the model in case when the movement is pretty sort. For this reason, other metrics are tried.
•	Dynamic Time Warping (DTW): This technique is used to find an optimal alignment between two sequences. the quality of the route even if is not only measure for the time cost is one on the main KPIs used. For this reason, this technique is one of the best options for the present problem due to the time dependency suppose. 
•	Hausdorff Distance: This distance tries to measure the maximum distance between both embeddings in the metric space.
Due to the problem presented by the FID appears the need of new metrics

- Based on Google [1] recommendations is possible to use the following distnaces:
•	Euclidean distance: Measure the distance between ends of vectors.
•	Cosine distance:  Cosine of angle θ between vectors.
•	Scalar product: Cosine multiplied by lengths of both vectors.

- Based on Geek4Geeks [2] recommendations is possible to use the following distances:
•	Longest Common Subsequence (LCS): As its name suggests, the LSC return the length of the longest common subsequence in two strings. This metric is used for strings but is possible to apply in the present business case because it brings information about if the route diverges from the estimate one every time or have part of the path in common.
•	Levenshtein distance: represent the minimum number of operations that could be performed to turn a vector v1 into a different vector v2.

Taken multiple options finally only 4 are chosen to build models with them (DTW, LCS, Levenshtein & Hausdorff distance).
Once the features for the model are built and is possible to look for the best model & research its hyperparameters.
The first option use is an XGBoostClassifier which use to have an efficient performance. For the hyperparameters research the option use is Optuna [3]. To make an interactive model selection a Streamlit [4] app is running with the main parameters to evaluate the model performance and interactive widgets which allows to set some model parameters as the cut point, and the features used build the model.
In the first research is possible to see that, indeed, the model is good enough (Figure 1) and is not needed to try another model. Therefore, the model is saved as the version xgboost_model_v8.joblib which only use (DTW and Levenstein features) to be used as many times as its needed.

 ![image](https://github.com/user-attachments/assets/fa19e959-4fe6-4e89-87d2-23db2747407c)
Figure 1

Google [1]: https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity?hl=es-419
Geek4Geeks [2]: https://www.geeksforgeeks.org/
Optuna [3]: https://github.com/optuna/optuna 
Streamlit [4]: https://docs.streamlit.io/ 
