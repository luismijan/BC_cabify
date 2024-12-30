# PART 1: Experiment Design
The social logging is a powerful tool to ease the installation process whether you are a new user or registered user who has reinstalled the app.

To test the asset of this tool is possible to use multiple methodologies. This study will be focused on those based on machine learning and data solutions.

## 1.- Metodology

On the data solutions size the best one could be a classic A|B test, even if is possible to build another solution based on reinforce learning just like the company did for other projects as pricing. 
In the A|B test, the control group are those which use the traditional logging and treatment group will be the social loggings users.

In an A|B test the most important part is always the metrics selection. The present business case could present 3 questions to choose the best metrics:

- Do users register or log more when using social logging?
- Are the social logging users consistent or they only logging and left the app? (Do they turn into active users?)
- Do the users uninstall the app more with social logging?

Based on those 3 questions is possible to design the metrics used for the A|B test. 

The first metrics answer to the first question, measuring the number of loggings. Also, will check the dropout rate which is the inverse of the registered users taking the total app installations as rate base.

The second question will be answered with the number of active users. This metric depends of the definition of active users because the user interaction with the app could variate a lot depending of the business specifications. For example, is an active user from the moment the user search a journey? or are they active only when the do the journey?

Finally, the last question would be answer by the number of users who uninstall the app in a sort time. The period needed to consider a sort time may be choosen following the business specifications. 

Using two metrics is possible to have 4 different results:

| **Result**      | **Conclusion** |
| :---        |       ---: |
| **Both** metrics are **better**  | The social logging is better |
| **Better** in logging but **Worse** in consistency     | Is the cost of social logging implementation paid by the information of the users logged? |
| **Worse** in logging but **Better** in consistency    | Is the information and potential clients lost paid by the quality of the current users? |
| **Both** metrics are **worse** or the **uninstall** metric is **greater** |  Social logging is worse |

Once all the metrics are defined is import to consider the time evolution. For this porpoise is possible to treat the metrics not only as a complete par but, also as deltas. To define the deltas also is important to define the success and fail registration journey based on the define metrics:

- Success: 
1.	Download
2.	Registration completed.
3.	The user turns into an active user.
   
- Failed: In the failed journey the funnel has not such a simplicity.
1.	Download.
2.	Registration when is not completed paying attention if that implicate an app uninstall.
3.	In case the users are registred, the user does not turn into an active user paying attention if that implicate an app uninstall.

So is important to analyse the time that user's need to complete the journey successfully. For example, the first day a 75% of the users complete the journey, the 25% rest of the journey are lost in this part of the funnel. Then the analysis will be repeated in the following 5, 7, etc days until have a enough consistent analysis to know the evolution of the users portfolio and how the portfolio evolves. Also is important to know the part of the funnel where the failed users are lost to get better the company process.

Finally, a simple statistic test like a t-stundent test will confirm or not the statistically differences between groups paying attention to the time and the absolute metrics.

## 2.- Implementation

Once the data is available and the experiment designed is mandatory to select the best way to implement it. For this porpouse will be necessary to know the information available about any user when is downloading the app. With the available information is possible to segment the client to choose better the sample always having a right representation of any segment of the users portfolio. 

Another important part of the experiment design is deciding its duration and when to start it. For both questions is important to define when the loggings and registrations are normalized. The best way to find this period out is combine the previous analysis of the registration and loggings on previous years, and get supported by the business knowledge. For example, for previous analysis the business experts knows the summer is not the period to starts any experiment.

Maybe the most important part of any experiment based in statistics for this porpuose is possible to use the following formula [1] trying to find the perfect sample which reduce the type I and II errors:


$$
n = \frac{Z^2pqN}{e^2(N-1) + Z^2pq}
$$

Where:
> \( n \) = Sample Size

> \( Z \) = Z value for the confidence level.

> \( p \) = Expected proportion

> \( q \) = 1 - p (0.5)

> \( N \) = Population size

> \( e \) = Margin of error.

Once the segmentation and proportions of any group are chosen any new user will be randomized put into treatment or control group only paying attention to the proportion to be representative at every segment.

## 3.- Benefits and Weakness

Now the experiment is designed but is the experiment perfect?. Any experiment has weakness and the present experiment must suprass its weakness to be selected.

- Benefits:

> Is a cientific way to measure the benefits of social logging

> Is cheaper than other cientifics measurements

> It is an easily replicable experiment in this case.

- Weakness:

> The experiment is based in statisticall analysis so the confidence of the experiment will never be perfect.

> To implent the experiment is mandatory to deploy the social logging so, it takes some costs

> If there is a group which is clearly worse all the users included in this group, who have a potetial benefit for the company, could be lost.


## 4.- Bibliography

Sample size [1]: https://onlinetoolkit.co/es/calculadora-tamano-muestra/


# **PART 2**: Model Prototyping

Cabify is drivers are guide by a router algorithm which allows to find the most eficient way to go from point A to B. Even so, sometimes drivers take a different route from the suggested by the algorithm. 

If cabify drivers would do only a few rides different from the planed route per day would be easy to find those which are similars by hand, but they are much more than is possible to analyze by humans. 

In this context the needs for a algorithm which recognise the similarity and differences in routes to tag they as "Both are the same" and "They differ".

## 2.- State of the art 

To archive the goal of have the project in the marked time the reseach starts with 2 of the most importants large lenguage models (LLM) at the moment (GPT-4 by Microsoft Bing Copilot & Gemini by Google). After analyze the porpoused options the most reliable appears to be the Gemini one which recommend using multiple embedings distances to measure the distance between routes and use those distances to classify. The LLM model purpose is use the following distances:

> **Fréchet inception distance (FID)**: This metric is used to measure the difference between real & predicted distribution by compare the mean and covariance in distributions. This metric brings a good information for the proposed model but is only accurate for same shaped embeddings. A possible solution to calculate the FID is suppose that once the sorted route is over the car will keep always in the same place. This solution will solve the technical problem, but it could confuse the model in case when the movement is pretty sort. For this reason, other metrics are tried.

> **Dynamic Time Warping (DTW)**: This technique is used to find an optimal alignment between two sequences. the quality of the route even if is not only measure for the time cost is one on the main KPIs used. For this reason, this technique is one of the best options for the present problem due to the time dependency suppose.

> **Hausdorff Distance**: This distance tries to measure the maximum distance between both embeddings in the metric space.

Due to the problem presented by the FID appears the need of new metrics, so a new research starts:


- Based on Google [1] recommendations is possible to use the following distnaces:
> **Euclidean distance**: Measure the distance between ends of vectors.

> **Cosine distance**:  Cosine of angle θ between vectors.

> **Scalar product**: Cosine multiplied by lengths of both vectors.

- Based on Geek4Geeks [2] recommendations is possible to use the following distances:
 
> **Longest Common Subsequence (LCS)**: As its name suggests, the LSC return the length of the longest common subsequence in two strings. This metric is used for strings but is possible to apply in the present business case because it brings information about if the route diverges from the estimate one every time or have part of the path in common.

> **Levenshtein distance**: represent the minimum number of operations that could be performed to turn a vector v1 into a different vector v2.

Taken multiple options finally only 4 are chosen to try build models with them (DTW, LCS, Levenshtein & Hausdorff distance).


Once the features for the model would be built and would be possible to look for the best model & research its hyperparameters.

The first option use is an XGBoostClassifier [3] which used to have an efficient performance. 

For the hyperparameters research the option use is Optuna [4]. 

To make an interactive model selection which allows to select the best features and evaluates its performances, is possible to build an Streamlit [5] app. The app is running with the main parameters to evaluate the model performance and interactive widgets which allows to set some model parameters as the cut point, and the features used build the model.

## 3.- Data cleaning.

The first step to start a model is always to analyse and clean the dataset. 

At the beginning the lack of journeys comparing to those which are noted in the enounced of the problem. No matter the way the json file is read (whether as text or a sequence of dictionaries) only 3027 routes are found.

Also, at the initial information is possible to get the first hint to clean the dataset due to the present of some routes duplicated due to the analysis of 2 or more annotators. For this labour the first step was to analyse duplicated labelled journeys, finding out a not depreciable part of those (more than the 60% of the sample) are labelled at least once with “I don’t know”. Trying to have a more specific and bigger sample only the journey labelled with a different ticket than “I don’t know”, of those duplicated, will be saved. 
Also, those journeys that differs and are analyse but is impossible to know which annotator were right. Even in the case when there were more annotators tagging in a wey the journey than those who tag in the opposite is impossible to comfirm that the right one are the majority. So, trying to avoid the “garbage in, garbage out” principle all the duplicated and not eassilly confidable journeys are dropped.
Finally, all those journeys labelled as “I don’t know” are dropped due to the possibility of confuse the algorithm with routes which are different but tagged as "I don't know" and also, journeys tagged on the same way but simillars. 

## 4.- Model results

Once the data is compleatlly clean and transform into the distances said in the state of the art, is possible to start building the model ussing the algorithm also said on this part. 
To ensure the asset of the model is possible to pay attention to multiples metrics. Some of the most important metrics are acuraccy, the precission, recall and f1 score. The curve of false positives and false negatives is important to know the discrimination power of the model too.

To build the model the dataset is splited into train (80% of the sample) and test (20%) stratifying by the annotation. All the follow metrics and plots are ready to compare the train with the test data. 

At the be first comparation is in the **False Negative Rate** and **False Positive Rate** curve. This curve shows that there is not a single point where the proving the quality of the model at having a quickly but progressive drop in the false positive rate and the opposite in the false negative. Also, the curve shows the perfect cut point at 0.6.

![image](https://github.com/user-attachments/assets/ef88526d-e6d6-4d53-9620-436482c3e3a6)

Looking at the metrics the recall is always a little better than precission but in both cases is always metrics good enought. The only point to pay attention is the lightly increase of the recall in the "Both are the same" so would be recomendable to analyze it with more evaluation data before past it to production.

### Train
| | Precision | Recall | F1 |
| :--- | :---: | :---: | ---: |
| Both are the same | 0.9255 | 0.9057 | 0.9155 |
| They differ | 0.9290 | 0.9442 | 00.9365 |
| | | | |
| Accuracy | | | 0.9275	|
| Macro Avg | 0.9273 |0.9250 | 0.9260 |
| Weighted Avg | 0.9275 | 0.9275 | 0.9274 |

### Test
| | Precision | Recall | F1 |
| :--- | :---: | :---: | ---: |
| Both are the same | 0.9058 | 0.9099 | 0.9079 |
| They differ | 0.9365 | 0.9335 | 0.9350 |
| | | | |
| Accuracy | | | 0.9238 |
| Macro Avg | 0.9212 | 0.9217	| 0.9214 |
| Weighted Avg | 0.9238 | 0.9238 | 0.9238 |

Finally looking at the shap [6] values is possible to see the clear behaviour of the variables:
- DTW: 

> **Low values**: Is the most clear behaviour and always involves high odds results, what means is more probable than the routes are different.

> **High values**: Are the most repeated and has different results in the model (most probably depending on the Levenshtein value).

- Levenshtein: 

> **Low values**: Involves high odds results, what means is more probable than the routes are different.

> **High values**: Involves lower odds results, what means is more probable than the routes are similars.

### Train

![image](https://github.com/user-attachments/assets/60c31d7b-a1bc-4cf0-973a-8f9bd24912fa)

### Test

![image](https://github.com/user-attachments/assets/1f361c21-6365-4e73-8bb9-931087c8436e)

## 5.- Bibliography

> Google [1]: https://developers.google.com/machine-learning/clustering/dnn-clustering/supervised-similarity?hl=es-419

> Geek4Geeks [2]: https://www.geeksforgeeks.org/

> XGBoost [3]: https://xgboost.readthedocs.io/en/stable/python/index.html

> Optuna [4]: https://github.com/optuna/optuna

> Streamlit [5]: https://docs.streamlit.io/

> Shap [6]: https://shap.readthedocs.io/en/latest/index.html
