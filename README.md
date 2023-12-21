# California-housing-prediction
In this project we are using California's housing data to predict housing prices. it's a regression task.
install anaconda navigator through the link:https://www.anaconda.com/download
download all the files from this repsitory.
launch jupyter notebook in anaconda navigator
![Screenshot 2023-12-21 094626](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/033700ef-08fa-4250-aeb6-dde5a99f9f36)
create a new python notebook and start coding as shown in the above screen shot. or else
![Screenshot 2023-12-21 094735](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/20409f8e-00d3-4673-bf28-25e4232b3e41)
upload all notebook and datasets downloaded from this repository and run.
note:I used two models one is linear regression it gives rmse value 0 so, its overfitting 
second one is ramdomn forest it works well but since its a ensemble methods it takes a little time.
I used pipelines,one for transformation of data and one for scaling and training.
import the data and look at attribute discription as shown below. 
![Screenshot 2023-12-21 100605](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/71307aa4-f563-4a9b-8096-61393af02920)
# now check the data distribution using histogram
![Screenshot 2023-12-21 100812](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/b1bc8154-9faf-4c56-91f5-888570fc7f99)
we notice a few things in this:
First, the median income attribute does not look like it is expressed in US dollars(USD). The data is scaled.The numbers represent roughly tens of thousands of dollars (e.g., 3 actually means about 30,000)
The housing median age and the median house value were also capped.Your Machine Learning algorithms may learn that prices never go beyond that limit.
Finally, many histograms are tail heavy: they extend much farther to the right of the median than to the left. This may make it a bit harder for some Machine Learning algorithms to detect patterns. We will try transforming these attributes later on to have more bell-shaped distributions.
# Creation of training and test set.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
test_set.head()
Let’s look at the median income histogram more closely most median income values are clustered around 1.5 to 6 (i.e.15,000–60,000), but some median incomes go far beyond 6. It is important to have a sufficient number of instances in your dataset for each stratum, or else the estimate of the stratum’s importance may be biased. This means that you should not have too many strata, and each stratum should be large enough. The following code uses the pd.cut() function to create an income category attribute with 5 categories (labeled from 1 to 5): category 1 ranges from 0 to 1.5 (i.e., less than 15,000), category 2 from 1.5 to 3, and so on.
![Screenshot 2023-12-21 101052](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/72596306-861e-463f-90d3-1fc72ce352ad)
# lets look at correlation matrix
![Screenshot 2023-12-21 101205](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/8c650dc9-a3d5-4756-bbf0-8a80ee9c5aa0)
![Screenshot 2023-12-21 101308](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/3002111d-8e73-4ce7-8e8a-0bd5d1b1860f)
The correlation coefficient ranges from –1 to 1. When it is close to 1, it means that there is a strong positive correlation; for example, the median house value tends to go up when the median income goes up. When the coefficient is close to –1, it means that there is a strong negative correlation; you can see a small negative correlation between the latitude and the median house value (i.e., prices have a slight tendency to go down when you go north).
Preparing the data for Machine Learning Algorithm
house = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
house_labels = strat_train_set["median_house_value"].copy()
# DATA Cleaning
 we will fill the the numerical missing values with their medians.
 Scikit-Learn provides a handy class to take care of missing values: SimpleImputer
from sklearn.impute import SimpleImputer
imputer= SimpleImputer(strategy='median')
house_num= house.drop('ocean_proximity', axis=1)
imputer.fit(house_num).
Although Scikit-Learn provides many useful transformers, you will need to write your own.

# Let's create a custom transformer to add extra attributes:

![Screenshot 2023-12-21 101643](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/b8248d95-013c-4f0a-8c73-51a5d0bdd3c5)
 The pipeline exposes the same methods as the final estimator. In this example, the last estimator is a StandardScaler,
 which is a transformer, so the pipeline has a transform() method that applies all the transforms to the data in sequence 
(and of course also a fit_transform() method, which is the one we used).
 we have handled the categorical columns and the numerical columns separately. It would be more convenient to have a single transformer able to 
 handle all columns, applying the appropriate transformations to each column.
# select and train a model

![Screenshot 2023-12-21 101801](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/ca030057-e885-4b33-b5eb-0de6d6351302)

![Screenshot 2023-12-21 101815](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/3a7cbe81-194b-4fd9-9ea3-4f65e3abd939)

![Screenshot 2023-12-21 101836](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/938d2c5f-26d2-4565-aedd-1efad4aea1df)
# Fine-Tuning the model using randomized search

![Screenshot 2023-12-21 102135](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/20c1cfb7-dccd-4de5-aa8f-24eb9cccc2f5)

# finally creating a single pipe line for scaling and traing of a model(RandomForestRegressor(random_state=42),threshold=0.005))

![Screenshot 2023-12-21 102304](https://github.com/Abhishek-N-appu/California-housing-prediction/assets/150499125/4c176358-3373-4e2a-b897-5ef430c2d94e)





