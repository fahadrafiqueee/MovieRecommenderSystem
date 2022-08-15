# MOVIE-RECOMMENDATION-SYSTEM
The main aim of this project is to combine multiple services and open-source tools to make a movie recommendation system to provide users with accurate movie recommendations based on the content of a particular movie.

This project works on a filtering based on the description or some data provided for this product. This recommendation system finds the similarity between products based on their context or description. The user's history is taken into account to find similar products that you may like.
Firstly, the user's preferences, the user's interest, the user's personal information, such as the age, or sometimes the history of the user. This data is represented by the user vector. Secondly, the product-related information is called the item vector. The element vector contains the properties(features) of product which is movies in the system, then this can be used by to calculate the similarity between them. The recommendations of movies are calculated using cosine similarity. If 'A' is the user vector and 'B' is an element vector, the cosine similarity.

# DataSet Link and some other files: -
Dataset - https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv

Two files from juypter notebook will be dumped in a form of pkl (i.e.Similarity.pkl and movies_list.pkl).
These files are used in the recommendation system.

# PROBLEM STATEMENT: - 
Many other apps, like Netflix, are all about connecting users to the movies they enjoy. It is therefore necessary to supply users with what they require. As a result, it is vital to understand what the user requires, which can be accomplished through recommendation algorithms. This improves the user experience.

Recommender Systems for Films: This section focuses on creating different types of recommendation engines, such as the Simple Generic Recommender, Content Based Filter, and User Based Collaborative Filter. The systems' performance is assessed in both qualitative and quantitative terms.

Also, the reason to choose python to build the recommender system is that python boasts a wide array of open-source libraries for recommender systems, including stream lit, pandas, pickle etc. It is great for data sets and more analyses and also Python's libraries are much more useful in a practical way.

Imagine a scenario where a person wants to enjoy their weekend by watching a suitable movie but often ends up endlessly scrolling in an attempt to find something to watch. They would already have certain preferences and he/she can use this recommender to watch any movie which is something relatable. It would save both the person and the entertainment providers a lot of hassle if the person input the movie which they want to see in an attempt to get recommendations and saved time for the customer.

# DATA COLLECTION 
In this recommendation system, two datasets of Tmdb_5000_movies & Tmdb_5000_credits have been used which we can find on Kaggle.
Tmdb_5000_movies contain 5000 movie names along with movie, Budget, language, overview, title, popularity, tagline, status (released or not), keywords, genres, production company, name of the directors etc.
Tmdb_5000_credits contain movie_id, cast, crew.
#DATA PREPROCESSING Data preprocessing, a component of information preparation, describes any form of processing performed on data to prepare it for one more processing procedure. it's traditionally been a very important preliminary step for the info science process. Data preprocessing transforms the info into a format that's more easily and effectively processed in data processing, machine learning and other data science tasks.

The key steps in data preprocessing: -

Data profiling: Data profiling is the technique of examining, analyzing and reviewing records or data to accumulate information about its quality. It starts with a survey of existing records and their characteristics. Data scientists discover records units which are pertinent to the matter at hand, inventory its significant attributes, and form a hypothesis of features that may be relevant for the proposed analytics or machine learning task. They also relate data sources to the relevant business concepts and consider which preprocessing libraries may well be used.
Data cleaning: The aim of information cleaning right here is to find the proper manner to rectify high-satisfactory issues like eliminating bad data, and filling in missing data to form the information efficient for the model.
Data reduction: information sets often include redundant data that arise from characterizing phenomena in numerous ways or data that are not relevant to a specific ML, AI or analytics task. Data reduction uses techniques like principal component analysis to remodel the data into an easier form suitable for particular use cases.
Data transformation: Data scientists consider how different components of the info should be structured to realize the most effective results. Structure unstructured data, combining relevant variables where it is smart, and choosing important ranges to specialize in are all samples of this.
Data enrichment: to supply the simplest results, data scientists analyze how different components of the information should be arranged. Structure unstructured data integrate relevant variables when appropriate, and target critical ranges are all examples
Data validation: At this stage of pre-processing, the sorted data is split into two sets. the primary set of knowledge is employed to coach and a machine learning or deep learning model generates insights. The second set is the testing data which is employed to measure the accuracy of the resulting model. This second step helps to identify any problems within the hypothesis employed in the cleaning and has the engineering of the information. If the info scientists are satisfied with the results, they'll push the preprocessing task to a knowledge engineer who figures out a way to scale it for production. If not, the info scientists can return and make changes to the way they implemented the information cleansing and have engineering steps.
At first, two datasets (Movies and credits) merged into one table on the idea of the movie title then the unnecessary columns have been removed.

Here is the list of features from the dataset which don't seem to be required within the recommendation system because it doesn't impact an excessive amount of the information so we are removing them from the table: budget homepage id original_language original_title popularity production_comapny production_countries release-date(not sure)

Now, we left with the columns ('movie_id','title','overview','genres', 'keywords','cast','crew'). In Python language, "ast" techniques is applied to the genres and keywords columns and access the top 3 cast actor from the cast table and remaining actor was drop. Same goes with the director’s name. We will create a function to get the director name from crew column because there is a lot of crew names in the data as we only want the director name and remove the whitespace from the columns ( Sam Worthington, Sam Johny). Sometimes user enters Sam and the machine get confused that is why removing whitespace and also applied the same function for the director name, movies, caste and keywords columns and splitting each word in overview columns Adding overview, genres, keywords, cast, and crew added into one table --> "Tag" Column table has to be created because of vectorization after merging the content of all data together into one column and removing the columns overview, genres, keywords, cast and crew.

# MODELIING: - 
Vectorization - Vectorization is a jargon term for a traditional method of turning raw data (text) into vectors of real numbers, which is the format that ML models allow. This approach has been around since the dawn of computing, has proven to be effective in a variety of disciplines, and is currently being applied in NLP. Vectorization is a phase in feature extraction in Machine Learning. By translating text to numerical vectors, the goal is to extract some identifiable features from the text for the model to learn from. In this project, for modelling the dataset, Cosine similarity is used.

# Screenshots: - 

![1 ipg](https://user-images.githubusercontent.com/82112139/184529433-46119771-9d37-430d-b41c-1f411b733215.jpg)


![2](https://user-images.githubusercontent.com/82112139/184529436-5610ee71-566c-4823-9906-687f03f65689.jpg)


![3](https://user-images.githubusercontent.com/82112139/184529443-8fd71cfd-6e58-45f1-9253-6a13a6cdd091.jpg)


![4](https://user-images.githubusercontent.com/82112139/184529445-00cfadb2-ab57-40e4-910f-206555b6ead1.gif)


# Working and Implementation of Cosine Similarity: -
Cosine similarity is a metric for determining how similar documents are regardless of size. It estimates the cosine of the angle formed by two vectors projected in a multi-dimensional space mathematically. Because of the cosine similarity, even if two comparable documents are separated by the Euclidean distance (due to the size of the documents), they are likely to be oriented closer together. This conclude that smaller the angle, cosine similarity will be higher.


![working](https://user-images.githubusercontent.com/82112139/184529453-9d05683a-1f19-4d24-b700-55b478ef7d47.jpg)



# LIMITATIONS: -

This system can be improved by combining multiple services and open-source tools to add on some more features like giving recommendations to user by collaborative filtering to get more accurate recommendations. This can also be more user friendly by adding details about the recommended movies which includes crew, movie trailer, Movies (Netflix, Hotstar, HBO etc) link by which the user can directly movie to the platform which is streaming that particular movie.
