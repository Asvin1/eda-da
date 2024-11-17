# %% [markdown]
# ## EDA THEORY
# 
# ## DIGITAL ASSIGNMENT
# 
# **ASVIN JAIN
# 21BDS0110**

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import seaborn as sns
import os

# %%
df = pd.read_csv("chess_games.csv", delimiter=',')
df.head()

# %%
df.info()

# %%
df.isnull().sum()

# %%
df[['black_rating', 'white_rating', 'turns']].hist(figsize=(12, 6), bins=50, grid=False)

# %%
c=pd.qcut(df['turns'], 4)
c

# %%
pd.value_counts(c)

# %%
df.winner.value_counts().plot(kind='bar', figsize=(10,6))
plt.title("Frequency of winner")
plt.ylabel('Frequency')
plt.xlabel('Winner')

# %%
sns.boxplot(x="turns",data=df)
plt.show()

# %%
match_time=df["time_increment"].value_counts()[:20] # take first 20 value if we take all values it will be very big bar there is no need for that first 20 is good enough
match_time=match_time.sort_values(ascending=True)
x,y=(match_time.index,match_time)

plt.barh(x, y/1000) # we divide match counts to 1000 for   better visualization

plt.xlabel("total match played(thousand)",labelpad=20 ) # it is thousand because we divide with 1000
plt.ylabel("Time Control",labelpad=20)
plt.title("Time Control - Match Count")

# %%
opening=df["opening_fullname"].value_counts().sort_values(ascending=False)[:50]
opening.sort_values(ascending=True,inplace=True)
plt.rcParams["figure.figsize"] = (8,12)
plt.barh(opening.index,opening)
plt.xlabel("Match Count",labelpad=20)
plt.ylabel("Opening strategy Name",labelpad=20)
plt.title("Opening Strategy-Match Count",loc="left")

# %%
plt.scatter(df[:500]["white_rating"], df[:500]["black_rating"])
plt.title("Black vs White rating")
plt.xlabel("White")
plt.ylabel("Black")

# %%
sns.set_theme(style="ticks", color_codes=True)
sns.pairplot(df[:1000],vars = ['black_rating', 'white_rating', 'turns'], kind="reg",hue='victory_status')
plt.show()

# %%
from scipy import stats

corr = stats.pearsonr(df["opening_moves"], df["turns"])
print("p-value:\t", corr[1])
print("cor:\t\t", corr[0])

# %%
correlation = df.corr(method='pearson', numeric_only=True)
correlation

# %%
sns.heatmap(correlation,xticklabels=correlation.columns,
            yticklabels=correlation.columns)

# %%
#LINEAR REGRESSION
X = df[['white_rating']]
y = df[['black_rating']]
# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
#Training a Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Fitting the training data to our model
regressor.fit(X_train, y_train)
#check prediction score/accuracy
print(regressor.score(X_test, y_test))


# %%
# predict the y values
y_pred=regressor.predict(X_test)
# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({'Actual': y_test.values.flatten(),
'Predicted': y_pred.flatten()})
evaluate.head(10)
evaluate.head(10).plot(kind = 'bar')

# %%
#Computing accuracy
# Scoring the model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# R2 Score
print(f"R2 score: {r2_score(y_test, y_pred)}")
# Mean Absolute Error (MAE)
print(f"MAE score: {mean_absolute_error(y_test, y_pred)}")
# Mean Squared Error (MSE)
print(f"MSE score: {mean_squared_error(y_test, y_pred)}")

# %%
#CLUSTERING
from scipy.cluster.hierarchy import dendrogram, linkage

data = list(zip( df["opening_moves"],df["turns"]))

linkage_data = linkage(data[:50], method='ward', metric='euclidean')
dendrogram(linkage_data)

# %%
#PCA
X = df[['black_rating', 'white_rating', 'turns']].values
y = df[['opening_moves']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
explained_variance


