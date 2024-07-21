# # # # ###################### import libraries  ####################
import numpy as np
import pandas as pd

# # # ###################### Reading CSV Dataset  #################
df=pd.read_csv("Ordinal_categorial_dataset.csv")
# print(df)
# print(df.describe())
# print(df.shape)
# print(df.info())
# print(df.head())
# # # print(df.tail(3))
# print(df.sample(4))

#############################################################
#############################################################
###################### Dealing Ordinal Data #################
#############################################################
#############################################################

# size_mapping = {'XL': 3,'L': 2,'M': 1}
# df['size'] = df['size'].map(size_mapping)
# print(df)


# # # ################ Reverting into Ordinal Data ########################

# # inv_size_mapping = {v: k for k, v in size_mapping.items()}
# # print(inv_size_mapping)
# # df1=df['size'].map(inv_size_mapping)
# # print(df1)

# # # ###################### Using LabelEncoder Class ###########################
# from sklearn.preprocessing import LabelEncoder 
# class_le = LabelEncoder()
# df['classlabel']= class_le.fit_transform(df['classlabel'])
# # df['color']= class_le.fit_transform(df['color'])
# print(df)

# # # ################## Dealing Categorical data using get_dummies Method ###

# dummy_var=pd.get_dummies(df.color, prefix='Color',drop_first=True)
# print(dummy_var)
# # # # # # dummy_var=pd.get_dummies(df['color'], prefix='Color',drop_first=True)
# df=pd.concat([df,dummy_var],axis='columns')
# print(df)

# # # # # ######################drop first dummy variable column #####################

# df=df.drop(['color'],axis='columns')
# print(df)

# df.to_csv("preprocessed1.csv", index=None)
# # # #####Separating input features and target  ################

# # X=df.drop(['classlabel'], axis='columns')
# # y=df['classlabel']
# # # print(X)
# # # # # # print(y)
# # X1=df.iloc[:,[0,1,3,4]].values
# # y1=df.iloc[:,2].values
# # # print(X1)
# # # ####### Converting into DataFrame  if required ###############

# # X2=pd.DataFrame(X1, columns=["size",'price','Color_Green','Color_Red'])
# # y2=pd.DataFrame(y1, columns=['classlabel'])

# # data=pd.concat([X2,y2], axis=1)
# # print(data)

# # # ######## Writing CSV Dataset (Save Dataset)  #################

# # data.to_csv("preprocessed1.csv", index=None)

# # ######### Splitting dataset into train and test dataset  #######

# # # # from sklearn.model_selection import train_test_split
# # # # x_train, x_test, y_train, y_test = train_test_split(X1, y1, test_size=0.3, random_state=1)

# # # ###################### LogisticRegression algorithms Application  ##########

# # # # from sklearn.linear_model import LogisticRegression
# # # # LR=LogisticRegression()
# # # # LR.fit(x_train,y_train)
# # # # print(LR.score( x_test,y_test)*100)


# # # ###########################################################################
# # # ###########################################################################
# # # ###################### Handeling Missing values ###########################
# # # ###########################################################################
# # # ###########################################################################
# # # # import numpy as np

# import pandas as pd

# data = pd.read_csv("hiring.csv")
# # print(data)

# names=['Experience','test_score','interview_score','salary' ]
# data.columns=names
# print(data)
# # # # # data1=data.iloc[:,:].values
# # # # # data=pd.DataFrame(data1,columns=names)
# # print(data)

# data.Experience = data.Experience.fillna("zero")
# # print(data)

# # # # # ##################################################
# # # # # ########### Install word2number library  #########
# # # # # ###########  pip install word2number   ###########
# # # # # ##################################################

# from word2number import w2n

# data.Experience = data.Experience.apply(w2n.word_to_num)
# print(data)

# # # # # ##################################################
# # # # # ########## Filling Null value with Median ########
# # # # # ##################################################
# import math
# median_t_score = data['test_score'].median()
# print(median_t_score)
# data['test_score'] = data['test_score'].fillna(median_t_score)
# print(data)
# # # # ##################################################
# # # # #########  Filling Null value with Mean ##########
# # # # ##################################################
# mean_t_score = math.floor(data['test_score'].mean())
# print(mean_t_score)
# data['test_score'] = data['test_score'].fillna(mean_t_score)
# print(data)

# # # # ######### Filling Null value with zero ###########

# data['test_score'] = data['test_score'].fillna(0)
# print(data)

# # # # #########  Filling Null value with Previous value ###

# data['test_score'] = data['test_score'].fillna(method="ffill")
# print(data)

# # # # ##########  Filling Null value with next value #######

# # # data= data.fillna(method="ffill")
# # # print(data)

# # # # #########  Filling Null value with next value #######

# data['test_score'] = data['test_score'].fillna(method="bfill")
# print(data)

# # # # ########  Using StandardScaler  ####################

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# ########## transform data
# # data[['test_score','salary']] = scaler.fit_transform(data[['test_score','salary']])
# data['test_score'] = scaler.fit_transform(data[['test_score']])
# print(data)

# # # # #############  Using Normalizer to Normalize data between 0 and 1 ####################

# from sklearn.preprocessing import Normalizer
# Nor=Normalizer()
# # data[['test_score','interview_score']] = Nor.fit_transform(data[['test_score','interview_score']].values.reshape(-2,2))
# # data['test_score'] = Nor.fit_transform(data['test_score'].values.reshape(-1,1))
# data['test_score'] = Nor.fit_transform(data[['test_score']])
# print(data)

# # # # ##############  convert data between 0 and 1 ####################

# # # # # data['test_score'] = data.apply(lambda x: x / np.max(x), axis=0)['test_score']
# # # # # print(data)

# # # # ##############  convert data between 0 and 1 usnig max min method ####################

# # # # # data[['test_score','interview_score']] = data.apply(lambda x:(x.astype(float) - min(x))/(max(x)-min(x)), axis = 0)[['test_score','interview_score']]
# # # # # print(data)

# # # # ##############  convert data between 0 and 1 usnig max min method ####################

# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# data['test_score'] = scaler.fit_transform(data['test_score'].values.reshape(-1,1))
# print(data)

# # # # ############  Separating Input features and Target values ####################

# X=data.iloc[:,:-1].values
# y=data.iloc[:,3].values

# # # # #####################  Spliting dataset into train and test data ####################


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # # # # #####################  use linearRegression Model  ####################

# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# model.fit(X_train,y_train)
# acc=model.score(X_test,y_test)
# print(round(acc*100,2),"%")

























