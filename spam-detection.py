#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('spam (1).csv', encoding='latin1')


# In[3]:


df.sample(5)


# In[4]:


df.shape


# In[59]:


#DATA clean 2) EDA 3)Text preprocessing 4)model building 5)Evaluation 6)website convert


# # 1.Data cleaning

# In[5]:


df.info()


# In[6]:


#drop last 3 column
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[7]:


df.sample(5)


# In[8]:


#renaming the cols
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[9]:


df.sample(5)


# In[10]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[11]:


df['target'] = encoder.fit_transform(df['target'])


# In[12]:


df.sample(5)


# In[13]:


#missing value
df.isnull().sum()


# In[14]:


#check  duplicate values
df.duplicated().sum()


# In[15]:


#remove duplicates values
df=df.drop_duplicates(keep='first')


# In[16]:


df.duplicated().sum()


# In[72]:


df.shape


# # 2.EDA

# In[17]:


df.sample(5)


# In[18]:


df['target'].value_counts()


# In[52]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[76]:


#Data is imbalanced i.e spam is less than ham


# In[19]:


pip install nltk


# In[23]:


import nltk


# In[24]:


nltk.download('punkt')


# In[25]:


df['num_characters'] = df['text'].apply(len)


# In[26]:


df.head()


# In[27]:


# num of words
df['num_words'] = df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[28]:


df.head()


# In[29]:


df['num_sentences'] = df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[30]:


df.head()


# In[31]:


df[['num_characters','num_words','num_sentences']].describe()


# In[32]:


# ham
df[df['target'] == 0][['num_characters','num_words','num_sentences']].describe()


# In[33]:


#spam
df[df['target'] == 1][['num_characters','num_words','num_sentences']].describe()


# In[34]:


import seaborn as sns


# In[35]:


sns.histplot(df[df['target']==0]['num_characters'])
sns.histplot(df[df['target']==1]['num_characters'],color='red')


# In[36]:


sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='red')


# In[37]:


#relationships no.ofwords with no.ofsentences
sns.pairplot(df,hue='target')


# In[38]:


numeric_df = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr() 
sns.heatmap(correlation_matrix, annot=True)


# # 3)Data Preprocessing
#    i)   Lower case
#    ii)  Tokenization(words split)
#    iii) Removing special characters
#    iv)  Removing stop words and punctution
#    v)   stemming (dance,dancing,danced->dance)

# In[58]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[59]:


df['text'][10]


# In[60]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string



# In[61]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('loving')


# In[62]:


df['transformed_text']=df['text'].apply(transform_text)


# In[44]:


df.sample(5)


# In[45]:


pip install wordcloud


# In[63]:


from wordcloud import WordCloud
wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')


# In[64]:


spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))


# In[65]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[66]:


ham_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))


# In[67]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[68]:


df.sample(5)


# In[69]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[70]:


len(spam_corpus)


# In[71]:


from collections import Counter

# Assuming spam_corpus is a list of words or tokens
word_counts = Counter(spam_corpus).most_common(30)

# Create a DataFrame from the word counts
word_counts_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

# Plot the barplot
sns.barplot(x='Word', y='Count', data=word_counts_df)
plt.xticks(rotation='vertical')
plt.show()


# In[72]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[73]:


len(ham_corpus)


# In[74]:


from collections import Counter

# Assuming spam_corpus is a list of words or tokens
word_counts = Counter(ham_corpus).most_common(30)

# Create a DataFrame from the word counts
word_counts_df = pd.DataFrame(word_counts, columns=['Word', 'Count'])

# Plot the barplot
sns.barplot(x='Word', y='Count', data=word_counts_df)
plt.xticks(rotation='vertical')
plt.show()


# # 4) Model Building
# 

# In[167]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score
import pickle


# In[168]:


# Loading data and defining the target and features
X = df['transformed_text']
y = df['target']


# In[169]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[170]:


# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[171]:


# Defining classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'SVM': SVC(kernel='sigmoid', gamma=1.0)
}


# In[172]:


# Training and evaluating classifiers
results = []
for name, clf in classifiers.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    results.append({'Algorithm': name, 'Accuracy': accuracy, 'Precision': precision})


# In[180]:


# Displaying results
results_df = pd.DataFrame(results)
print(results_df)


# In[174]:


# Saving the TF-IDF vectorizer and best performing classifier
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
best_classifier_name = results_df.loc[results_df['Precision'].idxmax(), 'Algorithm']
best_classifier = classifiers[best_classifier_name]
pickle.dump(best_classifier, open('model.pkl', 'wb'))


# In[186]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(SVC,open('model.pkl','wb'))


# In[ ]:




