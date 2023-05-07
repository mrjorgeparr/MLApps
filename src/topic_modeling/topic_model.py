import csv
import os
import json
from gensim.models import LsiModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel


with open("data\preprocessed_data.csv", 'r', encoding= "utf8") as file:
  data=[]
  csvreader = csv.DictReader(file)
  for row in csvreader:
    data.append(row["cleaned_text"])

dataset= []

for item in data:
    parsed_item = json.loads(item.replace("'",'"'))
    dataset.append(parsed_item)


no_below = 4
no_above = .80 

D = Dictionary(dataset)
D.filter_extremes(no_below=no_below,no_above=no_above)

## Get the TFIDF of the corpus
mycorpus_bow = [D.doc2bow(doc) for doc in dataset]
tfidf = TfidfModel(mycorpus_bow)
mycorpus_tfidf = tfidf[mycorpus_bow]

## Topic modeling
num_topics=10


##Topic modeling using LDA
ldag = LdaModel(corpus=mycorpus_bow, id2word=D, num_topics=num_topics)
fig, axes = plt.subplots(2, 5, figsize=(16, 10), sharex=True)
for i in range(5):
    list_data=[[item[0], item[1]] for item in ldag.show_topic(i, topn)]
    df=pd.DataFrame(list_data, columns=['Token', 'Weight'])
    sns.barplot(x='Weight', y='Token', data=df, color='c', orient='h', ax=axes[0][i])
    axes[0][i].set_title('Topic ' + str(i))
    list_data2=[[item[0], item[1]] for item in ldag.show_topic(i+4, topn)]
    df=pd.DataFrame(list_data2, columns=['Token', 'Weight'])
    sns.barplot(x='Weight', y='Token', data=df, color='c', orient='h', ax=axes[1][i])
    axes[1][i].set_title('Topic ' + str(i+5))    


## Evaluating the coherence of LDA models
n_topics=[5,10,15,20,25,50]
list_coherence=[]
for i in n_topics:
   ldag=LdaModel(corpus=mycorpus_bow, id2word=D, num_topics=i)
   if __name__=='main':    
      coherencemodel = CoherenceModel(ldag, texts=dataset, dictionary=D, coherence='c_v')
      list_coherence.append(coherencemodel.get_coherence())

## Results
n_topics=[5,10,15,20,25,50]
list_coherence=[0.2982752252963977, 0.32734188430371663, 0.3208099372534216, 0.3589756322369969, 0.33815754553315786, 0.3486448266421634]
plt.plot(n_topics, list_coherence)
plt.xlabel('number of topics')
plt.ylabel('coherence')
plt.show()




