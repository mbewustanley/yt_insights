import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import CountVectorizer

df = pd.read_csv('https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')
df.head()

df.shape()
df.sample()['clean_comment'].values
df.info()
df.isnull().sum()
df[df['clean_comment'].isna()]
df[df['clean_comment'].isna()].value_counts()

df.dropna(inplace=True)

df.duplicated().sum()  # how many rows have diplicated values
df[df.duplicated()]  # show
df.drop_duplicates(inplace=True)

df[(df['clean_comment'].str.strip() == ' ')]  # show how many rows have new lines
df = df[~(df['clean_comment'].str.strip() == ' ')]  # remove them

df['clean_comment'] = df['clean_comment'].str.lower() #convert all text in column to lowercase

df[df['clean_comment'].apply(lambda x: x.endswith(' ') or x.startswith(' '))]  # show rows with empty space at beginning or end of text
df['clean_comment'] = df['clean_comment'].str.strip() # remove trailing or leading whitespaces


# identify comments containing url
url_pattern = r'http[s]?://(?:[a-zA-Z]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
comments_with_urls = df[df['clean_comment'].str.contains(url_pattern, regex=True)]

# display the comments containing urls
comments_with_urls.head()

comments_with_newline = df[df['clean_comment'].str.contains('\n')]
comments_with_newline.head()

df['clean_comment'] = df['clean_comment'].str.replace('\n', ' ', regex=True)





# Exploratory Data Analysis

sns.countplot(data=df, x='category')  #shows an imbalanced data
df['category'].value_counts(normalize=True).mul(100).round(2) #show the distribution of sentiments

df['word_count'] = df['clean_comment'].apply(lambda x: len(x.split())) # create new feature showing the word count
df['word_count'].describe()

sns.displot(df['word_count'], kde=True)

plt.figure(figsize=(10,6))
#plot KDE for category 1
sns.kdeplot(df[df['category']== 1]['word_count'], label='positive', fill=True)
#plot KDE for category 0
sns.kdeplot(df[df['category']== 0]['word_count'], label='Neutral', fill=True)
#plot KDE for category -1
sns.kdeplot(df[df['category']== -1]['word_count'], label='Negative', fill=True)

plt.title('Word count Distribution by category')
plt.xlabel('Word count')
plt.ylabel('Density')
plt.legend()
plt.show()

sns.boxplot(df['word_count'])
#create boxplot for wordcount categorized by category
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='category', y='word_count')
plt.title('Boxplot of word count by Category')
plt.xlabel('Category')
plt.ylabel('Word count')
plt.show()

#create scatterplot for wordcount categorized by category
plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='category', y='word_count', alpha=0.5)
plt.title('Scatterplot of word count by Category')
plt.xlabel('Category')
plt.ylabel('Word count')
plt.show()

#median word counts among sentiments
sns.barplot(df, x='category', y='word_count', estimator='median')



# lets use nltk - natural language token library
# download NLTK stopwords
nltk.download('stopwords')

# define the list of english stopwords
stop_word = set(stopwords.words('english'))

# create new column 'num_stop_words' by counting the number of stopwords in each comment
df['num_stop_words'] = df['clean_comment'].apply(lambda x: len([word for word in x.split() if word in stop_word]))

#create a distribution plot (histplot) for the 'num_stop_words column
plt.figure(figsize=(10,6))
sns.histplot(df['num_stop_words'], kde=True)
plt.title('Distribution of stopwords count in comments')
plt.xlabel('Number of stop words')
plt.ylabel('Frequency')
plt.show()

# create the figures and axes
plt.figure(figsize=(10,6))
#plot KDE for category 1
sns.kdeplot(df[df['category']== 1]['num_stop_words'], label='positive', fill=True)
#plot KDE for category 0
sns.kdeplot(df[df['category']== 0]['num_stop_words'], label='Neutral', fill=True)
#plot KDE for category -1
sns.kdeplot(df[df['category']== -1]['num_stop_words'], label='Negative', fill=True)

# Add titles and labels
plt.title('Num stop words Distribution by category')
plt.xlabel('Stop word count')
plt.ylabel('Density')

plt.legend()
plt.show()

#median stopword counts among sentiments
sns.barplot(df, x='category', y='num_stop_words', estimator='median')



# Most used stop words
# create a frequency distribution of stopwords in the clean_comment column
# extract all stopwords from the comments using previously defined 'common_stopwords
all_stop_words = [word for comment in df['clean_comment'] for word in comment.split() if word in stop_word]

#count most common stopwords
most_common_stop_words = Counter(all_stop_words).most_common(25)

#convert the most common stop words to a dataframe for plotting
top_25_df = pd.DataFrame(most_common_stop_words, columns=['stop_word', 'count'])

#create a bar plot 
plt.figure(figsize=(10,6))
plt.barplot(data=top_25_df, x='count', y='stop_word', palette='viridis')
plt.title('Top 25 Most Common Stop Words')
plt.xlabel('Count')
plt.ylabel('Stop_word')
plt.show()


# create a feature for number of characters in clean comment
df['num_char'] = df['clean_comment'].apply(len)

df['num_char'].describe()



# To see the special characters we have using counter from collections
#combine elements into one large string
all_text =' '.join(df['clean_comment'])

#count the frequency of each character
char_frequency = Counter(all_text)

#convert the char frequency into a dataframe for better display
char_frequency_df = pd.DataFrame(char_frequency.items(), columns=['character', 'frequency']).sort_values(by='frequency')

char_frequency_df['character'].values


# Create a new column 'num_punctuation_chars' to count punctuation characters in each comment
df['num_punctuation_chars'] = df['clean_comment'].apply(lambda x: sum([1 for char in x if char in '.,!?;:"\'()[]{}-']))




#Vectorization
# using ngrams
# create a function to extract the top 25 bigrams
def get_top_bigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# get top 25 bigrams
top_25_bigrams = get_top_bigrams(df['clean_comment'], 25)

# convert to dataframe for plotting
top_25_bigrams_df = pd.DataFrame(top_25_bigrams, columns=['bigram', 'count'])

#plot the countplot for the top 25 bigrams
plt.figure(figsize=(10,6))
sns.barplot(data=top_25_bigrams_df, x='count', y='bigram', palette='magma')
plt.title('Top 25 Most Common Bigrams')
plt.ylabel('bigram')
plt.xlabel('count')
plt.show()


# create a function to extract the top 25 trigrams
def get_top_trigrams(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3,3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word,idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:n]

# get top 25 bigrams
top_25_trigrams = get_top_trigrams(df['clean_comment'], 25)

# convert to dataframe for plotting
top_25_trigrams_df = pd.DataFrame(top_25_trigrams, columns=['trigram', 'count'])

#plot the countplot for the top 25 bigrams
plt.figure(figsize=(10,6))
sns.barplot(data=top_25_trigrams_df, x='count', y='trigram', palette='magma')
plt.title('Top 25 Most Common trigrams')
plt.ylabel('trigram')
plt.xlabel('count')
plt.show()


#Remove non english characters from the 'clean_comment' column
#Keeping only standard english letters, digits, and common punctuations
import re

df['clean_comment'] = df['clean_comment'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))

all_txt = ' '.join(df['clean_comment'])

#count the frequency of each character
char_freq = Counter(all_text)

#convert the character frequency to a dataframe for better display
char_freq_df = pd.DataFrame(char_freq.items(), columns=['character', 'frequency']).sort_values(by='frequency')



#Defining stopwords but keeping essential ones for sentimental analysis
stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}

#remove stop words from 'clean_comment' column, retaining essential ones
df['clean_comment'] = df['clean_comment'].apply(
    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words])
)


# Adding Lemmatizer
from nltk.stem import WordNetLemmatizer

#define the lemmatizer
lemmatizer = WordNetLemmatizer()

#apply lemmatization to the 'clean_comment_no_stop_words' column
df['clean_comment'] = df['clean_comment'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()])
) # this reduces dimensionality also


#

# Implementing Word Cloud
from wordcloud import WordCloud

def plot_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show

plot_word_cloud(df['clean_comment'])


#plot top n words
def plot_top_n_words(df, n=20):
    #flatten all words in the content column
    words = ' '.join(df['clean_comment']).split()

    #get top n common words
    counter = Counter(words)
    most_common_words = counter.most_common(n)

    #split the words into counts for plotting
    words, counts = zip(*most_common_words)

    #plot the top n words
    plt.figure(figsize=(10,6))
    sns.barplot(x=list(counts), y=list(words))
    plt.title(f'Top {n} Most Frequent Words' )
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

plot_top_n_words(df, n=50)



# plot top n words by category
def plot_top_n_words_by_category(df, n=20, start=0):  
    #flatten all words in content column and count their occurence by category
    word_category_counts = {}

    for idx, row in df.iterrows():
        words = row['clean_comment'].split()
        category = row['category']

        for word in words:
            if word not in word_category_counts:
                word_category_counts[word] = {-1:0, 0:0, 1:0}  #initialize counts for each sentiment category

            # increment the count for the corresponding sentiment category
            word_category_counts[word][category] +=1
    
    # get total counts across all categories for each word
    total_word_counts = {word: sum(counts.values()) for word,counts in word_category_counts.items()}

    # get the top N most frequent words across all categories
    most_common_words = sorted(total_word_counts.items(), key=lambda x: x[1], reverse=True)[start:start+n]
    top_words = [word for word, _ in most_common_words]

    #prepare data for plotting
    word_labels = top_words
    negative_counts = [word_category_counts[word][-1] for word in top_words]
    neutral_counts = [word_category_counts[word][0] for word in top_words]
    positive_count = [word_category_counts[word][1] for word in top_words]

    #plot the stacked bar
    plt.figure(figsize=(12,8))
    bar_width = 0.75

    #plot the negative, neutral, and positive counts in a stacked manner
    plt.barh(word_labels, negative_counts, color='red', label='Negative(-1)', height=bar_width)
    plt.barh(word_labels, neutral_counts, left=negative_counts, color='grey', label='Neutral(0)', height=bar_width)
    plt.barh(word_labels, positive_count, left=[i+j for i,j in zip(negative_counts, neutral_counts)], color='green', label='Positive(1)', height=bar_width)

    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.title(f'Top {n} Most Frequent Words with Stacked Sentiment Categories')
    plt.legend()
    plt.gca().invert_yaxis()   #invert y axis to show the highest frequency at the top
    plt.show()

plot_top_n_words_by_category(df, 20)