# Creating a Sentiment Analysis Web App
## Using PyTorch and SageMaker

_Deep Learning Nanodegree Program | Deployment_

Now that we have a basic understanding of how SageMaker works we will try to use it to construct a
complete project from end to end. Our goal will be to have a simple web page which a user can use to
enter a movie review. The web page will then send the review off to our deployed model which will
predict the sentiment of the entered review.

## Instructions

Some template code has already been provided for you, and you will need to implement additional
functionality to successfully complete this notebook. You will not need to modify the included code
beyond what is requested. Sections that begin with '**TODO**' in the header indicate that you need
to complete or implement some portion within them. Instructions will be provided for each section
and the specifics of the implementation are marked in the code block with a `# TODO: ...` comment.
Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions for you to answer which relate to the task
and your implementation. Each section where you will answer a question is preceded by a
'**Question:**' header. Carefully read each question and provide your answer below the '**Answer:**'
header by editing the Markdown cell.

> **Note**: Code and Markdown cells can be executed using the **Shift+Enter** keyboard shortcut. In
  addition, a cell can be edited by typically clicking it (double-click for Markdown cells) or by
  pressing **Enter** while it is highlighted.

## General Outline

Recall the general outline for SageMaker projects using a notebook instance.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

For this project, you will be following the steps in the general outline with some modifications. 

First, you will not be testing the model in its own step. You will still be testing the model,
however, you will do it by deploying your model and then using the deployed model by sending the
test data to it. One of the reasons for doing this is so that you can make sure that your deployed
model is working correctly before moving forward.

In addition, you will deploy and use your trained model a second time. In the second iteration you
will customize the way that your trained model is deployed by including some of your own code. In
addition, your newly deployed model will be used in the sentiment analysis web app.

## Step 1: Downloading the data

As in the XGBoost in SageMaker notebook, we will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). 
  In _Proceedings  of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_.
  Association for Computational Linguistics, 2011.

```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

    mkdir: cannot create directory ‘../data’: File exists
    --2020-03-30 22:59:10--  http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    Resolving ai.stanford.edu (ai.stanford.edu)... 171.64.68.10
    Connecting to ai.stanford.edu (ai.stanford.edu)|171.64.68.10|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 84125825 (80M) [application/x-gzip]
    Saving to: ‘../data/aclImdb_v1.tar.gz’
    
    ../data/aclImdb_v1. 100%[===================>]  80.23M  27.0MB/s    in 3.0s    
    
    2020-03-30 22:59:13 (27.0 MB/s) - ‘../data/aclImdb_v1.tar.gz’ saved [84125825/84125825]

## Step 2: Preparing and Processing the data

Also, as in the XGBoost notebook, we will be doing some initial data processing. The first few steps
are the same as in the XGBoost example. To begin with, we will read in each of the reviews and
combine them into a single input structure. Then, we will split the dataset into a training set and
a testing set.

```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels
```

```python
data, labels = read_imdb_data()
print("IMDB reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
            len(data['train']['pos']), len(data['train']['neg']),
            len(data['test']['pos']), len(data['test']['neg'])))
```

    IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg

Now that we've read the raw training and testing data from the downloaded dataset, we will combine
the positive and negative reviews and shuffle the resulting records.

```python
from sklearn.utils import shuffle

def prepare_imdb_data(data, labels):
    """Prepare training and test sets from IMDb movie reviews."""
    
    # Combine positive and negative reviews and labels
    data_train = data['train']['pos'] + data['train']['neg']
    data_test = data['test']['pos'] + data['test']['neg']
    labels_train = labels['train']['pos'] + labels['train']['neg']
    labels_test = labels['test']['pos'] + labels['test']['neg']
    
    # Shuffle reviews and corresponding labels within training and test sets
    data_train, labels_train = shuffle(data_train, labels_train)
    data_test, labels_test = shuffle(data_test, labels_test)
    
    # Return a unified training data, test data, training labels, test labets
    return data_train, data_test, labels_train, labels_test
```

```python
train_X, test_X, train_y, test_y = prepare_imdb_data(data, labels)
print("IMDb reviews (combined): train = {}, test = {}".format(len(train_X), len(test_X)))
```

    IMDb reviews (combined): train = 25000, test = 25000

Now that we have our training and testing sets unified and prepared, we should do a quick check and
see an example of the data our model will be trained on. This is generally a good idea as it allows
you to see how each of the further processing steps affects the reviews and it also ensures that the
data has been loaded correctly.

```python
print("X Training Example:", train_X[100], "\n")
print("Label Example:", train_y[100])
```

    X Training Example: The first thing you meet when you study fascism is ostracism:because this" philosophy " is a fake one,there's a need to use scapegoats to assess the "thought".Ettore Scola's movie,probably his masterpiece, focuses on the outcasts,the scapegoats of the regime.<br /><br />Of the historical event (Hitler and Mussolini's alliance),we will see almost nothing:some military march,some garlands,some scattered voices ..Our two heroes are not invited for the feast of virility. "Genius is essentially masculine" :this is the golden rule Antonietta (a never better Sophia Loren)embroidered on her cushion;Antonietta ,whose world amounts to her kitchen,whose pride is her offsprings .At the beginning of the movie,she's a victim of this hypermacho world,but she does not realize it.She thinks she should be happy.Gabriel,on the contrary ,is politically aware,he knows about the cancer that is destroying inexorably his country.But as a gay man,he is no longer part of it,he's about to be arrested.<br /><br />Forgetting everything that comes between them,they realize what they have in common and they make love.This is an act of rebellion,particularly for Antonietta ,whose ethic should forbid such a thing.Becoming an adulteress in a land where politics and religion combine to repress women as ever leads her to some kind of political awareness.One of the last shots shows her listening to the news on the radio.<br /><br />Expect the unexpected and maybe a doctrine which denies the human being his intimate personality will see that its days are numbered. 
    
    Label Example: 1

The first step in processing the reviews is to make sure that any html tags that appear should be
removed. In addition we wish to tokenize our input, that way words such as *entertained* and
*entertaining* are considered the same with regard to sentiment analysis.

```python
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *

import re
from bs4 import BeautifulSoup

def review_to_words(review):
    nltk.download("stopwords", quiet=True)
    stemmer = PorterStemmer()
    
    text = BeautifulSoup(review, "html.parser").get_text() # Remove HTML tags
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) # Convert to lower case
    words = text.split() # Split string into words
    words = [w for w in words if w not in stopwords.words("english")] # Remove stopwords
    words = [PorterStemmer().stem(w) for w in words] # stem
    
    return words
```

The `review_to_words` method defined above uses `BeautifulSoup` to remove any html tags that appear
and uses the `nltk` package to tokenize the reviews. As a check to ensure we know how everything is
working, try applying `review_to_words` to one of the reviews in the training set.

```python
# TODO: Apply review_to_words to a review (train_X[100] or any other review)
print(review_to_words(train_X[100]))
```

    ['first', 'thing', 'meet', 'studi', 'fascism', 'ostrac', 'philosophi', 'fake', 'one', 'need', 'use', 'scapegoat', 'assess', 'thought', 'ettor', 'scola', 'movi', 'probabl', 'masterpiec', 'focus', 'outcast', 'scapegoat', 'regim', 'histor', 'event', 'hitler', 'mussolini', 'allianc', 'see', 'almost', 'noth', 'militari', 'march', 'garland', 'scatter', 'voic', 'two', 'hero', 'invit', 'feast', 'viril', 'geniu', 'essenti', 'masculin', 'golden', 'rule', 'antonietta', 'never', 'better', 'sophia', 'loren', 'embroid', 'cushion', 'antonietta', 'whose', 'world', 'amount', 'kitchen', 'whose', 'pride', 'offspr', 'begin', 'movi', 'victim', 'hypermacho', 'world', 'realiz', 'think', 'happi', 'gabriel', 'contrari', 'polit', 'awar', 'know', 'cancer', 'destroy', 'inexor', 'countri', 'gay', 'man', 'longer', 'part', 'arrest', 'forget', 'everyth', 'come', 'realiz', 'common', 'make', 'love', 'act', 'rebellion', 'particularli', 'antonietta', 'whose', 'ethic', 'forbid', 'thing', 'becom', 'adulteress', 'land', 'polit', 'religion', 'combin', 'repress', 'women', 'ever', 'lead', 'kind', 'polit', 'awar', 'one', 'last', 'shot', 'show', 'listen', 'news', 'radio', 'expect', 'unexpect', 'mayb', 'doctrin', 'deni', 'human', 'intim', 'person', 'see', 'day', 'number']

**Question:** Above we mentioned that `review_to_words` method removes html formatting and allows us
to tokenize the words found in a review, for example, converting *entertained* and *entertaining* into
*entertain* so that they are treated as though they are the same word. What else, if anything, does
this method do to the input?

**Answer:** Let's go through the logic of the function line by line

* BeautifulSoup uses HTML parser to remove HTML tags
* The text is split into word array or list in the context of Python
* Each stop word is removed from the list, e.g. who, which, whom, this, that and etc...
* PorterStemmer removes morphological affixes from words, leaving only the word stem. Hence, entertained and
  entertaining are reduced down to entertain.

Then finally, the reduced list of words is returned from the function

The method below applies the `review_to_words` method to each of the reviews in the training and
testing datasets. In addition it caches the results. This is because performing this processing step
can take a long time. This way if you are unable to complete the notebook in the current session,
you can come back without needing to process the data a second time.

```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
```

```python
# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)
```

    Read preprocessed data from cache file: preprocessed_data.pkl

## Transform the data

In the XGBoost notebook we transformed the data from its word representation to a bag-of-words
feature representation. For the model we are going to construct in this notebook we will construct a
feature representation which is very similar. To start, we will represent each word as an integer.
Of course, some of the words that appear in the reviews occur very infrequently and so likely don't
contain much information for the purposes of sentiment analysis. The way we will deal with this
problem is that we will fix the size of our working vocabulary and we will only include the words
that appear most frequently. We will then combine all of the infrequent words into a single category
and, in our case, we will label it as `1`.

Since we will be using a recurrent neural network, it will be convenient if the length of each
review is the same. To do this, we will fix a size for our reviews and then pad short reviews with
the category 'no word' (which we will label `0`) and truncate long reviews.

### (TODO) Create a word dictionary

To begin with, we need to construct a way to map words that appear in the reviews to integers. Here
we fix the size of our vocabulary (including the 'no word' and 'infrequent' categories) to be `5000`
but you may wish to change this to see how it affects the model.

> **TODO:** Complete the implementation for the `build_dict()` method below. Note that even though
  the vocab_size is set to `5000`, we only want to construct a mapping for the most frequently
  appearing `4998` words. This is because we want to reserve the special labels `0` for 'no word'
  and `1` for 'infrequent word'.

```python
import numpy as np

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    for sentence in data:
        for word in sentence:
            if not word in word_count:
                word_count[word] = 0
            word_count[word] += 1
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.

    sorted_words = [word for word, count in sorted(word_count.items(), key=lambda pair: pair[1])]
    sorted_words.reverse()

    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict
```

```python
word_dict = build_dict(train_X)
```

**Question:** What are the five most frequently appearing (tokenized) words in the training set?
Does it makes sense that these words appear frequently in the training set?

**Answer:** Since dictionary in Python 3.6+ has order property. We can see the most frequent
appearing by printing the dictionary or look for words with index value [2, 3, 4, 5, 6].

```python
# TODO: Use this space to determine the five most frequently appearing words in the training set.
print(list(filter(lambda key: word_dict[key] <= 6, word_dict.keys())))
```

    ['movi', 'film', 'one', 'like', 'time']

### Save `word_dict`

Later on when we construct an endpoint which processes a submitted review we will need to make use
of the `word_dict` which we have created. As such, we will save it to a file now for future use.

```python
data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)
```

```python
with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)
```

### Transform the reviews

Now that we have our word dictionary which allows us to transform the words appearing in the reviews
into integers, it is time to make use of it and convert our reviews to their integer sequence
representation, making sure to pad or truncate to a fixed length, which in our case is `500`.

```python
def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)
```

```python
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
```

As a quick check to make sure that things are working as intended, check to see what one of the
reviews in the training set looks like after having been processeed. Does this look reasonable? What
is the length of a review in the training set?

```python
# Use this cell to examine one of the processed reviews to make sure everything is working as intended.
print(train_X[1])
print(train_X_len[1])
```

    [ 140 3990 2436    1    1    1    1 2019   37  717  211  136  211 1126
       37  458 2307  156   14   19   64 1869 4673 2348  131    6   79    1
     2067   58  411  566 3485    1   31    1  128    1  252  694 2916 3438
     2386    1   50  526   17 1011  526 1676  145 1391  365   49 1592  526
        1 3990    1    1  122  255   49   40  177 1186  132 1811  345 3415
      122   31   87 3414  194 1765    1   48 2490 2643   25  507 1898 1265
     1726 3990 4724 1164 1475   71  122   17    1   55  324    1 3960  620
       98  246  426  633 2489    1  207  294   87  193 1252   10 1265  630
     1879 3990   69  632  193  765  380   11  791 1050 2616   13 1856  420
      849 1400 3705  864 1855  633  815 2828  469   92  859  825 1055   61
       79  330  825    4  574 2265   79    1    1  609   72 4213  989 2616
      163 3990  551  572  706  574  633   11    1 1101 3990  989 2314  171
     1122   14   92  986 1093    3 1160    1  161 2564  856 1010 3990  849
      273 1201   17  127   13 1395 3485   57  685 2470  302   98  849  162
      665 4672  383   65 2935 2219 4471 1646   55   92   42  217    9  899
      486   31  129  633 3990  666   81    1 2061  189   53  426   14    8
       53 1430  314   31  526   92 3990 2161 1798    1  526 1676    1  226
        1  129 1200 2385  384   11  242    4  464  196 3485   11 3990    1
        1  273 3704   65  196  526   23   38   27   16  152    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0]
    263

**Question:** In the cells above we use the `preprocess_data` and `convert_and_pad_data` methods to
process both the training and testing set. Why or why not might this be a problem?

**Answer:** It should not be a problem because we want all data that go into the RNN to have the
same format and dimension.

## Step 3: Upload the data to S3

As in the XGBoost notebook, we will need to upload the training dataset to S3 in order for our
training code to access it. For now we will save it locally and we will upload to S3 later on.

### Save the processed training dataset locally

It is important to note the format of the data that we are saving as we will need to know it when we
write the training code. In our case, each row of the dataset has the form `label`, `length`,
`review[500]` where `review[500]` is a sequence of `500` integers representing the words in the
review.

```python
import pandas as pd

# CSV: label, sentence length, word_1, word_2, word_3, ... no_word
pd.concat([pd.DataFrame(train_y), pd.DataFrame(train_X_len), pd.DataFrame(train_X)], axis=1) \
        .to_csv(os.path.join(data_dir, 'train.csv'), header=False, index=False)
```

### Uploading the training data

Next, we need to upload the training data to the SageMaker default S3 bucket so that we can provide
access to it while training our model.

```python
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'projects/rnn_sentiment_analysis'

role = sagemaker.get_execution_role()
```

```python
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

**NOTE:** The cell above uploads the entire contents of our data directory. This includes the
`word_dict.pkl` file. This is fortunate as we will need this later on when we create an endpoint
that accepts an arbitrary review. For now, we will just take note of the fact that it resides in the
data directory (and so also in the S3 training bucket) and that we will need to make sure it gets
saved in the model directory.

## Step 4: Build and Train the PyTorch Model

In the XGBoost notebook we discussed what a model is in the SageMaker framework. In particular, a
model comprises three objects

 - Model Artifacts,
 - Training Code, and
 - Inference Code,
 
each of which interact with one another. In the XGBoost example we used training and inference code
that was provided by Amazon. Here we will still be using containers provided by Amazon with the
added benefit of being able to include our own custom code.

We will start by implementing our own neural network in PyTorch along with a training script. For
the purposes of this project we have provided the necessary model object in the `model.py` file,
inside of the `train` folder. You can see the provided implementation by running the cell below.

```python
!pygmentize train/model.py
```

The important takeaway from the implementation provided is that there are three parameters that we
may wish to tweak to improve the performance of our model. These are the embedding dimension, the
hidden dimension and the size of the vocabulary. We will likely want to make these parameters
configurable in the training script so that if we wish to modify them we do not need to modify the
script itself. We will see how to do this later on. To start we will write some of the training code
in the notebook so that we can more easily diagnose any issues that arise.

First we will load a small portion of the training data set to use as a sample. It would be very
time consuming to try and train the model completely in the notebook as we do not have access to a
gpu and the compute instance that we are using is not particularly powerful. However, we can work
on a small bit of the data to get a feel for how our training script is behaving.

```python
import torch
import torch.utils.data

# Read in only the first 250 rows
train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
```

### (TODO) Writing the training method

Next we need to write the training code itself. This should be very similar to training methods that
you have written before to train PyTorch models. We will leave any difficult aspects such as model
saving / loading and parameter loading until a little later.

```python
def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad() # Zero the parameter gradients
            outputs = model.forward(batch_X) # Forward pass the inputs
            loss = loss_fn(outputs, batch_y) # Compute loss
            loss.backward() # Compute the derivatives or gradients
            optimizer.step() # Perform update
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
```

Supposing we have the training method above, we will test that it is working by writing a bit of
code in the notebook that executes our training method on the small sample training set that we
loaded earlier. The reason for doing this in the notebook is so that we have an opportunity to fix
any errors that arise early when they are easier to diagnose.

```python
import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 5, optimizer, loss_fn, device)
```

    Epoch: 1, BCELoss: 0.69208242893219
    Epoch: 2, BCELoss: 0.6806374549865722
    Epoch: 3, BCELoss: 0.6703490257263184
    Epoch: 4, BCELoss: 0.6590749740600585
    Epoch: 5, BCELoss: 0.6456955313682556

In order to construct a PyTorch model using SageMaker we must provide SageMaker with a training
script. We may optionally include a directory which will be copied to the container and from which
our training code will be run. When the training container is executed it will check the uploaded
directory (if there is one) for a `requirements.txt` file and install any required Python libraries,
after which the training script will be run.

### (TODO) Training the model

When a PyTorch model is constructed in SageMaker, an entry point must be specified. This is the
Python file which will be executed when the model is trained. Inside of the `train` directory is a
file called `train.py` which has been provided and which contains most of the necessary code to
train our model. The only thing that is missing is the implementation of the `train()` method which
you wrote earlier in this notebook.

**TODO**: Copy the `train()` method written above and paste it into the `train/train.py` file where
required.

The way that SageMaker passes hyperparameters to the training script is by way of arguments. These
arguments can then be parsed and used in the training script. To see how this is done take a look
at the provided `train/train.py` file.

```python
from sagemaker.pytorch import PyTorch

rnn_estimator = PyTorch(entry_point="train.py",
                        source_dir="train",
                        role=role,
                        framework_version='0.4.0',
                        train_instance_count=1,
                        train_instance_type='ml.p2.xlarge',
                        hyperparameters={
                          'epochs': 10,
                          'hidden_dim': 200,
                        })
```

```python
rnn_estimator.fit({'training': input_data})
```

    2020-03-30 23:02:05 Starting - Starting the training job...
    2020-03-30 23:02:07 Starting - Launching requested ML instances......
    2020-03-30 23:03:05 Starting - Preparing the instances for training......
    2020-03-30 23:04:30 Downloading - Downloading input data......
    2020-03-30 23:05:14 Training - Downloading the training image.[34mbash: cannot set terminal process group (-1): Inappropriate ioctl for device[0m
    [34mbash: no job control in this shell[0m
    [34m2020-03-30 23:05:37,721 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training[0m
    [34m2020-03-30 23:05:37,746 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.[0m
    [34m2020-03-30 23:05:37,749 sagemaker_pytorch_container.training INFO     Invoking user training script.[0m
    [34m2020-03-30 23:05:37,996 sagemaker-containers INFO     Module train does not provide a setup.py. [0m
    [34mGenerating setup.py[0m
    [34m2020-03-30 23:05:37,997 sagemaker-containers INFO     Generating setup.cfg[0m
    [34m2020-03-30 23:05:37,997 sagemaker-containers INFO     Generating MANIFEST.in[0m
    [34m2020-03-30 23:05:37,997 sagemaker-containers INFO     Installing module with the following command:[0m
    [34m/usr/bin/python -m pip install -U . -r requirements.txt[0m
    [34mProcessing /opt/ml/code[0m
    [34mCollecting pandas (from -r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/74/24/0cdbf8907e1e3bc5a8da03345c23cbed7044330bb8f73bb12e711a640a00/pandas-0.24.2-cp35-cp35m-manylinux1_x86_64.whl (10.0MB)[0m
    [34mCollecting numpy (from -r requirements.txt (line 2))
      Downloading https://files.pythonhosted.org/packages/ff/18/c0b937e2f84095ae230196899e56d1d7d76c8e8424fb235ed7e5bb6d68af/numpy-1.18.2-cp35-cp35m-manylinux1_x86_64.whl (20.0MB)[0m
    [34mCollecting nltk (from -r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/f6/1d/d925cfb4f324ede997f6d47bea4d9babba51b49e87a767c170b77005889d/nltk-3.4.5.zip (1.5MB)[0m
    [34mCollecting beautifulsoup4 (from -r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/cb/a1/c698cf319e9cfed6b17376281bd0efc6bfc8465698f54170ef60a485ab5d/beautifulsoup4-4.8.2-py3-none-any.whl (106kB)[0m
    [34mCollecting html5lib (from -r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/a5/62/bbd2be0e7943ec8504b517e62bab011b4946e1258842bc159e5dfde15b96/html5lib-1.0.1-py2.py3-none-any.whl (117kB)[0m
    [34mRequirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.5/dist-packages (from pandas->-r requirements.txt (line 1)) (2.7.5)[0m
    [34mCollecting pytz>=2011k (from pandas->-r requirements.txt (line 1))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/e7/f9/f0b53f88060247251bf481fa6ea62cd0d25bf1b11a87888e53ce5b7c8ad2/pytz-2019.3-py2.py3-none-any.whl (509kB)[0m
    [34mRequirement already satisfied, skipping upgrade: six in /usr/local/lib/python3.5/dist-packages (from nltk->-r requirements.txt (line 3)) (1.11.0)[0m
    [34mCollecting soupsieve>=1.2 (from beautifulsoup4->-r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/05/cf/ea245e52f55823f19992447b008bcbb7f78efc5960d77f6c34b5b45b36dd/soupsieve-2.0-py2.py3-none-any.whl[0m
    [34mCollecting webencodings (from html5lib->-r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl[0m
    [34mBuilding wheels for collected packages: nltk, train
      Running setup.py bdist_wheel for nltk: started[0m
    [34m  Running setup.py bdist_wheel for nltk: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/96/86/f6/68ab24c23f207c0077381a5e3904b2815136b879538a24b483
      Running setup.py bdist_wheel for train: started
      Running setup.py bdist_wheel for train: finished with status 'done'
      Stored in directory: /tmp/pip-ephem-wheel-cache-04nyv165/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3[0m
    [34mSuccessfully built nltk train[0m
    [34mInstalling collected packages: numpy, pytz, pandas, nltk, soupsieve, beautifulsoup4, webencodings, html5lib, train
      Found existing installation: numpy 1.15.4
        Uninstalling numpy-1.15.4:[0m
    [34m      Successfully uninstalled numpy-1.15.4[0m
    [34mSuccessfully installed beautifulsoup4-4.8.2 html5lib-1.0.1 nltk-3.4.5 numpy-1.18.2 pandas-0.24.2 pytz-2019.3 soupsieve-2.0 train-1.0.0 webencodings-0.5.1[0m
    [34mYou are using pip version 18.1, however version 20.0.2 is available.[0m
    [34mYou should consider upgrading via the 'pip install --upgrade pip' command.[0m
    [34m2020-03-30 23:05:50,370 sagemaker-containers INFO     Invoking user script
    [0m
    [34mTraining Env:
    [0m
    [34m{
        "framework_module": "sagemaker_pytorch_container.training:main",
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training"
        },
        "output_dir": "/opt/ml/output",
        "network_interface_name": "eth0",
        "output_data_dir": "/opt/ml/output/data",
        "additional_framework_parameters": {},
        "input_config_dir": "/opt/ml/input/config",
        "resource_config": {
            "hosts": [
                "algo-1"
            ],
            "network_interface_name": "eth0",
            "current_host": "algo-1"
        },
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "hosts": [
            "algo-1"
        ],
        "module_dir": "s3://sagemaker-us-west-2-171758673694/sagemaker-pytorch-2020-03-30-23-02-04-514/source/sourcedir.tar.gz",
        "num_cpus": 4,
        "hyperparameters": {
            "epochs": 10,
            "hidden_dim": 200
        },
        "num_gpus": 1,
        "user_entry_point": "train.py",
        "input_dir": "/opt/ml/input",
        "current_host": "algo-1",
        "job_name": "sagemaker-pytorch-2020-03-30-23-02-04-514",
        "module_name": "train",
        "input_data_config": {
            "training": {
                "RecordWrapperType": "None",
                "TrainingInputMode": "File",
                "S3DistributionType": "FullyReplicated"
            }
        },
        "model_dir": "/opt/ml/model",
        "log_level": 20[0m
    [34m}
    [0m
    [34mEnvironment variables:
    [0m
    [34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate[0m
    [34mSM_FRAMEWORK_PARAMS={}[0m
    [34mSM_INPUT_DIR=/opt/ml/input[0m
    [34mSM_USER_ARGS=["--epochs","10","--hidden_dim","200"][0m
    [34mSM_HP_HIDDEN_DIM=200[0m
    [34mSM_CHANNEL_TRAINING=/opt/ml/input/data/training[0m
    [34mSM_USER_ENTRY_POINT=train.py[0m
    [34mSM_CURRENT_HOST=algo-1[0m
    [34mSM_MODULE_DIR=s3://sagemaker-us-west-2-171758673694/sagemaker-pytorch-2020-03-30-23-02-04-514/source/sourcedir.tar.gz[0m
    [34mSM_HP_EPOCHS=10[0m
    [34mSM_OUTPUT_DIR=/opt/ml/output[0m
    [34mSM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [34mSM_MODEL_DIR=/opt/ml/model[0m
    [34mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"epochs":10,"hidden_dim":200},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","job_name":"sagemaker-pytorch-2020-03-30-23-02-04-514","log_level":20,"model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-west-2-171758673694/sagemaker-pytorch-2020-03-30-23-02-04-514/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train.py"}[0m
    [34mSM_NETWORK_INTERFACE_NAME=eth0[0m
    [34mSM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}[0m
    [34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [34mSM_HPS={"epochs":10,"hidden_dim":200}[0m
    [34mSM_NUM_GPUS=1[0m
    [34mSM_MODULE_NAME=train[0m
    [34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config[0m
    [34mSM_CHANNELS=["training"][0m
    [34mSM_HOSTS=["algo-1"][0m
    [34mSM_NUM_CPUS=4[0m
    [34mSM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main[0m
    [34mPYTHONPATH=/usr/local/bin:/usr/lib/python35.zip:/usr/lib/python3.5:/usr/lib/python3.5/plat-x86_64-linux-gnu:/usr/lib/python3.5/lib-dynload:/usr/local/lib/python3.5/dist-packages:/usr/lib/python3/dist-packages[0m
    [34mSM_LOG_LEVEL=20
    [0m
    [34mInvoking script with the following command:
    [0m
    [34m/usr/bin/python -m train --epochs 10 --hidden_dim 200
    
    [0m
    [34mUsing device cuda.[0m
    [34mGet train data loader.[0m
    
    2020-03-30 23:05:37 Training - Training image download completed. Training in progress.[34mModel loaded with embedding_dim 32, hidden_dim 200, vocab_size 5000.[0m
    [34mEpoch: 1, BCELoss: 0.6715459993907383[0m
    [34mEpoch: 2, BCELoss: 0.5961756791387286[0m
    [34mEpoch: 3, BCELoss: 0.5165451302820322[0m
    [34mEpoch: 4, BCELoss: 0.4454252525251739[0m
    [34mEpoch: 5, BCELoss: 0.40794419816562105[0m
    [34mEpoch: 6, BCELoss: 0.3654509551671086[0m
    [34mEpoch: 7, BCELoss: 0.3443656661072556[0m
    [34mEpoch: 8, BCELoss: 0.321930575735715[0m
    [34mEpoch: 9, BCELoss: 0.31921329182021474[0m
    [34mEpoch: 10, BCELoss: 0.3177856495793985[0m
    [34m2020-03-30 23:08:48,367 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2020-03-30 23:08:57 Uploading - Uploading generated training model
    2020-03-30 23:08:57 Completed - Training job completed
    Training seconds: 267
    Billable seconds: 267

## Step 5: Testing the model

As mentioned at the top of this notebook, we will be testing this model by first deploying it and
then sending the testing data to the deployed endpoint. We will do this so that we can make sure
that the deployed model is working correctly.

## Step 6: Deploy the model for testing

Now that we have trained our model, we would like to test it to see how it performs. Currently our
model takes input of the form `review_length, review[500]` where `review[500]` is a sequence of `500`
integers which describe the words present in the review, encoded using `word_dict`. Fortunately for
us, SageMaker provides built-in inference code for models with simple inputs such as this.

There is one thing that we need to provide, however, and that is a function which loads the saved
model. This function must be called `model_fn()` and takes as its only parameter a path to the
directory where the model artifacts are stored. This function must also be present in the python
file which we specified as the entry point. In our case the model loading function has been provided
and so no changes need to be made.

**NOTE**: When the built-in inference code is run it must import the `model_fn()` method from the
`train.py` file. This is why the training code is wrapped in a main guard ( ie, `if __name__ == '__main__':` )

Since we don't need to change anything in the code that was uploaded during training, we can simply
deploy the current model as-is.

**NOTE:** When deploying a model you are asking SageMaker to launch an compute instance that will
wait for data to be sent to it. As a result, this compute instance will continue to run until *you*
shut it down. This is important to know since the cost of a deployed endpoint depends on how long it
has been running for.

In other words **If you are no longer using a deployed endpoint, shut it down!**

**TODO:** Deploy the trained model.

```python
# TODO: Deploy the trained model
rnn_predictor = rnn_estimator.deploy(initial_instance_count=1, 
                                     instance_type='ml.m4.xlarge')
```

## Step 7 - Use the model for testing

Once deployed, we can read in the test data and send it off to our deployed model to get some
results. Once we collect all of the results we can determine how accurate our model is.

```python
test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)
```

```python
# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, rnn_predictor.predict(array))
    
    return predictions
```

```python
predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]
```

```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```

    0.8532

**Question:** How does this model compare to the XGBoost model you created earlier? Why might these
two models perform differently on this dataset? Which do *you* think is better for sentiment analysis?

**Answer:** It's slightly better than the hyparameter tuned XGBoost model but I suspect it could be
even better if I perform more hyperparameter tuning like increasing the embedding dimension. I also
believe that ReLU may be a better choice than sigmoid for the non-linear layer.

In general, RNN is more suitable for sentiment analysis because XGBoost tends to focus on phrases,
like associating positivity or negativity with words but RNN factors in temporal order. However,
tree model (XGBoost) does look at ordering as well, from root to leaf node, each tree node looks at
the input from left to right `X[0][0], X[0][1], x[0][2]...` but RNN has hidden vector that
propagates forward. Not to mention that we are using LSTM, this hidden vector can propagate even
further, up to 100 words from some studies. 

### (TODO) More testing

We now have a trained model which has been deployed and which we can send processed reviews to and
which returns the predicted sentiment. However, ultimately we would like to be able to send our
model an unprocessed review. That is, we would like to send the review itself as a string. For
example, suppose we wish to send the following review to our model.

```python
test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'
```

The question we now need to answer is, how do we send this review to our model?

Recall in the first section of this notebook we did a bunch of data processing to the IMDb dataset.
In particular, we did two specific things to the provided reviews.
 - Removed any html tags and stemmed the input
 - Encoded the review as a sequence of integers using `word_dict`
 
In order process the review we will need to repeat these two steps.

**TODO**: Using the `review_to_words` and `convert_and_pad` methods from section one, convert `test_review` into a numpy array `test_data` suitable to send to our model. Remember that our model expects input of the form `review_length, review[500]`.

```python
# TODO: Convert test_review into a form usable by the model and save the results in test_data
test_review_words = review_to_words(test_review)
test_data, test_data_len = convert_and_pad_data(word_dict, [test_review_words])
print("test data has length {}".format(test_data_len))
test_data = pd.concat([pd.DataFrame(test_data_len), pd.DataFrame(test_data)], axis=1)

from IPython.display import display
display(test_data) # We should expect to see 501 rows, with first element as sentence length.
```

    test data has length [20]

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20</td>
      <td>1</td>
      <td>1375</td>
      <td>49</td>
      <td>53</td>
      <td>3</td>
      <td>4</td>
      <td>878</td>
      <td>173</td>
      <td>392</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 501 columns</p>
</div>

Now that we have processed the review, we can send the resulting array to our model to predict the
sentiment of the review.

```python
rnn_predictor.predict(test_data)
```

    array(0.9551276, dtype=float32)

Since the return value of our model is close to `1`, we can be certain that the review we submitted
is positive.

### Delete the endpoint

Of course, just like in the XGBoost notebook, once we've deployed an endpoint it continues to run
until we tell it to shut down. Since we are done using our endpoint for now, we can delete it.

```python
rnn_estimator.delete_endpoint()
```

## Step 6 (again) - Deploy the model for the web app

Now that we know that our model is working, it's time to create some custom inference code so that
we can send the model a review which has not been processed and have it determine the sentiment of
the review.

As we saw above, by default the estimator which we created, when deployed, will use the entry script
and directory which we provided when creating the model. However, since we now wish to accept a
string as input and our model expects a processed review, we need to write some custom inference
code.

We will store the code that we write in the `serve` directory. Provided in this directory is the `model.py` file that we used to construct our model, a `utils.py` file which contains the `review_to_words` and `convert_and_pad` pre-processing functions which we used during the initial data processing, and `predict.py`, the file which will contain our custom inference code. Note also that `requirements.txt` is present which will tell SageMaker what Python libraries are required by our custom inference code.

When deploying a PyTorch model in SageMaker, you are expected to provide four functions which the SageMaker inference container will use.
 - `model_fn`: This function is the same function that we used in the training script and it tells SageMaker how to load our model.
 - `input_fn`: This function receives the raw serialized input that has been sent to the model's endpoint and its job is to de-serialize and make the input available for the inference code.
 - `output_fn`: This function takes the output of the inference code and its job is to serialize this output and return it to the caller of the model's endpoint.
 - `predict_fn`: The heart of the inference script, this is where the actual prediction is done and is the function which you will need to complete.

For the simple website that we are constructing during this project, the `input_fn` and `output_fn` methods are relatively straightforward. We only require being able to accept a string as input and we expect to return a single value as output. You might imagine though that in a more complex application the input or output may be image data or some other binary data which would require some effort to serialize.

### (TODO) Writing inference code

Before writing our custom inference code, we will begin by taking a look at the code which has been provided.

```python
!pygmentize serve/predict.py
```

As mentioned earlier, the `model_fn` method is the same as the one provided in the training code and the `input_fn` and `output_fn` methods are very simple and your task will be to complete the `predict_fn` method. Make sure that you save the completed file as `predict.py` in the `serve` directory.

**TODO**: Complete the `predict_fn()` method in the `serve/predict.py` file.

### Deploying the model

Now that the custom inference code has been written, we will create and deploy our model. To begin with, we need to construct a new PyTorchModel object which points to the model artifacts created during training and also points to the inference code that we wish to use. Then we can call the deploy method to launch the deployment container.

**NOTE**: The default behaviour for a deployed PyTorch model is to assume that any input passed to the predictor is a `numpy` array. In our case we want to send a string so we need to construct a simple wrapper around the `RealTimePredictor` class to accomodate simple strings. In a more complicated situation you may want to provide a serialization object, for example if you wanted to sent image data.

```python
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

model = PyTorchModel(model_data=rnn_estimator.model_data,
                     role = role,
                     framework_version='0.4.0',
                     entry_point='predict.py',
                     source_dir='serve',
                     predictor_cls=StringPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

### Testing the model

Now that we have deployed our model with the custom inference code, we should test to see if
everything is working. Here we test our model by loading the first `250` positive and negative
reviews and send them to the endpoint, then collect the results. The reason for only sending some
of the data is that the amount of time it takes for our model to process the input and then perform
inference is quite long and so testing the entire data set would be prohibitive.

```python
import glob

def test_reviews(data_dir='../data/aclImdb', stop=250):
    
    results = []
    ground = []
    
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        
        files_read = 0
        
        print('Starting', sentiment, 'files')
        
        # Iterate through the files and send them to the predictor
        for f in files:
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                # Read in the review and convert to 'utf-8' for transmission via HTTP
                review_input = review.read().encode('utf-8')
                # Send the review to the predictor and store the results
                results.append(int(predictor.predict(review_input)))
                
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results
```

```python
ground, results = test_reviews()
```

    Starting pos files
    Starting neg files

```python
from sklearn.metrics import accuracy_score
accuracy_score(ground, results)
```

    0.854

As an additional test, we can try sending the `test_review` that we looked at earlier.

```python
predictor.predict(test_review)
```

    b'1'

Now that we know our endpoint is working as expected, we can set up the web page that will interact
with it. If you don't have time to finish the project now, make sure to skip down to the end of this
notebook and shut down your endpoint. You can deploy it again when you come back.

## Step 7 (again): Use the model for the web app

> **TODO:** This entire section and the next contain tasks for you to complete, mostly using the AWS console.

So far we have been accessing our model endpoint by constructing a predictor object which uses the endpoint and then just using the predictor object to perform inference. What if we wanted to create a web app which accessed our model? The way things are set up currently makes that not possible since in order to access a SageMaker endpoint the app would first have to authenticate with AWS using an IAM role which included access to SageMaker endpoints. However, there is an easier way! We just need to use some additional AWS services.

<img src="Web App Diagram.svg">

The diagram above gives an overview of how the various services will work together. On the far right
is the model which we trained above and which is deployed using SageMaker. On the far left is our
web app that collects a user's movie review, sends it off and expects a positive or negative
sentiment in return.

In the middle is where some of the magic happens. We will construct a Lambda function, which you can think of as a straightforward Python function that can be executed whenever a specified event occurs. We will give this function permission to send and recieve data from a SageMaker endpoint.

Lastly, the method we will use to execute the Lambda function is a new endpoint that we will create using API Gateway. This endpoint will be a url that listens for data to be sent to it. Once it gets some data it will pass that data on to the Lambda function and then return whatever the Lambda function returns. Essentially it will act as an interface that lets our web app communicate with the Lambda function.

### Setting up a Lambda function

The first thing we are going to do is set up a Lambda function. This Lambda function will be executed whenever our public API has data sent to it. When it is executed it will receive the data, perform any sort of processing that is required, send the data (the review) to the SageMaker endpoint we've created and then return the result.

#### Part A: Create an IAM Role for the Lambda function

Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.

Using the AWS Console, navigate to the **IAM** page and click on **Roles**. Then, click on **Create role**. Make sure that the **AWS service** is the type of trusted entity selected and choose **Lambda** as the service that will use this role, then click **Next: Permissions**.

In the search box type `sagemaker` and select the check box next to the **AmazonSageMakerFullAccess** policy. Then, click on **Next: Review**.

Lastly, give this role a name. Make sure you use a name that you will remember later on, for example `LambdaSageMakerRole`. Then, click on **Create role**.

#### Part B: Create a Lambda function

Now it is time to actually create the Lambda function.

Using the AWS Console, navigate to the AWS Lambda page and click on **Create a function**. When you get to the next page, make sure that **Author from scratch** is selected. Now, name your Lambda function, using a name that you will remember later on, for example `sentiment_analysis_func`. Make sure that the **Python 3.6** runtime is selected and then choose the role that you created in the previous part. Then, click on **Create Function**.

On the next page you will see some information about the Lambda function you've just created. If you scroll down you should see an editor in which you can write the code that will be executed when your Lambda function is triggered. In our example, we will use the code below. 

```python
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '**ENDPOINT NAME HERE**',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = event['body'])                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : result
    }
```

Once you have copy and pasted the code above into the Lambda code editor, replace the `**ENDPOINT NAME HERE**` portion with the name of the endpoint that we deployed earlier. You can determine the name of the endpoint using the code cell below.

```python
predictor.endpoint
```

    'sagemaker-pytorch-2020-03-31-00-38-37-511'

Once you have added the endpoint name to the Lambda function, click on **Save**. Your Lambda function
is now up and running. Next we need to create a way for our web app to execute the Lambda function.

### Setting up API Gateway

Now that our Lambda function is set up, it is time to create a new API using API Gateway that will
trigger the Lambda function we have just created.

Using AWS Console, navigate to **Amazon API Gateway** and then click on **Get started**.

On the next page, make sure that **New API** is selected and give the new api a name, for example, `sentiment_analysis_api`. Then, click on **Create API**.

Now we have created an API, however it doesn't currently do anything. What we want it to do is to trigger the Lambda function that we created earlier.

Select the **Actions** dropdown menu and click **Create Method**. A new blank method will be created, select its dropdown menu and select **POST**, then click on the check mark beside it.

For the integration point, make sure that **Lambda Function** is selected and click on the **Use Lambda Proxy integration**. This option makes sure that the data that is sent to the API is then sent directly to the Lambda function with no processing. It also means that the return value must be a proper response object as it will also not be processed by API Gateway.

Type the name of the Lambda function you created earlier into the **Lambda Function** text entry box and then click on **Save**. Click on **OK** in the pop-up box that then appears, giving permission to API Gateway to invoke the Lambda function you created.

The last step in creating the API Gateway is to select the **Actions** dropdown and click on **Deploy API**. You will need to create a new Deployment stage and name it anything you like, for example `prod`.

You have now successfully set up a public API to access your SageMaker model. Make sure to copy or write down the URL provided to invoke your newly created public API as this will be needed in the next step. This URL can be found at the top of the page, highlighted in blue next to the text **Invoke URL**.

## Step 4: Deploying our web app

Now that we have a publicly available API, we can start using it in a web app. For our purposes, we have provided a simple static html file which can make use of the public api you created earlier.

In the `website` folder there should be a file called `index.html`. Download the file to your computer and open that file up in a text editor of your choice. There should be a line which contains **\*\*REPLACE WITH PUBLIC API URL\*\***. Replace this string with the url that you wrote down in the last step and then save the file.

Now, if you open `index.html` on your local computer, your browser will behave as a local web server and you can use the provided site to interact with your SageMaker model.

If you'd like to go further, you can host this html file anywhere you'd like, for example using github or hosting a static site on Amazon's S3. Once you have done this you can share the link with anyone you'd like and have them play with it too!

> **Important Note** In order for the web app to communicate with the SageMaker endpoint, the endpoint has to actually be deployed and running. This means that you are paying for it. Make sure that the endpoint is running when you want to use the web app but that you shut it down when you don't need it, otherwise you will end up with a surprisingly large AWS bill.

**TODO:** Make sure that you include the edited `index.html` file in your project submission.

Now that your web app is working, trying playing around with it and see how well it works.

**Question**: Give an example of a review that you entered into your web app. What was the predicted sentiment of your example review?

**Answer:** I recommend that you use Postman or `curl` to hit my AWS endpoint because the index.html assumes `result` from response but the provided Lambda code returns `body` instead of `result`.

```
curl -H "Content-Type: application/json" -X POST -d '{"body": "I like Parasite because of its masterful cinematography"}' https://0wyriiyfe7.execute-api.us-west-2.amazonaws.com/production/sagemaker-pytorch-rnn/
```
 
My endpoint says it is positive!

```
{"statusCode": 200, "headers": {"Content-Type": "text/plain", "Access-Control-Allow-Origin": "*"}, "body": "1"}
```

### Delete the endpoint

Remember to always shut down your endpoint if you are no longer using it. You are charged for the length of time that the endpoint is running so if you forget and leave it on you could end up with an unexpectedly large bill.


```python
predictor.delete_endpoint()
```
