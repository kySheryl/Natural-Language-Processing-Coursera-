import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'data/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    embeddings={}
    embeddings_dim = 0
    with open(embeddings_path, 'r') as f:
      first = True
      for line in f:
        all = line.strip().split()
        vec = []
        for v in all[1:]:
          vec.append(numpy.float32(v))
        embeddings[all[0]]=vec
        if first:
          embeddings_dim = len(vec)
          first = False
    return embeddings, embeddings_dim
    ########################


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""

    # Hint: you have already implemented exactly this function in the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    result = np.zeros(dim)
    count = 0
    for word in question.split():
      if word in embeddings:
        count += 1
        result = np.add(result, embeddings[word])
    if count == 0:
      return result
    else:
      return result/count
    ########################


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
