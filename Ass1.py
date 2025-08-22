
import nltk
import string
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("omw-1.4")  


sample_text = """
DNYANESHWARI BALASAHEB KADAM, FROM RAHURI."""


sentences = sent_tokenize(sample_text)
print("Sentence Tokenization:")
print(sentences)

words = word_tokenize(sample_text)
print("\nWord Tokenization:")
print(words)


stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word.lower() not in stop_words and word.isalpha()]
print("\nWords after Stopword and Punctuation Removal:")
print(filtered_words)


stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_words]
print("\nStemmed Words:")
print(stemmed_words)


lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nLemmatized Words:")
print(lemmatized_words)

pos_tags = pos_tag(filtered_words)
print("\nPOS Tags:")
print(pos_tags)