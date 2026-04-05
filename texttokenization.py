import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
text = "Machine Learning and Data Science are important skills in modern technology careers."
tokens = word_tokenize(text)
print("Tokens:")
print(tokens)
stop_words = set(stopwords.words('english'))
filtered_words = []
for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)
print("\nAfter Removing Stop Words:")
print(filtered_words)