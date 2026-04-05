import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Google is collaborating with Indian universities to improve AI education in Chennai in 2025.
The CEO Sundar Pichai spoke about this initiative at a tech summit in New York.
"""

doc = nlp(text)

print("Sentences:")
for sent in doc.sents:
    print("-", sent.text)

print("\n------------------")

print("Keywords:")
keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
print(keywords)

print("\n------------------")

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)