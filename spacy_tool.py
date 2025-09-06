import spacy

nlp = spacy.load("en_core_web_sm")

doc = nlp("person is curruntly working on AI")

print([token.text for token in doc])

