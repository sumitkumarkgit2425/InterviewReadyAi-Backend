
import spacy
from wordfreq import zipf_frequency
nlp = spacy.load('en_core_web_sm')
text = '''We are seeking an innovative Senior Android Software Engineer to lead our mobile modernization initiative. The ideal candidate will have deep expertise in building large-scale, consumer-facing mobile applications from the ground up prioritizing speed and pixel-perfect design. You must be heavily proficient in modern Android development using Kotlin, and have migrated legacy Java codebases. We expect strong mastery over architectural patterns, specifically transitioning from MVP to clean MVVM and MVI. You should regularly utilize Jetpack Compose for declarative UI, alongside Coroutines and Flow for complex asynchronous programming and state management. Experience with dependency injection using Dagger Hilt, and local persistent storage via Room Database is mandatory. We interface heavily with external data; you must have experience consuming GraphQL APIs and optimizing network performance over REST using Retrofit and OkHttp. Furthermore, you will be expected to author comprehensive unit tests utilizing Mockito and JUnit, and configure automated deployment pipelines through GitHub Actions'''
doc = nlp(text)

raw_phrases = []
current_phrase = []

for token in doc:
    if token.is_stop or token.is_punct or len(token.lemma_) <= 2:
        if current_phrase:
            raw_phrases.append(' '.join(current_phrase))
            current_phrase = []
        continue

    word = token.text
    is_acronym = word.isupper() and len(word) > 1
    is_mixed_case = any(c.isupper() for c in word[1:])
    is_title = token.is_title and not token.is_sent_start
    is_noun = token.pos_ in ['PROPN', 'NOUN']
    
    if (is_acronym or is_mixed_case or is_title) and is_noun:
        current_phrase.append(word)
    else:
        if current_phrase:
            raw_phrases.append(' '.join(current_phrase))
            current_phrase = []
if current_phrase:
    raw_phrases.append(' '.join(current_phrase))

final_skills = set()
for phrase in raw_phrases:
    words = phrase.split()
    avg_zipf = sum(zipf_frequency(w.lower(), 'en') for w in words) / len(words)
    has_acronym = any(w.isupper() for w in words)
    has_mixed = any(any(c.isupper() for c in w[1:]) for w in words)
    
    if avg_zipf < 4.3 or has_acronym or has_mixed:
        final_skills.add(phrase.lower())

print('FINAL:', final_skills)

