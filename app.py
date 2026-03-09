from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"}), 200

@app.route('/match', methods=['POST'])
def match_resume():
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        jd_text = data.get('jd_text', '')

        import spacy
        from rapidfuzz import fuzz

        # Load the Small English model to fit in Render 512MB RAM limit
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        # PHASE 1: Smart Extraction (Statistical Rarity / TF-IDF Alternative)
        def extract_technical_skills(text):
            doc = nlp(text)
            from wordfreq import zipf_frequency
            
            # Step 1: Capitalization Structural Check
            tech_indicator = []
            for token in doc:
                if token.is_stop or token.is_punct or len(token.lemma_) <= 2:
                    if len(tech_indicator) > 0 and tech_indicator[-1][0]: 
                        tech_indicator.append((False, ""))
                    continue

                word = token.text
                is_acronym = word.isupper() and len(word) > 1
                is_mixed_case = any(c.isupper() for c in word[1:])
                is_title = token.is_title and not token.is_sent_start
                is_noun = token.pos_ in ['PROPN', 'NOUN']
                
                if (is_acronym or is_mixed_case or is_title) and is_noun:
                    tech_indicator.append((True, word))
                else:
                    tech_indicator.append((False, ""))

            # Step 2: Phrase Contiguity
            raw_phrases = []
            current_phrase = []
            for is_tech, word in tech_indicator:
                if is_tech:
                    current_phrase.append(word)
                else:
                    if len(current_phrase) > 0:
                        raw_phrases.append(" ".join(current_phrase))
                    current_phrase = []
            if len(current_phrase) > 0:
                raw_phrases.append(" ".join(current_phrase))
                        
            # Step 3: Statistical Rarity Mathematical Filter (Option 2)
            # We completely bypass maintaining a hardcoded "ignore list" of Marketing Fluff and Job Titles!
            # Instead, we mathematically calculate how rare the word is in the English language.
            final_skills = set()
            for phrase in raw_phrases:
                words = phrase.split()
                # A Zipf frequency evaluates rarity. >4.5 is common English. <4.0 is extremely rare domain logic.
                avg_zipf = sum(zipf_frequency(w.lower(), 'en') for w in words) / len(words)
                
                # We unconditionally keep exact Acronyms (e.g. MVP, REST) and MixedCase logic (OkHttp, GraphQL)
                has_acronym = any(w.isupper() for w in words)
                has_mixed = any(any(c.isupper() for c in w[1:]) for w in words)
                
                # If a phrase is composed entirely of common English words (avg Zipf > 4.3), we drop it.
                if avg_zipf < 4.3 or has_acronym or has_mixed:
                    final_skills.add(phrase.lower())
                    
            return list(final_skills)

        resume_terms = extract_technical_skills(resume_text)
        jd_terms = extract_technical_skills(jd_text)

        # Pre-process the resume string for RapidFuzz
        resume_str = " ".join(resume_terms)

        # PHASE 2: Importance Weighting (Frequency Counting)
        # We count the frequency of each extracted term in the JD to weigh its importance
        from collections import Counter
        jd_weights = dict(Counter(jd_terms))

        missing_candidates = {}
        matched_weight = 0.0
        total_weight = sum(jd_weights.values())

        if total_weight == 0:
            return jsonify({"match_percentage": 0.0, "missing_keywords": []}), 200

        for jd_word, weight in jd_weights.items():
            # 1. Exact direct string match
            if jd_word in resume_terms:
                matched_weight += weight
                continue
                
            # PHASE 3: Fuzzy Match (e.g. React.js vs React)
            best_fuzzy_match = fuzz.partial_ratio(jd_word, resume_str)
            if best_fuzzy_match > 85:
                matched_weight += weight
                continue
                
            # If no exact or fuzzy match, it's missing! 
            # (Note: Neural/Semantic math was stripped out to fit Render.com 512MB RAM limits)
            missing_candidates[jd_word] = weight

        # PHASE 4: Weighted Scoring
        match_percentage = round((matched_weight / total_weight) * 100, 2)

        # Sort missing candidates by highest weight, keep top 15 unique words
        sorted_missing = [word for word, weight in sorted(missing_candidates.items(), key=lambda x: x[1], reverse=True)]
        
        # Deduplicate while preserving order
        seen = set()
        missing_keywords = []
        for word in sorted_missing:
            if word not in seen:
                seen.add(word)
                missing_keywords.append(word)
            if len(missing_keywords) == 15:
                break

        return jsonify({
            "match_percentage": match_percentage,
            "missing_keywords": missing_keywords
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
