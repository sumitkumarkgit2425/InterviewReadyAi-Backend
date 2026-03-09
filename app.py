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

        # PHASE 1: Smart Extraction (Scalable Structural Heuristic)
        def extract_technical_skills(text):
            doc = nlp(text)
            skills = set()
            
            tech_indicator = []
            
            for token in doc:
                # 1. Ignore standard stop words and punctuation immediately
                if token.is_stop or token.is_punct or len(token.text) <= 1:
                    tech_indicator.append((False, ""))
                    continue
                    
                # 2. Check Structural Heuristics
                is_acronym = token.text.isupper() and len(token.text) > 1
                is_propn = token.pos_ == 'PROPN'
                is_title = token.is_title and not token.is_sent_start
                
                if is_acronym or is_propn or is_title:
                    # Use lower text for acronyms, lemma for others
                    word = token.text.lower() if is_acronym else token.lemma_.lower()
                    tech_indicator.append((True, word))
                else:
                    tech_indicator.append((False, ""))

            # 3. Extract contiguous multi-word phrases (e.g., "Jetpack Compose", "Android SDK")
            current_phrase = []
            for is_tech, word in tech_indicator:
                if is_tech:
                    current_phrase.append(word)
                else:
                    if len(current_phrase) > 0:
                        skills.add(" ".join(current_phrase))
                    current_phrase = []
                    
            if len(current_phrase) > 0:
                skills.add(" ".join(current_phrase))
                        
            return list(skills)

        resume_terms = extract_technical_skills(resume_text)
        jd_terms = extract_technical_skills(jd_text)

        # Pre-process the resume string for RapidFuzz
        resume_str = " ".join(resume_terms)

        
        # Pre-process word vectors for semantic matching
        resume_doc = nlp(resume_str)

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
                
            # PHASE 4: Semantic Vector Match (e.g. "Neural Networks" vs "Machine Learning")
            jd_word_vec = nlp(jd_word)
            highest_semantic_sim = 0.0
            
            # Only calculate if the word actually has a known vector representation in English
            if jd_word_vec.has_vector and resume_terms:
                # Compare against the unique parsed resume terms, not the massive raw document
                for term in set(resume_terms):
                    resume_word_vec = nlp(term)
                    if resume_word_vec.has_vector:
                        sim = jd_word_vec.similarity(resume_word_vec)
                        if sim > highest_semantic_sim:
                            highest_semantic_sim = sim
            
            # If the vectors are 70%+ identical conceptually, issue a partial matching score!
            if highest_semantic_sim > 0.70:
                matched_weight += (weight * highest_semantic_sim) # Partial points
            else:
                missing_candidates[jd_word] = weight

        # PHASE 5: Weighted Scoring
        match_percentage = round((matched_weight / total_weight) * 100, 2)

        # Sort missing candidates by highest weight, keep top 15
        sorted_missing = [word for word, weight in sorted(missing_candidates.items(), key=lambda x: x[1], reverse=True)]
        missing_keywords = sorted_missing[:15]

        return jsonify({
            "match_percentage": match_percentage,
            "missing_keywords": missing_keywords
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Start the server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
