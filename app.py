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

        # PHASE 1: Smart Extraction (Scalable Structural Heuristic + NER)
        def extract_technical_skills(text):
            doc = nlp(text)
            skills = set()
            tech_indicator = []
            
            # Step 1: Identify strictly capitalized/acronym technical words
            for token in doc:
                if token.is_stop or token.is_punct or len(token.text) <= 1:
                    tech_indicator.append((False, ""))
                    continue
                    
                is_acronym = token.text.isupper() and len(token.text) > 1
                is_propn = token.pos_ in ['PROPN', 'NOUN']
                is_title = token.is_title and not token.is_sent_start
                
                # Only extract if it's grammatically a noun/pronoun, or an acronym
                if is_acronym or (is_propn and is_title):
                    word = token.text.lower() if is_acronym else token.lemma_.lower()
                    tech_indicator.append((True, word))
                else:
                    tech_indicator.append((False, ""))

            # Step 2: Extract contiguous multi-word phrases (e.g., "Jetpack Compose")
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
                
            # Step 3: Named Entity Recognition Filtering
            # Job Titles, Companies, and Locations get classified by spaCy as entities. 
            # We filter out any phrase that overlaps with an ORG, PERSON, or GPE tag.
            invalid_entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE']]
            
            # Plus a tiny fallback for exact job titles that are sometimes misclassified by the "Small" model
            fallback_ignore = {"senior", "junior", "engineer", "developer", "manager", "lead"}
            
            for phrase in raw_phrases:
                phrase_lower = phrase.lower()
                
                # Check 1: Does it match a recognized Entity? (e.g., "Google" -> ORG)
                is_entity = any(phrase_lower in ent or ent in phrase_lower for ent in invalid_entities)
                
                # Check 2: Does the phrase contain a known title keyword?
                has_title_word = any(title_word in phrase_lower.split() for title_word in fallback_ignore)
                
                if not is_entity and not has_title_word:
                    skills.add(phrase)
                    
            return list(skills)

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
