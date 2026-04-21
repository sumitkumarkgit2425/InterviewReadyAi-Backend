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

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")

        def extract_technical_skills(text):
            doc = nlp(text)
            from wordfreq import zipf_frequency
            
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
                        
            final_skills = set()
            for phrase in raw_phrases:
                words = phrase.split()
                avg_zipf = sum(zipf_frequency(w.lower(), 'en') for w in words) / len(words)
                has_acronym = any(w.isupper() for w in words)
                has_mixed = any(any(c.isupper() for c in w[1:]) for w in words)
                
                if avg_zipf < 4.3 or has_acronym or has_mixed:
                    final_skills.add(phrase.lower())
                    
            return list(final_skills)

        resume_terms = extract_technical_skills(resume_text)
        jd_terms = extract_technical_skills(jd_text)

        resume_str = " ".join(resume_terms)

        from collections import Counter
        jd_weights = dict(Counter(jd_terms))

        missing_candidates = {}
        matched_weight = 0.0
        total_weight = sum(jd_weights.values())

        if total_weight == 0:
            return jsonify({"match_percentage": 0.0, "missing_keywords": []}), 200

        for jd_word, weight in jd_weights.items():
            if jd_word in resume_terms:
                matched_weight += weight
                continue
                
            best_fuzzy_match = fuzz.partial_ratio(jd_word, resume_str)
            if best_fuzzy_match > 85:
                matched_weight += weight
                continue
                
            missing_candidates[jd_word] = weight

        match_percentage = round((matched_weight / total_weight) * 100, 2)

        sorted_missing = [word for word, weight in sorted(missing_candidates.items(), key=lambda x: x[1], reverse=True)]
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
    app.run(host='0.0.0.0', port=5000, debug=True)
