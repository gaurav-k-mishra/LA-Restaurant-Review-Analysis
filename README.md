# LA Restaurant Review Analysis — NLP Pipeline

A full-cycle natural language processing project analyzing customer reviews of top-rated restaurants in Los Angeles, built as part of an applied data science project (2025).

**Team:** Ali Ladha · Chase Pattee · Gaurav Mishra · Robby Stohel

---

## Project Overview

Customer reviews contain rich, unstructured information about dining experiences — but extracting actionable insight requires more than reading them. This project designs and implements a reproducible NLP pipeline to analyze thousands of reviews across 240 top-rated LA restaurants, uncovering patterns in sentiment, emotion, and thematic content that can inform real business decisions.

### What the analysis covers

- **Sentiment polarity** — scoring each review from negative (−1) to positive (+1) using TextBlob
- **Emotion profiling** — quantifying eight basic emotions (joy, trust, anger, fear, sadness, disgust, surprise, anticipation) using the NRC Emotion Lexicon
- **Topic modeling** — identifying five latent themes via Latent Dirichlet Allocation (LDA)
- **Sentiment by topic** — breaking down polarity and emotion scores within each discovered theme
- **Star rating analysis** — correlating topic dominance with Yelp-style star ratings
- **Business recommendations** — translating analytical findings into actionable restaurant strategy

---

## Methodology

| Stage | Description |
|---|---|
| Data ingestion | 240 top-rated LA restaurants; fields include review text, star rating, restaurant name, and price tier |
| Text preprocessing | Contraction expansion, SymSpell spelling correction, NER entity joining, POS filtering, lemmatization, phrase modeling |
| Sentiment scoring | Polarity via TextBlob; emotion scores via NRCLex (8 emotions, normalized per review) |
| Topic modeling | LDA with CountVectorizer; optimal topic count selected by log-likelihood and perplexity evaluation |
| Sentiment by topic | Average polarity and emotion scores computed per dominant topic |
| Star rating by topic | Mean star rating mapped to each LDA topic for quality benchmarking |

---

## Discovered Topics

| Topic | Label | Avg. Polarity | Avg. Star Rating |
|---|---|---|---|
| 0 | Dining Experience | 0.34 | 4.27 |
| 1 | Order Service | 0.17 | 4.24 |
| 2 | Food Quality | 0.40 | 4.36 |
| 3 | Restaurant Atmosphere | — | — |
| 4 | Menu & Price | — | — |

**Topic 2 (Food Quality)** is the highest-rated and most positively charged topic, reflecting strong customer satisfaction with freshness and preparation. **Topic 1 (Order Service)** is the lowest-rated, with elevated negative emotions including fear, anger, and disgust — pointing to operational friction in ordering and table service.

---

## Key Findings

- Joy and Trust dominate the overall emotional radar — diners generally feel happy and confident
- Food Quality drives the highest satisfaction (polarity 0.40, avg. rating 4.36)
- Order Service is the clearest pain point, with the most elevated negative emotions and lowest average rating
- Anticipation paired with lower satisfaction scores in some topics may signal expectation gaps
- Emotion profiles per topic provide richer diagnostic signal than polarity scores alone

---

## Tech Stack

- **Language:** Python
- **NLP libraries:** TextBlob, NRCLex, spaCy, Gensim (LDA), SymSpell
- **ML framework:** scikit-learn (custom pipeline transformers)
- **Data:** pandas, CountVectorizer
- **Visualization:** matplotlib, radar/bar charts
- **Environment:** Jupyter Notebook

---

## Pipeline Entities

`reviews` · `polarity_scores` · `emotion_scores` · `topics` · `topic_assignments` · `star_ratings` · `restaurant_metadata`

---

## Key Highlights

- End-to-end scikit-learn pipeline with custom `PolarityScorer` and `EmotionScorer` transformers
- SymSpell spelling correction reduces vocabulary sparsity from user-generated text
- NER-aware preprocessing treats multi-word entities (e.g., "Los Angeles") as single tokens
- LDA topic count selected empirically via log-likelihood and perplexity curves
- Separate POS filtering stage (nouns + adjectives only) applied before topic modeling to reduce noise
- Business recommendations derived directly from quantitative emotional and thematic findings
