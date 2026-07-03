"""
Breaker Test: Sentences designed to expose failure patterns in the mood analyzer.
"""

from mood_analyzer import MoodAnalyzer
import dataset

ma = MoodAnalyzer()

# Breaker sentences designed to confuse the model
breakers = [
    # Sarcasm: positive words but negative intent
    "I absolutely love getting stuck in traffic",
    
    # Slang with multiple meanings (not in word lists)
    "That's sick! No really, this is fire",
    
    # Misleading emoji: "fine" is neutral but emoji is positive
    "I'm fine 🙂",
    
    # Mixed emotions: conflicting signals
    "I'm exhausted but proud of myself",
    
    # Negation edge case: "not bad" should be positive but might confuse
    "Not bad, could be better",
]

print("=" * 70)
print("BREAKER TEST: Analyzing sentences designed to confuse the model")
print("=" * 70)

for i, post in enumerate(breakers, 1):
    print(f"\n[BREAKER {i}]")
    print(f"Sentence: {post}")
    
    # Get tokens, score, prediction, and explanation
    tokens = ma.preprocess(post)
    score = ma.score_text(post)
    pred = ma.predict_label(post)
    expl = ma.explain(post)
    
    print(f"  Tokens: {tokens}")
    print(f"  Score: {score}")
    print(f"  Predicted: {pred}")
    print(f"  Explanation: {expl}")
    
    # Analyze which words affected the score
    positive_in_tokens = [t for t in tokens if t in ma.positive_words]
    negative_in_tokens = [t for t in tokens if t in ma.negative_words]
    emoji_in_tokens = [t for t in tokens if t in {"joy", "smile", "sad_smile", "frown", "skull"}]
    
    print(f"  Words that affected score:")
    if positive_in_tokens:
        print(f"    - Positive: {positive_in_tokens}")
    if negative_in_tokens:
        print(f"    - Negative: {negative_in_tokens}")
    if emoji_in_tokens:
        print(f"    - Emojis: {emoji_in_tokens}")
    
    # Identify ignored words
    ignored = [t for t in tokens if t not in ma.positive_words and t not in ma.negative_words and t not in {"joy", "smile", "sad_smile", "frown", "skull"}]
    if ignored:
        print(f"  Words ignored (not in lists): {ignored}")

print("\n" + "=" * 70)
print("FAILURE PATTERN ANALYSIS")
print("=" * 70)

print("""
Based on the breaker tests above, here are the identified failure patterns:

1. SARCASM NOT DETECTED:
   - "I absolutely love getting stuck in traffic"
   - Expected: negative (sarcasm)
   - Problem: "love" is +1, "absolutely" isn't a negation token
   - Root cause: No sarcasm detection; only simple negation (not, never, no, n't)

2. SLANG WORDS IGNORED:
   - "That's sick! No really, this is fire"
   - Expected: positive (slang means good)
   - Problem: "sick" and "fire" aren't in POSITIVE_WORDS or NEGATIVE_WORDS
   - Root cause: Limited vocabulary; slang isn't in the word lists

3. SINGLE-WORD MISCLASSIFICATION:
   - "I love this class so much"
   - Expected: positive
   - Actual: mixed (score=1)
   - Root cause: Threshold is > 1 for positive, so score of 1 becomes "mixed"
   - Fix: Could lower threshold to >= 1 or expand word lists

4. EMOJI DOMINANCE:
   - "I'm fine 🙂" 
   - Problem: "fine" alone = 0, emoji "smile" = +1
   - Single emoji can override text intent
   - Root cause: Emojis have outsized weight relative to word count

5. NEGATION EDGE CASE:
   - "Not bad, could be better"
   - Problem: "not bad" should be somewhat positive but isn't handled well
   - Root cause: Negation only checks next token; "not bad" needs special handling
""")
