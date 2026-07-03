# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

from typing import List, Dict, Tuple, Optional
import re

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        # Basic cleanup
        cleaned = text.strip().lower()

        # Map common emoticons/emojis to word tokens so they act as signals.
        emoticon_map = {
          ":)": "smile",
          ":-)": "smile",
          ":(": "frown",
          ":-(": "frown",
          "😂": "joy",
          "🥲": "sad_smile",
          "💀": "skull",
        }
        for k, v in emoticon_map.items():
          cleaned = cleaned.replace(k, f" {v} ")

        # Remove punctuation (keep letters, numbers, underscore and whitespace)
        cleaned = re.sub(r"[^a-z0-9_\s]", " ", cleaned)

        # Split into tokens
        tokens = cleaned.split()

        # Normalize repeated characters: collapse 3+ repeats to 2 (soooo -> soo)
        tokens = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in tokens]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        # Basic scoring with two improvements:
        # 1) Handle simple negation words that invert the next token's polarity.
        # 2) Treat common emoji/emoticon tokens as stronger signals.
        tokens = self.preprocess(text)

        negation_tokens = {"not", "never", "no", "n't"}

        # stronger weights for emotion tokens created in preprocess
        emoji_weights = {
          "joy": 2,
          "smile": 1,
          "sad_smile": -1,
          "frown": -1,
          "skull": -2,
        }

        score = 0
        i = 0
        while i < len(tokens):
          t = tokens[i]

          # handle simple negation by looking at the next token
          if t in negation_tokens and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in self.positive_words:
              score -= 1
              i += 2
              continue
            if nxt in self.negative_words:
              score += 1
              i += 2
              continue

          # emoji/emoticon tokens
          if t in emoji_weights:
            score += emoji_weights[t]
            i += 1
            continue

          # regular word matches (count each occurrence)
          if t in self.positive_words:
            score += 1
          if t in self.negative_words:
            score -= 1

          i += 1

        return score

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score = self.score_text(text)

        # Map numeric score to labels. Use "mixed" for weak signals.
        if score > 1:
          return "positive"
        if score < -1:
          return "negative"
        if score == 0:
          return "neutral"
        # score is either 1 or -1 -> ambiguous/mixed
        return "mixed"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
