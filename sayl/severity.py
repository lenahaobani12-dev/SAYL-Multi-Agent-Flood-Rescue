"""
Severity analysis of Arabic distress messages (the NLP part of the Distress Agent).

Owner: Person 2 (Distress Agent + Command Center)

Two methods are combined (hybrid):

1. Rule-based analysis (always available)
   * Text normalisation (diacritics, alef / ya / ta marbuta, Arabic digits).
   * Keyword lexicon: each keyword gives a severity level and a tag
     (e.g. "رضيع" -> infant, level 4; "يغرق" -> drowning, level 5).
   * Severity = the highest level found, +1 if a vulnerable person is in a hazard,
     +1 if 5 or more people, -1 if the message says they are safe.
   * Negation handling: "ما فيه مصابين" does NOT count as "injured".
   * Safety phrases ("احنا بخير") lower the score.
   * Number of people from digits or Arabic number words.

2. Machine-learning classifier (used when scikit-learn is installed)
   * TF-IDF on character n-grams (robust to dialect spelling) + Logistic Regression,
     trained on data/distress_messages.json (labelled 1..5).

Final severity: the rules decide; when the ML model is confident (probability >= 0.6)
and predicts a HIGHER severity, the higher value is used, because in an emergency
under-estimating a case is more dangerous than over-estimating it.

Severity scale (also used to label the data)
--------------------------------------------
1 = property damage only, nobody in danger
2 = water inside / evacuation request, people are safe for now
3 = people trapped, or vulnerable people present (elderly, children, disabled)
4 = serious risk: injury, infant, medical device, water at chest level, electricity
5 = life-threatening now: drowning, swept away, unconscious / not breathing
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "distress_messages.json")

# keyword -> (level, tag). The level is the severity this sign implies on its own (1..5).
# Keywords are normalised automatically, so write them the way people type them.
KEYWORDS: Dict[str, Tuple[int, str]] = {
    # vulnerable people
    "كبار سن": (3, "elderly"), "كبير سن": (3, "elderly"), "كبيره سن": (3, "elderly"), "كبير في السن": (3, "elderly"),
    "عجوز": (3, "elderly"), "شايب": (3, "elderly"), "جدتي": (3, "elderly"), "جدي": (3, "elderly"),
    "ابوي كبير": (3, "elderly"), "رجل كبير": (3, "elderly"),
    "اطفال": (3, "children"), "طفل": (3, "children"), "بزر": (3, "children"), "عيال": (3, "children"),
    "بنت صغيره": (3, "children"), "ولدي": (3, "children"),
    "رضيع": (4, "infant"), "حامل": (3, "pregnant"), "الطلق": (4, "pregnant"), "الشهر التاسع": (4, "pregnant"),
    "معاق": (3, "disabled"), "كرسي متحرك": (3, "disabled"), "ما تقدر تمشي": (3, "disabled"),
    "ما يقدرون يمشون": (3, "disabled"),
    # medical
    "مصاب": (4, "injured"), "جرحي": (4, "injured"), "ينزف": (4, "injured"), "تنزف": (4, "injured"),
    "نزيف": (4, "injured"), "كسر": (4, "injured"), "انكسرت": (4, "injured"),
    "اغمي": (5, "unconscious"), "مغمي": (5, "unconscious"), "مغمى": (5, "unconscious"),
    "ما يتنفس": (5, "not-breathing"), "نفسه ضايق": (4, "breathing"), "ربو": (4, "breathing"),
    "اكسجين": (4, "medical-device"), "غسيل كلي": (3, "medical"), "مريض": (3, "medical"),
    # trapped / location
    "محاصر": (3, "trapped"), "محبوس": (3, "trapped"), "عالق": (3, "trapped"), "علقنا": (3, "trapped"),
    "علقت": (3, "trapped"), "علقوا": (3, "trapped"), "ما نقدر نطلع": (3, "trapped"), "حاوطتنا": (3, "trapped"),
    "المصعد": (3, "trapped"), "داخلها": (3, "trapped"), "ننتظر": (3, "waiting"), "تنتظر": (3, "waiting"),
    "السطح": (3, "on-roof"), "سطح": (3, "on-roof"), "سقف سياره": (5, "on-car"), "فوق السياره": (4, "on-car"),
    "باص": (4, "vehicle-group"), "جسر": (4, "bridge"), "النفق": (3, "tunnel"),
    "متعلق": (5, "clinging"), "الوادي": (2, "in-wadi"), "حفره": (4, "pit"),
    # life threat
    "يغرق": (5, "drowning"), "تغرق": (5, "drowning"), "غريق": (5, "drowning"),
    "انجرف": (5, "swept-away"), "شال سياره": (5, "swept-away"), "تتحرك مع المويه": (5, "swept-away"),
    "يطفو": (5, "floating-body"), "ما يتحرك": (5, "not-moving"), "ضايعه في المويه": (5, "lost-in-water"),
    "طاح في المويه": (4, "fell-in-water"), "طاحت في المويه": (4, "fell-in-water"), "ما قدر يقوم": (4, "cannot-stand"),
    "تصارخ": (4, "screaming"), "يصارخ": (4, "screaming"), "قويه": (3, "strong-current"),
    # water level / hazards
    "المويه دخلت": (2, "water-inside"), "الماء دخل": (2, "water-inside"), "المياه دخلت": (2, "water-inside"),
    "تدخل": (2, "water-inside"), "ترتفع": (2, "rising-water"), "تزيد": (2, "rising-water"),
    "للركب": (2, "rising-water"), "للركبه": (2, "rising-water"), "الصدر": (4, "chest-deep"),
    "الشباك": (4, "chest-deep"), "السرير": (3, "rising-water"), "غرقان": (2, "flooded"),
    "تماس": (4, "electricity"), "سلك كهربا": (4, "electricity"), "الكهربا طافيه": (2, "power-cut"),
    "الكهربا انقطعت": (2, "power-cut"),
    # needs / urgency
    "نبي نطلع": (2, "evacuation"), "اخلاء": (2, "evacuation"), "يطلعنا": (2, "evacuation"),
    "الحقونا": (2, "urgent"), "انقذونا": (2, "urgent"), "ساعدونا": (2, "urgent"), "ساعدوني": (2, "urgent"),
    "الحقوا": (2, "urgent"), "تكفون": (2, "urgent"), "بسرعه": (2, "urgent"),
    "مو قادرين نتواصل": (3, "no-contact"), "لحالهم": (2, "alone"),
}

VULNERABLE = {"elderly", "children", "infant", "pregnant", "disabled", "medical", "injured"}
HAZARD = {"rising-water", "chest-deep", "electricity", "on-roof", "fell-in-water", "drowning", "strong-current"}

# phrases that show people are safe: they lower the level by one
SAFE_PHRASES = ["بخير", "ما احد تضرر", "ما فيه احد", "مافي احد", "فاضيه", "ما كنت فيها", "انتبهوا", "حبيت ابلغ"]

NEGATIONS = {"ما", "مافي", "مافيه", "مو", "مب", "بدون", "لا", "ولا", "ماحد"}

NUMBER_WORDS = {
    "واحد": 1, "اثنين": 2, "ثنين": 2, "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4,
    "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6, "سبع": 7, "سبعه": 7, "ثمان": 8, "ثمانيه": 8,
    "تسع": 9, "تسعه": 9, "عشر": 10, "عشره": 10, "عشرين": 20, "ثلاثين": 30,
}

_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْـ]")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = text.replace("الموية", "المويه").replace("موية", "مويه")
    text = text.translate(_ARABIC_DIGITS)
    return re.sub(r"\s+", " ", text).strip()


_NORM_KEYWORDS = {normalize(k): v for k, v in KEYWORDS.items()}
_NORM_SAFE = [normalize(k) for k in SAFE_PHRASES]


def _negated(norm: str, start: int) -> bool:
    """True if one of the two words before position `start` is a negation."""
    before = norm[:start].split()[-2:]
    return any(w in NEGATIONS for w in before)


def count_people(norm: str) -> int:
    m = re.search(r"\b(\d{1,3})\b", norm)
    if m:
        return int(m.group(1))
    for word, n in NUMBER_WORDS.items():
        if re.search(rf"(^|\s)(و)?{word}(\s|$)", norm):
            return n
    if "عائله" in norm or "عايله" in norm or "احنا" in norm or "نحن" in norm:
        return 2
    return 1


def rule_analysis(text: str) -> Dict[str, object]:
    """Rule-based severity. Returns severity, people, tags and the reasons for any adjustment."""
    norm = normalize(text)
    tag_points: Dict[str, int] = {}               # tag -> level
    for kw, (pts, tag) in _NORM_KEYWORDS.items():
        for m in re.finditer(re.escape(kw), norm):
            if not _negated(norm, m.start()):
                tag_points[tag] = max(pts, tag_points.get(tag, 0))   # strongest word of each tag
                break
    tags = set(tag_points)
    people = count_people(norm)
    severity = max(tag_points.values(), default=1)
    reasons = []
    if severity < 5 and tags & VULNERABLE and tags & HAZARD:
        severity += 1
        reasons.append("vulnerable person + hazard")
    if severity < 5 and people >= 5 and severity >= 3:
        severity += 1
        reasons.append("5+ people")
    if any(p in norm for p in _NORM_SAFE):
        severity = max(1, severity - 1)
        reasons.append("people say they are safe")
    return {"severity": severity, "people": people, "tags": list(tag_points), "reasons": reasons}


class SeverityClassifier:
    """TF-IDF (character n-grams) + Logistic Regression. Optional: needs scikit-learn."""

    def __init__(self) -> None:
        self.pipeline = None

    def fit(self, texts: List[str], labels: List[int]) -> "SeverityClassifier":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipeline = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=5.0, class_weight="balanced"),
        )
        self.pipeline.fit([normalize(t) for t in texts], labels)
        return self

    def predict(self, text: str) -> Tuple[int, float]:
        """Predicted severity and the model's confidence (probability of that class)."""
        proba = self.pipeline.predict_proba([normalize(text)])[0]
        k = int(proba.argmax())
        return int(self.pipeline.classes_[k]), float(proba[k])


def load_dataset(path: str = DATA_FILE) -> Tuple[List[str], List[int]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d["text"] for d in data], [int(d["expected_severity"]) for d in data]


_classifier: Optional[SeverityClassifier] = None
_classifier_tried = False


def get_classifier() -> Optional[SeverityClassifier]:
    """Train the classifier once (lazily). Returns None if scikit-learn is missing."""
    global _classifier, _classifier_tried
    if not _classifier_tried:
        _classifier_tried = True
        try:
            texts, labels = load_dataset()
            _classifier = SeverityClassifier().fit(texts, labels)
        except Exception:                            # no scikit-learn or no data: rules only
            _classifier = None
    return _classifier


ML_CONFIDENCE = 0.6     # the ML opinion is used only when the model is at least this sure


def combine(rule_sev: int, ml_sev: Optional[int], ml_conf: float = 0.0) -> int:
    """Rules decide; a confident ML model can raise the severity (safety first), never lower it."""
    if ml_sev is None or ml_conf < ML_CONFIDENCE:
        return rule_sev
    return max(rule_sev, ml_sev)


def analyze(text: str, use_ml: bool = True) -> Dict[str, object]:
    """Main entry point used by the Distress Agent."""
    result = rule_analysis(text)
    ml_sev, ml_conf = None, 0.0
    if use_ml:
        clf = get_classifier()
        if clf is not None:
            ml_sev, ml_conf = clf.predict(text)
    result["rule_severity"] = result["severity"]
    result["ml_severity"] = ml_sev
    result["ml_confidence"] = round(ml_conf, 2)
    result["severity"] = combine(int(result["severity"]), ml_sev, ml_conf)
    return result
