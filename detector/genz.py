"""
Gen-Z style detection.

Gen-Z text is not a separate language. It is a *register* (style) of a language:
"ngl this biryani is bussin fr" is still English, "bhai ye lowkey op hai" is
still Hinglish, "km 6o bhai, jalso padi gayo" is still Gujlish.

So this module runs AFTER language detection and answers a second question:
"is this written in Gen-Z style?". It looks for
  * slang words / phrases   (rizz, delulu, no cap, bawaal, jalso, bhukka ...)
  * Gen-Z shortforms        (ngl, fr, istg, kr rha h, km 6o, su kr 6e ...)
and returns a score plus the matched terms with their meanings, so the UI can
show something like:  "Gujarati (Gen Z)  -  jalso = full fun".

To add new slang, just add an entry to the dictionaries below:
    "term": ("meaning", weight)
weight 1.0 = clearly Gen-Z on its own (e.g. "rizz")
weight 0.4-0.6 = common chat shortform, needs a few together (e.g. "idk", "h")
"""

from __future__ import annotations

GENZ_THRESHOLD = 1.0   # total weight needed to call a text Gen-Z

# --------------------------------------------------------------------------
# English Gen-Z slang - counted for every language, because Indian Gen-Z mixes
# these words into Hindi and Gujarati too ("bhai ye scene lowkey mid hai")
# --------------------------------------------------------------------------
ENGLISH = {
    # slang
    "rizz": ("charm / flirting skill", 1.0),
    "rizzler": ("someone with a lot of rizz", 1.0),
    "bussin": ("really good (usually food)", 1.0),
    "slay": ("did amazingly", 1.0),
    "slayed": ("did amazingly", 1.0),
    "slaying": ("doing amazingly", 1.0),
    "delulu": ("delusional (in a fun way)", 1.0),
    "situationship": ("undefined relationship", 1.0),
    "skibidi": ("meme word, random / weird", 1.0),
    "gyat": ("exclamation of surprise", 1.0),
    "sigma": ("lone-wolf cool person", 1.0),
    "aura": ("vibe / coolness points", 0.6),
    "mewing": ("jawline meme", 1.0),
    "npc": ("someone acting without thinking", 1.0),
    "sus": ("suspicious", 1.0),
    "yeet": ("throw / excitement", 1.0),
    "sheesh": ("wow, impressive", 1.0),
    "periodt": ("end of discussion", 1.0),
    "simp": ("overly devoted to someone", 1.0),
    "stan": ("big fan", 0.8),
    "ick": ("instant turn-off", 0.8),
    "glazing": ("over-praising someone", 1.0),
    "cooked": ("doomed / finished", 0.6),
    "lowkey": ("kind of / secretly", 1.0),
    "highkey": ("very / openly", 1.0),
    "mid": ("average, not great", 0.6),
    "based": ("bold, true to yourself", 0.5),
    "cringe": ("embarrassing", 0.6),
    "lit": ("amazing / exciting", 0.6),
    "goat": ("greatest of all time", 0.5),
    "vibe": ("feeling / energy", 0.4),
    "vibes": ("feeling / energy", 0.4),
    "vibing": ("enjoying the mood", 0.8),
    "slaps": ("is really good (music/food)", 0.8),
    "bruh": ("bro / disbelief", 0.6),
    "fam": ("friends / family", 0.5),
    "bestie": ("best friend", 0.6),
    "oomf": ("one of my followers", 1.0),
    "pov": ("point of view (meme format)", 0.6),
    "ghosted": ("ignored without explanation", 0.6),
    "salty": ("bitter / upset", 0.5),
    "flex": ("show off", 0.5),
    "ded": ("dying of laughter", 0.8),
    "op": ("overpowered / awesome", 0.8),
    "ratio": ("reply got more likes", 0.6),
    # shortforms
    "fr": ("for real", 1.0),
    "frfr": ("for real for real", 1.0),
    "ngl": ("not gonna lie", 1.0),
    "ong": ("on god (honestly)", 1.0),
    "istg": ("I swear to god", 1.0),
    "iykyk": ("if you know you know", 1.0),
    "bffr": ("be for real", 1.0),
    "idc": ("I don't care", 0.6),
    "idk": ("I don't know", 0.5),
    "ikr": ("I know, right", 0.6),
    "tbh": ("to be honest", 0.6),
    "imo": ("in my opinion", 0.5),
    "smh": ("shaking my head", 0.6),
    "wyd": ("what you doing", 0.8),
    "hbu": ("how about you", 0.6),
    "rn": ("right now", 0.6),
    "lmao": ("laughing a lot", 0.6),
    "lmfao": ("laughing a lot", 0.6),
    "lol": ("laughing out loud", 0.4),
    "omg": ("oh my god", 0.4),
    "nvm": ("never mind", 0.5),
    "ttyl": ("talk to you later", 0.5),
    "btw": ("by the way", 0.4),
    "pls": ("please", 0.3),
    "plz": ("please", 0.3),
    "u": ("you", 0.3),
    "ur": ("your / you're", 0.3),
    "gr8": ("great", 0.6),
    "2day": ("today", 0.6),
    "2mrw": ("tomorrow", 0.6),
    "tmrw": ("tomorrow", 0.4),
}

ENGLISH_PHRASES = {
    "no cap": ("no lie, seriously", 1.0),
    "it's giving": ("it has the vibe of", 1.0),
    "its giving": ("it has the vibe of", 1.0),
    "understood the assignment": ("did exactly what was needed, perfectly", 1.0),
    "main character": ("acting like the hero of the story", 1.0),
    "rent free": ("can't stop thinking about it", 1.0),
    "touch grass": ("go outside, get off the internet", 1.0),
    "hits different": ("feels special", 1.0),
    "spill the tea": ("share the gossip", 1.0),
    "say less": ("got it, no need to explain", 1.0),
    "on god": ("honestly", 0.8),
    "vibe check": ("checking the mood", 1.0),
    "caught in 4k": ("caught red-handed", 1.0),
    "im dead": ("that's hilarious", 0.8),
    "i'm dead": ("that's hilarious", 0.8),
    "red flag": ("warning sign", 0.6),
    "green flag": ("good sign", 0.6),
    "ate that": ("did it perfectly", 1.0),
    "ate and left no crumbs": ("did it perfectly", 1.0),
}

# --------------------------------------------------------------------------
# Hindi / Hinglish Gen-Z slang + shortforms (counted when base language = Hindi)
# --------------------------------------------------------------------------
HINDI = {
    "bawaal": ("crazy good / chaos", 1.0),
    "bawal": ("crazy good / chaos", 1.0),
    "jhakaas": ("awesome", 1.0),
    "bhasad": ("chaos / mess", 1.0),
    "lafda": ("fight / trouble", 0.8),
    "jhol": ("something fishy", 0.8),
    "pakau": ("boring person", 1.0),
    "bakaiti": ("nonsense talk", 1.0),
    "tapori": ("street-smart rowdy", 0.8),
    "senti": ("emotional", 0.8),
    "chep": ("clingy", 1.0),
    "faltu": ("useless", 0.5),
    "bakwas": ("nonsense", 0.4),
    "bhot": ("very (shortform of bahut)", 0.4),
    "bht": ("very (shortform of bahut)", 0.5),
    "mtlb": ("meaning (matlab)", 0.5),
    "pkka": ("sure (pakka)", 0.5),
    "thk": ("okay (theek)", 0.4),
    "h": ("is (hai)", 0.5),
    "hn": ("yes (haan)", 0.5),
    "kr": ("do (kar)", 0.4),
    "rha": ("-ing (raha)", 0.4),
    "rhi": ("-ing (rahi)", 0.4),
    "rhe": ("-ing (rahe)", 0.4),
    "kb": ("when (kab)", 0.4),
    "kl": ("tomorrow / yesterday (kal)", 0.4),
    "kha": ("where (kahan)", 0.3),
    "nhi": ("no (nahi)", 0.3),
    "kyu": ("why (kyun)", 0.3),
    "ky": ("what (kya)", 0.5),
    "bhaiyo": ("bros", 0.5),
}

HINDI_PHRASES = {
    "kya scene hai": ("what's the plan / what's up", 1.0),
    "scene kya hai": ("what's the plan / what's up", 1.0),
    "chill maar": ("relax", 1.0),
    "full on": ("totally", 0.6),
    "ek number": ("first class / awesome", 1.0),
    "bhot hard": ("very cool", 1.0),
    "bahut hard": ("very cool", 1.0),
    "dimag ki dahi": ("irritating me a lot", 1.0),
    "scene on hai": ("plan is on", 1.0),
    "bhai ne bola": ("bro has said it (so it's final)", 1.0),
}

# --------------------------------------------------------------------------
# Gujarati / Gujlish Gen-Z slang + shortforms (counted when base = Gujarati)
# --------------------------------------------------------------------------
GUJARATI = {
    "jalso": ("full fun / party mood", 1.0),
    "jalsa": ("full fun / party mood", 1.0),
    "bhukka": ("destroyed it (in a good way)", 1.0),
    "fadu": ("awesome", 1.0),
    "taddan": ("totally", 0.8),
    "sav": ("completely", 0.6),
    "lukkho": ("loafer", 0.8),
    "lukkha": ("loafers", 0.8),
    "jordaar": ("superb", 0.6),
    "jordar": ("superb", 0.4),
    "dhamaal": ("great fun", 0.8),
    "halo": ("let's go (Kathiyawadi)", 0.6),
    # shortforms: '6' is used for the 'chh' sound
    "6e": ("is (che)", 0.8),
    "6u": ("am (chu)", 0.8),
    "6o": ("are (cho)", 0.8),
    "6ie": ("are (chhie)", 0.8),
    "km6": ("how are you (kem cho)", 1.0),
    "kmcho": ("how are you (kem cho)", 0.8),
    "km": ("how (kem)", 0.4),
    "bdhu": ("everything (badhu)", 0.5),
    "su6": ("what's up (shu che)", 1.0),
    "kr": ("do (kar)", 0.4),
    "mst": ("great (mast)", 0.5),
}

GUJARATI_PHRASES = {
    "bhukka bolavi didha": ("totally nailed it", 1.0),
    "bhukka kadhi nakhya": ("totally nailed it", 1.0),
    "jalso padi gayo": ("had a blast", 1.0),
    "jalsa padi gaya": ("had a blast", 1.0),
    "vaat j alag che": ("it's a different level", 1.0),
    "vaat j alag chhe": ("it's a different level", 1.0),
    "majja ni life": ("life is fun", 1.0),
    "bhai bhai": ("legend! (praise)", 0.8),
    "kai pan": ("whatever / anything", 0.6),
}

# Slang common to Hindi AND Gujarati youth speech
SHARED = {
    "chhapri": ("try-hard, tacky", 1.0),
    "kadak": ("strong / cool", 0.6),
    "bindaas": ("carefree", 0.6),
    "jugaad": ("hack / quick fix", 0.5),
}

# Single-word English slang, used by the word tagger to mark them "en"
ENGLISH_WORDS = set(ENGLISH)


def _match(tokens: list[str], text: str, words: dict, phrases: dict, source: str, found: dict):
    padded = f" {text} "
    for phrase, (meaning, w) in phrases.items():
        if f" {phrase} " in padded and phrase not in found:
            found[phrase] = {"term": phrase, "meaning": meaning, "weight": w, "source": source}
    # words already inside a matched phrase are not counted again
    in_phrase = {w for term in found if " " in term for w in term.split()}
    for t in tokens:
        if t in words and t not in found and t not in in_phrase:
            meaning, w = words[t]
            found[t] = {"term": t, "meaning": meaning, "weight": w, "source": source}


def analyze(tokens: list[str], clean_text: str, base_language: str) -> dict:
    """
    tokens / clean_text : output of the normaliser
    base_language       : 'english', 'hindi', 'gujarati', ...
    """
    found: dict[str, dict] = {}
    _match(tokens, clean_text, ENGLISH, ENGLISH_PHRASES, "english", found)
    if base_language == "hindi":
        _match(tokens, clean_text, HINDI, HINDI_PHRASES, "hindi", found)
    if base_language == "gujarati":
        _match(tokens, clean_text, GUJARATI, GUJARATI_PHRASES, "gujarati", found)
    if base_language in ("hindi", "gujarati"):
        _match(tokens, clean_text, SHARED, {}, "shared", found)

    score = round(sum(f["weight"] for f in found.values()), 2)
    terms = sorted(found.values(), key=lambda f: -f["weight"])
    return {
        "is_genz": score >= GENZ_THRESHOLD,
        "genz_score": score,
        "genz_terms": [{"term": f["term"], "meaning": f["meaning"]} for f in terms],
    }
