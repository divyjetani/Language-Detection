"""
Command-line interface.

    python cli.py "kem cho maja ma ne?"          # detect one text
    python cli.py "kaise ho" --json               # full JSON output
    python cli.py --file sentences.txt            # one text per line
    python cli.py                                 # interactive mode
"""

import argparse
import json
import sys

from detector import LanguageDetector


def pretty(r: dict) -> str:
    lines = [f"Language   : {r['display_name']}  [{r['language']}]",
             f"Confidence : {r['confidence']:.1%}",
             f"Script     : {r['script']}"]
    if r["is_romanized"]:
        lines.append(f"Romanized  : yes (actual language: {r['base_language']})")
    if r["is_code_mixed"]:
        lines.append(f"Code-mixed : yes, CMI={r['code_mixing_index']} ({', '.join(r['languages_in_text'])})")
    if r.get("is_genz"):
        slang = ", ".join(f"{t['term']} = {t['meaning']}" for t in r["genz_terms"])
        lines.append(f"Gen-Z      : yes (score {r['genz_score']}) - {slang}")
    if r.get("tokens"):
        lines.append("Words      : " + " ".join(f"{t['token']}/{t['lang']}" for t in r["tokens"]))
    if r.get("note"):
        lines.append(f"Note       : {r['note']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Detect English, Hindi, Gujarati, Hinglish, Gujlish and other languages")
    ap.add_argument("text", nargs="*", help="text to analyse")
    ap.add_argument("--file", help="file with one text per line")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    args = ap.parse_args()

    det = LanguageDetector()

    def show(t):
        r = det.detect(t)
        print(json.dumps(r, ensure_ascii=False, indent=2) if args.json else pretty(r))
        print()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    print(f">>> {line.strip()}")
                    show(line.strip())
    elif args.text:
        show(" ".join(args.text))
    else:
        print("Interactive mode - type text and press Enter (Ctrl+C to quit)\n")
        try:
            while True:
                t = input(">>> ").strip()
                if t:
                    show(t)
        except (KeyboardInterrupt, EOFError):
            print()
            sys.exit(0)


if __name__ == "__main__":
    main()
