"""Word counter used for the SciRep caps (title <=20, abstract <=200, main text <=4,500).

Convention (fixed and committed per round-10 review): comments stripped (\\% kept),
formatting-command contents kept, \\cite/\\ref/\\label dropped, each inline math
expression counts as ONE word, figure/table environments (incl. legends) excluded from
the main-text count, Methods/references/end-matter excluded per the SciRep definition.
Counters differ; this one is the committed convention and is kept ~100 words under the
cap so that plausible alternatives (e.g. TeXcount with math-as-words) also stay under.
Usage: python count_words.py [path/to/main.tex]
"""
import re
import sys
from pathlib import Path


def expand_inputs(source, base_dir, seen=None):
    """Expand local ``\\input`` files so the split revision is counted correctly."""
    seen = set() if seen is None else seen

    def replace(match):
        name = match.group(1)
        path = (base_dir / name).with_suffix(".tex") if not name.endswith(".tex") \
            else base_dir / name
        path = path.resolve()
        if path in seen:
            raise RuntimeError(f"recursive LaTeX input: {path}")
        seen.add(path)
        text = path.read_text(encoding="utf-8")
        expanded = expand_inputs(text, path.parent, seen)
        seen.remove(path)
        return expanded

    return re.sub(r'\\input\{([^}]+)\}', replace, source)


def strip_tex(t):
    t = re.sub(r'(?<!\\)%.*', '', t)
    for _ in range(4):
        t = re.sub(r'\\(emph|textbf|textit|texttt|mbox)\{((?:[^{}]|\{[^{}]*\})*)\}', r'\2', t)
    t = re.sub(r'\\(cite|ref|label|nameref)\{[^}]*\}', '', t)
    t = re.sub(r'\\url\{[^}]*\}', ' URL ', t)
    t = re.sub(r'\$(?:[^$]*)\$', ' MATH ', t)
    t = re.sub(r'\\[a-zA-Z]+\*?', ' ', t)
    t = re.sub(r'[{}~\[\]]', ' ', t)
    t = re.sub(r'---?', ' ', t)
    return t.replace('\\%', '%')


def words(t):
    return len([w for w in strip_tex(t).split() if re.search(r'[A-Za-z0-9%]', w)])


def main(path):
    manuscript = Path(path).resolve()
    src = expand_inputs(manuscript.read_text(encoding="utf-8"), manuscript.parent)
    ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', src, re.S).group(1)
    body = re.search(r'\\section\*\{Introduction\}(.*?)\\section\*\{Methods\}', src, re.S).group(1)
    body = re.sub(r'\\begin\{(figure|table)\}.*?\\end\{\1\}', ' ', body, flags=re.S)
    title = re.search(r'\\title\{([^}]*)\}', src).group(1)
    print(f"TITLE: {len(title.replace('--', '-').split())} /20 words")
    print(f"ABSTRACT: {words(ab)} /200 words")
    print(f"MAIN TEXT (Intro+Results+Discussion, no floats/legends): {words(body)} /4500 words")
    caps = re.findall(r'\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', src)
    print(f"captions: {[words(c) for c in caps]} (each /350)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         __file__.rsplit('/src/', 1)[0] + '/drafts/latex/main.tex')
