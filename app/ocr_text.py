"""Deterministic repair of OCR line text before it reaches Task B / the LLM prompt. No model, no new dependency; segments keep their boxes.
Measured problem (reports/extraction_audit): the RapidOCR recogniser often drops the space between a printed date and the time that follows it, producing
'25/12/20188:13:39PM' or 'Date:05-03-201819:02:50'. Task B's date finder and the LLM date check then cannot see the date. Only this glued date+time pattern is repaired;
nothing else is rewritten and digits are never changed (a space is inserted, never a character removed or altered)."""
import re

_YEAR = r"(?:19|20)\d{2}"
_TIME_AHEAD = r"(?=\d{1,2}:\d{2})"
_RULES = [
    re.compile(rf"(?<!\d)(\d{{1,2}}[/\-.]\d{{1,2}}[/\-.]{_YEAR}){_TIME_AHEAD}"),              # 25/12/2018|8:13:39PM   05-03-2018|19:02:50
    re.compile(rf"(?<!\d)({_YEAR}[/\-.]\d{{1,2}}[/\-.]\d{{1,2}}){_TIME_AHEAD}"),               # 2018-12-25|08:13
    re.compile(rf"(?<!\d)(\d{{1,2}}\s?[A-Za-z]{{3,9}}\.?,?\s?{_YEAR}){_TIME_AHEAD}"),         # 25Mar 2018|9:27PM
]

def repair_text(t):
    for rx in _RULES: t = rx.sub(r"\1 ", t)
    return t

def repair_segments(segs):
    """-> new list; each segment dict is copied, 'text' repaired, 'text_original' kept only when something changed."""
    out = []
    for s in segs or []:
        t = s.get("text", ""); r = repair_text(t) if isinstance(t, str) else t
        out.append(dict(s, text=r, text_original=t) if r != t else s)
    return out
