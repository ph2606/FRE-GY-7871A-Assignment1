"""Score filing uncertainty after removing the body of Item 1A.

Section boundaries come from HTML block headings, not a search for the first
mention of ``Item 1A``. Linked contents entries and inline cross-references do
not define boundaries. Unresolved boundaries are reported and never assigned
a zero score. The narrative cleaning and token rules match ``src.parse``.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import json
import os
from pathlib import Path
import re

from bs4 import BeautifulSoup, NavigableString
import pandas as pd

from . import parse
from .lexicons import lm_word_lists, load_master_dictionary

SECTION_VERSION = "item1a-dom-boundaries-v2"
_BLOCKS = {"p", "div", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "li"}
_LABEL = re.compile(r"^\s*item\b", re.I)
_ITEM = re.compile(r"^i\s*t\s*e\s*m\s*(\d{1,2}(?:[\s.]*\(?[abc]\)?(?![a-z]))?)[\s.\-:\u2013\u2014\ufffd\u0096]*", re.I)
_TITLES = {
    "1": r"(?:legal\s+proceedings|business|financial\s+statements|condensed\s+consolidated)",
    "1A": r"r\s*i\s*s\s*k\s*f\s*a\s*c\s*t\s*o\s*r\s*s",
    "1B": r"unresolved\s+staff\s+comments",
    "1C": r"cyber\s*security",
    "2": r"(?:properties|unregistered\s+sales|management[\u2019'\s]*s?\s+discussion)",
    "3": r"(?:defaults\s+upon\s+senior|legal\s+proceedings|quantitative\s+and\s+qualitative)",
    "4": r"(?:mine\s+safety|controls\s+and\s+procedures)",
    "5": r"(?:other\s+information|market\s+for)",
    "6": r"(?:exhibits|reserved|selected\s+financial)",
}
_REFERENCE = re.compile(r"^[\s,;:\-\u2013\u2014]*(?:of|in|under|above|below|contains|describes|discusses|sets?\s+forth)\b", re.I)
_MARKER = re.compile("\ue000(\\d+)\ue001")

# PDF-style HTML often splits a heading word across several font spans.
def _spaced(phrase):
    return r"[\s\u2019']*".join(re.escape(c) for c in phrase.replace(" ", ""))


_TITLE_PHRASES = {
    "1": ["legal proceedings", "business", "financial statements", "condensed consolidated"],
    "1A": ["risk factors"], "1B": ["unresolved staff comments"],
    "1C": ["cybersecurity"],
    "2": ["properties", "unregistered sales", "managements discussion", "management discussion"],
    "3": ["defaults upon senior", "legal proceedings", "quantitative and qualitative"],
    "4": ["mine safety", "controls and procedures"],
    "5": ["other information", "other events", "market for"],
    "6": ["exhibits", "reserved", "selected financial"],
    "END": ["information about our executive officers", "information about the executive officers"],
}
_TITLES = {item: "(?:"+"|".join(_spaced(x) for x in phrases)+")"
           for item, phrases in _TITLE_PHRASES.items()}


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(text).replace("\xa0", " ")).strip()


def _clean_soup(raw_html: str) -> BeautifulSoup:
    """Apply the existing parser's rules before adding boundary markers."""
    soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.decomposed:
            continue
        name = (tag.name or "").lower()
        if name.startswith(parse._XBRL_PREFIXES) or name in parse._HIDDEN_IX or name == "xbrl":
            tag.decompose()
            continue
        style = (tag.attrs.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style or tag.has_attr("hidden"):
            tag.decompose()
        elif name.startswith("ix:"):
            tag.unwrap()
    return soup


def _marker_positions(soup: BeautifulSoup) -> tuple[str, dict[int, int]]:
    marked = _normal(soup.get_text(" "))
    pieces = _MARKER.split(marked)
    output, positions = [], {}
    length = 0
    for index in range(0, len(pieces), 2):
        segment = pieces[index].strip()
        if segment:
            length += len(segment) + bool(output)
            output.append(segment)
        if index + 1 < len(pieces):
            positions[int(pieces[index + 1])] = length + bool(output)
    return " ".join(output), positions


def _headings(raw_html: str) -> tuple[str, list[dict]]:
    soup = _clean_soup(raw_html)
    candidates = []
    blocks = []
    seen = set()
    for block in soup.find_all(list(_BLOCKS)):
        strings = (x for x in block.strings if _normal(str(x)))
        node = next(strings, None)
        if node is None or id(node) in seen:
            continue
        prefix = _normal(str(node))
        # Limit traversal of large container divs to a short heading prefix.
        if not re.match(r"^i(?:\s|t)", prefix, re.I):
            continue
        for part in strings:
            if len(prefix) >= 200:
                break
            prefix += " " + _normal(str(part))
        if not _ITEM.match(prefix):
            continue
        if node.find_parent("a", href=True):
            continue
        # Contents tables sometimes link only the title or page number, leaving
        # the item label itself unlinked. The entire linked contents row is skipped.
        row = node.find_parent("tr")
        if row and row.find("a", href=True) and len(_normal(row.get_text(" "))) < 350:
            continue
        seen.add(id(node))
        candidates.append(node)
        blocks.append(prefix)
    # Some annual reports use unnumbered body headings and an Item cross-reference
    # index. Follow the document's own contents links to the actual section anchor.
    anchored = {}
    for link in soup.find_all("a", href=re.compile(r"^#")):
        label = _normal(link.get_text(" "))
        if not label:
            continue
        for item in ["1A", "1B", "1C", "2", "3", "4", "5", "6", "END"]:
            pattern = _TITLES[item]
            if not re.fullmatch(pattern+r"[.\s]*", label, re.I):
                continue
            ident = link.get("href")[1:]
            target = soup.find(id=ident) or soup.find("a", attrs={"name": ident})
            if target is not None:
                anchored[id(target)] = (target, item, label)
            break
    for index, node in enumerate(candidates):
        node.insert_before(NavigableString(f"\ue000{index}\ue001"))
    for index, (target, item, label) in enumerate(anchored.values(), len(candidates)):
        target.insert_before(NavigableString(f"\ue000{index}\ue001"))
    text, positions = _marker_positions(soup)
    headings = []
    for candidate, position in positions.items():
        if candidate >= len(candidates):
            _, item, label = list(anchored.values())[candidate-len(candidates)]
            prefix = text[position:position+250]
            navigation = re.match(r"(?:\s*\d+\s*)?(?:(?:table\s+of\s+contents|index)\s*)?", prefix, re.I)
            offset = navigation.end()
            numbered = _ITEM.match(prefix[offset:])
            if numbered:
                offset += numbered.end()
            title = re.match(_TITLES[item], prefix[offset:], re.I)
            if title:
                stop = position+offset+title.end()
                headings.append({"candidate": candidate, "item": item, "start": position,
                                 "title_end": stop, "heading": text[position:stop]})
            continue
        match = _ITEM.match(text[position:position + 200])
        if not match:
            continue
        item = re.sub(r"[\s().]", "", match[1]).upper()
        if item not in _TITLES:
            continue
        title = re.match(_TITLES[item], text[position + match.end():], re.I)
        if not title:
            continue
        stop = position + match.end() + title.end()
        if item == "1A" and _REFERENCE.match(blocks[candidate][stop - position:]):
            continue
        headings.append({"candidate": candidate, "item": item, "start": position, "title_end": stop,
                         "heading": text[position:stop]})
    # Numeric tables can include an item heading. Locate boundaries first, then
    # preserve the boundary marker at the deleted table's original position.
    # Markers are excluded from the numeric-share calculation itself.
    for table in list(soup.find_all("table")):
        if table.decomposed:
            continue
        table_text = table.get_text(" ")
        if parse._numeric_share(_MARKER.sub("", table_text)) > parse.NUMERIC_TABLE_THRESHOLD:
            for marker in _MARKER.finditer(table_text):
                table.insert_before(NavigableString(marker[0]))
            table.decompose()
    text, positions = _marker_positions(soup)
    for heading in headings:
        position = positions[heading.pop("candidate")]
        heading["start"] = position
        heading["title_end"] = position + (len(heading["heading"]) if text[position:].startswith(heading["heading"]) else 0)
    headings = list({(h['item'], h['start']): h for h in headings}.values())
    return text, sorted(headings, key=lambda h: h['start'])


def exclude_item1a(raw_html: str, form: str) -> dict:
    """Return retained text and auditable section boundaries.

    ``found`` removes one verified body range. ``absent`` is restricted to
    10-Qs whose body has Legal Proceedings followed by another Part II item,
    with no Item 1A body heading. ``ambiguous`` has no retained score/text.
    """
    if form not in {"10-K", "10-Q"}:
        raise ValueError("Item 1A extraction accepts original 10-K and 10-Q filings only")
    text, headings = _headings(raw_html)
    starts = [h for h in headings if h["item"] == "1A"]
    end_items = {"1B", "1C", "2", "3", "4", "5", "6", "END"} if form == "10-K" else {"2", "3", "4", "5", "6"}
    viable = []
    for start in starts:
        ends = [h for h in headings if h["start"] > start["start"] and h["item"] in end_items]
        if not ends:
            continue
        end = ends[0]
        content = text[start["title_end"]:end["start"]].strip()
        words = parse.tokenize(content)
        # Contents rows contain a page number, not a narrative section.
        if not words:
            continue
        viable.append((start, end, len(words)))
    result = {"status": "ambiguous", "text": None, "start": None, "end": None,
              "start_heading": None, "end_heading": None, "removed_n_words": None,
              "original_n_words": len(parse.tokenize(text)), "retained_n_words": None,
              "removal_excerpt": "", "reason": "No verified pair of body section headings",
              "method": SECTION_VERSION, "candidate_headings": len(headings)}
    if viable:
        # Repeated running headings point to the same end. The first occurrence
        # includes the complete section. Distinct substantial sections are unsafe.
        longest = max(viable, key=lambda pair: pair[2])
        conflicting = [pair for pair in viable if pair[1]["start"] != longest[1]["start"] and pair[2] > 20]
        if conflicting:
            result["reason"] = "Multiple distinct narrative Item 1A ranges"
            return result
        start, end, _ = longest
        removed = text[start["start"]:end["start"]].strip()
        retained = _normal(text[:start["start"]] + " " + text[end["start"]:])
        result.update(status="found", text=retained, start=start["start"], end=end["start"],
                      start_heading=start["heading"], end_heading=end["heading"],
                      removed_n_words=len(parse.tokenize(removed)),
                      retained_n_words=len(parse.tokenize(retained)), removal_excerpt=removed[:500],
                      reason="Body Item 1A removed through the next numbered item")
        result["removed_text"] = removed
        return result
    legal = [h for h in headings if h["item"] == "1" and "legal" in h["heading"].lower()]
    if form == "10-Q" and not starts and legal:
        following = [h for h in headings if h["start"] > legal[-1]["start"] and h["item"] in end_items]
        if following:
            result.update(status="absent", text=text, retained_n_words=result["original_n_words"],
                          removed_n_words=0, reason="Part II body proceeds from Item 1 to a later item without Item 1A")
    if form == "10-Q" and not starts and not legal:
        parts = list(re.finditer(r"part\s+ii\b", text, re.I))
        if parts and parts[-1].start() > len(text)*.5:
            tail = text[parts[-1].end():]
            if re.match(r"[.\s]*(?:other\s+information\s*)?item\s*[56]\b", tail, re.I) and re.search(r"signatures", tail, re.I):
                result.update(status="absent", text=text, retained_n_words=result["original_n_words"],
                              removed_n_words=0, reason="Final Part II begins at Item 5 or 6 and ends in signatures; Item 1A omitted")
    if form == "10-Q" and not starts and result["status"] == "ambiguous":
        part_ii = list(re.finditer(r"\bpart\s+ii\s*[.\-\u2013\u2014]*\s*other\s+information\b", text, re.I))
        if part_ii and part_ii[-1].start() > len(text) / 2:
            start = part_ii[-1].end()
            tail = text[start:]
            following = [h for h in headings if h["start"] >= start and h["item"] in end_items]
            if (following and following[0]["start"] - start < 150
                    and not re.search(r"\bitem\s*1\s*a\b", tail, re.I)
                    and re.search(r"\bsignatures?\b", tail, re.I)):
                result.update(status="absent", text=text, retained_n_words=result["original_n_words"],
                              removed_n_words=0, reason="Explicit Part II body starts at a later item and reaches signatures with no Item 1A")
    return result


def _score_task(task: tuple) -> dict:
    root, metadata, dictionary_hash, parser_hash, module_hash, uncertainty = task
    accession = metadata["accession"]
    source = Path(root) / "data/filings" / (accession.replace("-", "") + ".html")
    payload = source.read_bytes()
    result = exclude_item1a(payload.decode("utf-8"), metadata["form"])
    retained = result.pop("text")
    removed = result.pop("removed_text", "")
    result.update({key: metadata[key] for key in ("accession", "ticker", "cik", "company", "form", "filing_date", "report_date", "doc_url")})
    result.update(raw_sha256=hashlib.sha256(payload).hexdigest(), parser_sha256=parser_hash,
                  section_parser_sha256=module_hash, dictionary_sha256=dictionary_hash,
                  text_sha256=metadata.get("text_sha256", ""))
    if retained is None:
        result.update(uncertainty_count=None, uncertainty_pct=None, original_uncertainty_pct=None, removed_uncertainty_count=None,
                      original_uncertainty_count=None)
    else:
        count = sum(word in uncertainty for word in parse.tokenize(retained))
        removed_count = sum(word in uncertainty for word in parse.tokenize(removed))
        result.update(uncertainty_count=count, uncertainty_pct=100 * count / result["retained_n_words"],
                      removed_uncertainty_count=removed_count, original_uncertainty_count=count + removed_count,
                      original_uncertainty_pct=100 * (count + removed_count) / result["original_n_words"])
    expected = int(float(metadata["n_words"]))
    if expected != result["original_n_words"]:
        raise ValueError(f"Original parser denominator mismatch for {accession}: {expected} != {result['original_n_words']}")
    if retained is not None and result["original_n_words"] != result["retained_n_words"] + result["removed_n_words"]:
        raise ValueError(f"Removal token partition mismatch for {accession}")
    return result


def build_risk_panel(root: Path, save: bool = True, workers: int = 4) -> pd.DataFrame:
    """Build or validate the cached Item 1A scores for all original filings.

    Every cache reuse verifies raw filing, dictionary and both parser hashes.
    Cache files are local outputs and must not be committed to the repository.
    """
    root = Path(root)
    metadata = pd.read_csv(root / "data/interim/filings_meta.csv", dtype={"cik": str}, keep_default_na=False)
    metadata = metadata.loc[metadata["form"].isin(["10-K", "10-Q"]) & metadata["parse_status"].eq("ok")]
    dictionary = root / "data/lexicons/LoughranMcDonald_MasterDictionary.csv"
    dictionary_hash = hashlib.sha256(dictionary.read_bytes()).hexdigest()
    parser_hash = hashlib.sha256(Path(parse.__file__).read_bytes()).hexdigest()
    module_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    uncertainty = lm_word_lists(load_master_dictionary(dictionary))["Uncertainty"]
    path = root / "outputs/risk_sections.jsonl"
    cached = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[row["accession"]] = row
    results, tasks = [], []
    for row in metadata.to_dict("records"):
        old = cached.get(row["accession"], {})
        source = root / "data/filings" / (row["accession"].replace("-", "") + ".html")
        if (old.get("dictionary_sha256") == dictionary_hash and old.get("parser_sha256") == parser_hash
                and old.get("section_parser_sha256") == module_hash
                and old.get("text_sha256") == row.get("text_sha256")
                and old.get("raw_sha256") == hashlib.sha256(source.read_bytes()).hexdigest()):
            results.append(old)
        else:
            tasks.append((str(root), row, dictionary_hash, parser_hash, module_hash, uncertainty))
    if tasks:
        if workers == 1:
            calculated = map(_score_task, tasks)
            results.extend(calculated)
        else:
            with ProcessPoolExecutor(max_workers=min(workers, os.cpu_count() or 1)) as pool:
                for index, row in enumerate(pool.map(_score_task, tasks), 1):
                    results.append(row)
                    if index % 100 == 0:
                        print(f"Item 1A: scored {index}/{len(tasks)} filings", flush=True)
    results.sort(key=lambda row: (row["cik"], row["filing_date"], row["accession"]))
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in results), encoding="utf-8")
        temporary.replace(path)
    return pd.DataFrame(results)
