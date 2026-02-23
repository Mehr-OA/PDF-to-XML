# from dotenv import load_dotenv
from lxml import etree

# load_dotenv()
import gc
from lxml import etree
from pathlib import Path
from lxml import etree
import unicodedata
from datetime import datetime
from FileService import get_collection_items_by_handle, upload_xml_to_renate
import requests
import re
from typing import Iterable, Union, List
from config_loader import CONFIG
from inference import generate_annotations

# ---------------- Detokenization & offsets ----------------
NO_SPACE_BEFORE = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "’", "”"}
NO_SPACE_AFTER = {"(", "[", "{", "£", "$", "€", "“"}
GLUE_TOKENS = {"-", "–", "—", "/"}  # attach without surrounding spaces

# >>> Fill these with your own terms (examples included)
LTP_KEYWORDS = [
    "low temperature plasma",
    "cold plasma",
    "non-equilibrium plasma",
    "atmospheric pressure plasma",
    "nonthermal plasma",
    "dielectric barrier discharge",
    "dbd",
    "glow discharge",
    "rf plasma",
    "microwave plasma",
    "plasma jet",
    "plasma torch",
    "plasma medicine",
    "surface treatment",
    "plasma polymerization",
    "plasma etching",
    "pecvd",
    "langmuir probe",
    "optical emission spectroscopy",
    "reactive oxygen species",
    "ros",
    "reactive nitrogen species",
    "rns",
    "plasma catalysis",
    "plasma activated water",
]

EXCLUSIONS = [
    "tokamak",
    "fusion",
    "stellarator",
    "magnetic confinement",
    "mhd",
    "magnetized plasma",
    "divertor",
    "gyrotron",
    "burning plasma",
    "inertial confinement",
    "laser plasma",
]


def _word_boundary_pattern(term: str) -> re.Pattern:
    words = term.strip().split()
    # \b around each alnum chunk; allow hyphens inside words
    parts = [r"\b" + re.escape(w).replace(r"\-", r"[\-–]") + r"\b" for w in words]
    pat = r"\s+".join(parts)
    return re.compile(pat, flags=re.IGNORECASE)


# Precompile for speed
_POS_PATS = [_word_boundary_pattern(t) for t in LTP_KEYWORDS]
_NEG_PATS = [_word_boundary_pattern(t) for t in EXCLUSIONS]


def clean_keyword(s: str) -> str:
    if s is None:
        return ""

    # strip quotes/spaces
    s = str(s).strip().strip("'").strip('"')

    # remove punctuation/brackets (turn into spaces so words don't merge)
    s = re.sub(r"[^\w\s]", " ", s)

    # normalize whitespace + lowercase
    s = re.sub(r"\s+", " ", s).strip().lower()

    return s


def clean_keyword_objects(keyword_list):
    for obj in keyword_list:
        if "text" in obj:
            obj["text"] = clean_keyword(obj["text"])
    return keyword_list


def clean_and_split_text(text):
    """Cleans text by removing unwanted characters and splits into sentences."""
    # Remove specific unwanted characters
    text = text.replace("\n", " ")  # Replace newlines with space to avoid joining words

    # Split text into sentences by '.'
    sentences = text.split(".")

    # Strip whitespace from each sentence
    cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    return cleaned_sentences


def update_entities_inplace(
    sentences, clean_fn, *, drop_empty=True, dedupe_within_sentence=True
):
    for i in range(len(sentences) - 1, -1, -1):
        sent = sentences[i]
        ents = sent.get("entities", [])
        new_ents = []
        seen = set()

        for e in ents:
            cleaned = (clean_fn(e.get("text", "")) or "").strip()

            if drop_empty and not cleaned:
                continue

            e["text"] = cleaned

            if dedupe_within_sentence:
                key = (e.get("type"), cleaned.lower())
                if key in seen:
                    continue
                seen.add(key)

            new_ents.append(e)

        sent["entities"] = new_ents

        # drop sentence if no entities remain
        if not new_ents:
            sentences.pop(i)

    return sentences


def _norm_keywords(meta_keywords: Union[str, Iterable[str], None]) -> str:
    if meta_keywords is None:
        return ""
    if isinstance(meta_keywords, str):
        # Split on ; or , if it looks like a single string
        if ";" in meta_keywords or "," in meta_keywords:
            toks = [k.strip() for k in re.split(r"[;,]", meta_keywords)]
            return " ".join(toks)
        return meta_keywords
    # Iterable (list/tuple/set)
    try:
        return " ".join([str(x) for x in meta_keywords])
    except TypeError:
        return ""


def label_ltp(
    title: str, abstract: str, paper_keywords: Union[str, Iterable[str], None] = None
):
    t = title or ""
    a = abstract or ""
    k = _norm_keywords(paper_keywords)
    text = f"{t} {a} {k}"

    pos_hits = sorted(
        {term for term, pat in zip(LTP_KEYWORDS, _POS_PATS) if pat.search(text)}
    )
    neg_hits = sorted(
        {term for term, pat in zip(EXCLUSIONS, _NEG_PATS) if pat.search(text)}
    )

    # simple decision rule: at least 1 positive AND no exclusions
    is_ltp = (len(pos_hits) > 0) and (len(neg_hits) == 0)

    return is_ltp


def parse_jats_xml(xml_url):
    r = requests.get(xml_url, stream=True)
    r.raise_for_status()
    r.raw.decode_content = True

    parser = etree.XMLParser(
        resolve_entities=False, no_network=True, recover=True, huge_tree=True
    )
    tree = etree.parse(r.raw, parser)
    root = tree.getroot()

    # Detect default ns (JATS commonly uses a default)
    jats_ns = root.nsmap.get(None)  # default namespace URI if present
    ns = {
        "j": jats_ns or "http://jats.nlm.nih.gov",  # JATS
        "mml": "http://www.w3.org/1998/Math/MathML",
        "xlink": "http://www.w3.org/1999/xlink",
    }

    # Title
    title_el = root.find(".//j:article-title", namespaces=ns)
    title = _t(title_el).strip() if title_el is not None else None

    # Abstract paragraphs
    abstract_ps = root.findall(".//j:abstract//j:p", namespaces=ns)
    abstract = [" ".join(_t(p).split()) for p in abstract_ps if _t(p).strip()]

    # Sections (top-level body/sec). You can recurse if needed.
    sections = []
    for sec in root.findall(".//j:body/j:sec", namespaces=ns):
        sec_title = _t(sec.find("./j:title", namespaces=ns)).strip()
        paras = [" ".join(_t(p).split()) for p in sec.findall(".//j:p", namespaces=ns)]
        sections.append({"title": sec_title or None, "paragraphs": paras})

    # Figure & table captions (use XPath union via .xpath)
    captions_nodes = root.xpath(
        ".//j:fig//j:caption | .//j:table-wrap//j:caption", namespaces=ns
    )
    captions = [" ".join(_t(c).split()) for c in captions_nodes if _t(c).strip()]

    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "captions": captions,
    }


def updated_item_metadata(item_uuid, payload, s):
    url = CONFIG.UPDATE_ITEMS_METADATA.format(item_uuid=item_uuid)
    upload_response = s.patch(url, json=payload)
    if upload_response.ok:
        print("Metadata updated successfully!")
    else:
        print("Metadata upload failed", upload_response.text)


def nfc(s):
    return unicodedata.normalize("NFC", s)


def detok_with_offsets(tokens):
    """
    Returns (text, spans) where spans[i] = (start,end) char offsets of tokens[i]
    in the detokenized text. Rules are simple and deterministic.
    """
    text_parts = []
    spans = []
    cur = 0

    prev_token = None
    for tok in tokens:
        add_space = True

        if prev_token is None:
            add_space = False
        elif tok in NO_SPACE_BEFORE:
            add_space = False
        elif prev_token in NO_SPACE_AFTER:
            add_space = False
        elif tok in GLUE_TOKENS or prev_token in GLUE_TOKENS:
            add_space = False

        if add_space:
            text_parts.append(" ")
            cur += 1

        start = cur
        text_parts.append(tok)
        cur += len(tok)
        end = cur
        spans.append((start, end))
        prev_token = tok

    text = "".join(text_parts)
    return nfc(text), spans  # normalize for stability


# ---------------- BIO → spans per type ----------------


def bio_to_spans(
    labels, token_spans, raw_text, entity_type, confidences=None, agg="min"
):
    """
    labels: BIO labels per word
    token_spans: [(start,end)] per word
    confidences: [float] per word for this entity type (optional). If None, defaults to 1.0.
    """
    if confidences is None:
        confidences = [1.0] * len(labels)

    out = []
    i = 0
    while i < len(labels):
        lab = labels[i]
        if lab.startswith("B-"):
            j = i + 1
            while j < len(labels) and labels[j].startswith("I-"):
                j += 1
            start = token_spans[i][0]
            end = token_spans[j - 1][1]
            span_conf = (
                min(confidences[i:j])
                if agg == "min"
                else sum(confidences[i:j]) / max(1, j - i)
            )
            out.append(
                {
                    "start": start,
                    "end": end,
                    "text": raw_text[start:end],
                    "type": entity_type,
                    "confidence": round(float(span_conf), 4),  # <— NEW
                }
            )
            i = j
        else:
            i += 1
    return out


# ---------------- Build PPAnn XML ----------------
PP = "https://example.org/ppann/1.0"
XL = "http://www.w3.org/1999/xlink"


def build_ppann_xml(bio_dict, updated_sentences, doi, renate_doi, title):
    root = etree.Element("document")
    root.set("version", "1.0")

    # source
    src = etree.SubElement(root, "source")
    if doi:
        etree.SubElement(src, "doi").text = doi
    if renate_doi:
        etree.SubElement(src, "renateDoi").text = renate_doi
    if title:
        etree.SubElement(src, "title").text = title

    # annotations container
    ann = etree.SubElement(root, "annotations")
    entities_el = etree.SubElement(ann, "entities")
    sentences_el = etree.SubElement(ann, "sentences")

    global_eid = 0  # unique entity id across the whole document
    for sid, sent in enumerate(updated_sentences, start=1):
        s_el = etree.SubElement(sentences_el, "sentence")
        s_el.set("id", f"s{sid}")

        if sent.get("text"):
            s_el.set("text", sent["text"])

        # entities container under this sentence
        entities_el = etree.SubElement(s_el, "entities")

        for ent in sent.get("entities", []):
            global_eid += 1
            e_el = etree.SubElement(entities_el, "entity")
            e_el.set("id", f"e{global_eid}")
            e_el.set("type", ent.get("type", "Unknown"))

            # store cleaned/normalized entity text here (if already cleaned in your object)
            if ent.get("text"):
                e_el.set("text", ent["text"])

            # optional: keep offsets if you have them
            if ent.get("start") is not None:
                e_el.set("start", str(ent["start"]))
            if ent.get("end") is not None:
                e_el.set("end", str(ent["end"]))

            # optional: keep original entity_id from your pipeline
            if ent.get("entity_id") is not None:
                e_el.set("orig_id", str(ent["entity_id"]))

    xml_bytes = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", pretty_print=True
    )
    return xml_bytes


def _t(node):
    """Return visible text from an XML node, preserving inline text + tails."""
    if node is None:
        return ""
    parts = [node.text or ""]
    for child in node:
        parts.append(_t(child))
        parts.append(child.tail or "")
    return "".join(parts)


def sec_to_dict(sec):
    """Recursively convert <sec> into a dict with title, paragraphs, and children."""
    title = sec.find("./title")
    paras = [" ".join(_t(p).split()) for p in sec.findall("./p")]
    # Some JATS use <sec><sec> nesting; capture recursively:
    children = [sec_to_dict(s) for s in sec.findall("./sec")]
    return {
        "id": sec.get("id"),
        "title": _t(title).strip() if title is not None else None,
        "paragraphs": [p for p in paras if p],
        "subsections": children,
    }


# --- Main parser -------------------------------------------------------------
def _t(el):
    return "" if el is None else "".join(el.itertext())


def get_xml():
    # get xml from renate
    # open it
    read_jats_xml("test.xml")


def read_jats_xml(file_path):
    art = parse_jats_xml(file_path)
    print("TITLE:", art["title"])
    print("abstract", art["abstract"][0])
    # print('sections', art['sections'])
    annotate_class_wise_text(art["abstract"][0])


def create_and_add_annotations(s, collection_id):
    items = get_collection_items_by_handle(collection_id, None)
    for it in items[2:3]:
        # print('item', it)
        item_uuid = it["uuid"]
        bundle_uuid = it["bundle_uuid"]
        name = it["name"]
        # get metadata here
        keywords = it["keywords"]
        xml_url = it["xml_content"]["content"]
        article = parse_jats_xml(xml_url)
        print(article)

        title = article["title"]
        abstract = article["abstract"]
        sections = article["sections"]
        # print('title', title)
        print(sections)
        if len(title) != 0 and len(abstract) != 0:
            ltp = label_ltp(title, abstract[0], keywords)
            print(ltp)
        else:
            continue
        print("----------")
        ltp = True  # remove it
        threshold_entities = []
        text_parts = []

        # add title
        if title:
            text_parts.append(title)

        # add abstract
        if abstract:
            text_parts.append(abstract[0])

        # add sections
        for section in sections:
            sec_title = section.get("title")
            paragraphs = section.get("paragraphs", [])

            if sec_title:
                text_parts.append(sec_title)

            for para in paragraphs:
                if para:
                    text_parts.append(para)

            # final concatenated text
        text = " ".join(text_parts)

        response = generate_annotations(text)
        entities = response.get("entities", [])
        sentences = response.get("sentences", [])
        updated_sentences = update_entities_inplace(sentences, clean_keyword)
        updated_entities = clean_keyword_objects(entities)
        # per_token = best_label_per_token(results)

        # Example: print token → (class, tag, score)
        # for t in per_token:
        # print(f"{t['token']:<15} {t['class'] or 'O':<22} {t['tag']:<18} {t['score']:.3f}")

        # Collapse BIO into entity spans

        # entities = bio_spans(per_token)

        for e in updated_entities:
            text = e["text"].strip().lower()

            if (
                text not in keywords
                and e["confidence"] > 0.95
                and e["type"] != "G#Unit"
                and len(text.split()) > 1
            ):
                print(f"[{text}] {e['type']} (score={e['confidence']:.3f})")
                threshold_entities.append(text)

            # remove duplicates
            threshold_entities = list(set(threshold_entities))
        print(len(threshold_entities))

        # print(merged_keywords)
        payload = [
            {"op": "add", "path": "/metadata/dc.subject", "value": {"value": entity}}
            for entity in threshold_entities
        ]
        # updated_item_metadata(item_uuid, payload, s)
        # for key, value in results.items():
        # if key.startswith("Label_"):
        # label_name = value
        # label_key = key.split("Label_")[1]
        # score_key = f"Score_{label_key}"
        # score = results.get(score_key)
        # print(f"{label_name}: {score}")

        # pass entities
        print(it["doi"])
        print(it["renate_doi"])
        xml = build_ppann_xml(updated_entities, updated_sentences, it["doi"], it["renate_doi"], title)

        # retrieve_high_quality_annotations(results)
        name = name.split(".")[0] + "-annotations"
        upload_xml_to_renate(s, xml, bundle_uuid, (name))

    # find high quality metadata
