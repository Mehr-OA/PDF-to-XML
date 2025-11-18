#from dotenv import load_dotenv
from lxml import etree
#load_dotenv()
from transformers import BertTokenizerFast
from models import BertCRF
import torch
import gc
from lxml import etree
from pathlib import Path
from lxml import etree
import unicodedata
from datetime import datetime
from FileService import get_collection_items_by_handle
import requests
import re
from typing import Iterable, Union, List
from config_loader import CONFIG

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# ---------------- Detokenization & offsets ----------------
NO_SPACE_BEFORE = {".", ",", ":", ";", "!", "?", "%", ")", "]", "}", "’", "”"}
NO_SPACE_AFTER  = {"(", "[", "{", "£", "$", "€", "“"}
GLUE_TOKENS     = {"-","–","—","/"}  # attach without surrounding spaces

entities = [
    'DiagnosticDevice',
    'ElectrodeConfiguration',
    'ElectrodeMaterial',
    'Experiment',
    'Modelling',
    'PhysicalEffect',
    'PlasmaApplication',
    'PlasmaMedium',
    'PlasmaProperties',
    'PlasmaTarget',
    'Unit',
    'PhysicalQuantity',
    'Species',
    'PowerSupply',
    'DischargeRegime',
    'PlasmaSource',
]

# >>> Fill these with your own terms (examples included)
LTP_KEYWORDS = [
    "low temperature plasma", "cold plasma", "non-equilibrium plasma",
    "atmospheric pressure plasma", "nonthermal plasma",
    "dielectric barrier discharge", "dbd", "glow discharge",
    "rf plasma", "microwave plasma", "plasma jet", "plasma torch",
    "plasma medicine", "surface treatment", "plasma polymerization",
    "plasma etching", "pecvd", "langmuir probe", "optical emission spectroscopy",
    "reactive oxygen species", "ros", "reactive nitrogen species", "rns",
    "plasma catalysis", "plasma activated water"
]

EXCLUSIONS = [
    "tokamak", "fusion", "stellarator", "magnetic confinement",
    "mhd", "magnetized plasma", "divertor", "gyrotron",
    "burning plasma", "inertial confinement", "laser plasma"
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

def label_ltp(title: str, abstract: str, paper_keywords: Union[str, Iterable[str], None] = None):
    t = (title or "")
    a = (abstract or "")
    k = _norm_keywords(paper_keywords)
    text = f"{t} {a} {k}"

    pos_hits = sorted({term for term, pat in zip(LTP_KEYWORDS, _POS_PATS) if pat.search(text)})
    neg_hits = sorted({term for term, pat in zip(EXCLUSIONS, _NEG_PATS) if pat.search(text)})

    # simple decision rule: at least 1 positive AND no exclusions
    is_ltp = (len(pos_hits) > 0) and (len(neg_hits) == 0)

    return is_ltp
    
def best_label_per_token(results):
    tokens = results["Tokens"]
    # find all class names from keys like "Label_X"
    classes = [k.split("Label_", 1)[1] for k in results.keys() if k.startswith("Label_")]
    classes.sort()

    out = []
    for i, tok in enumerate(tokens):
        best = {"token": tok, "class": None, "tag": "O", "score": 0.0}
        for c in classes:
            tag = results[f"Label_{c}"][i]
            score = results[f"Score_{c}"][i]
            if tag != "O" and score >= best["score"]:
                best = {"token": tok, "class": c, "tag": tag, "score": float(score)}
        out.append(best)
    return out

def bio_spans(token_labels):
    """
    Collapse BIO tags into spans:
    token_labels: list of {"token","class","tag","score"}
    Returns spans with aggregated score (mean) and text.
    """
    spans = []
    cur = None

    for idx, t in enumerate(token_labels):
        tag, cls, tok, sc = t["tag"], t["class"], t["token"], t["score"]

        if tag.startswith("B-"):
            # start a new span
            if cur:
                # close previous
                cur["score"] = sum(cur["scores"]) / max(1, len(cur["scores"]))
                del cur["scores"]
                spans.append(cur)
            cur = {"class": cls, "start": idx, "end": idx + 1, "tokens": [tok], "scores": [sc]}
        elif tag.startswith("I-") and cur and cls == cur["class"]:
            # continue current span
            cur["end"] = idx + 1
            cur["tokens"].append(tok)
            cur["scores"].append(sc)
        else:
            # outside any entity
            if cur:
                cur["score"] = sum(cur["scores"]) / max(1, len(cur["scores"]))
                del cur["scores"]
                spans.append(cur)
                cur = None

    if cur:
        cur["score"] = sum(cur["scores"]) / max(1, len(cur["scores"]))
        del cur["scores"]
        spans.append(cur)

    # add text for convenience
    for sp in spans:
        sp["text"] = " ".join(sp["tokens"])

    return spans
    
def updated_item_metadata(item_uuid, payload, s):
    url = CONFIG.UPDATE_ITEMS_METADATA.format(item_uuid=item_uuid)
    upload_response = s.patch(url, json=payload)
    if upload_response.ok:
        print("Metadata updated successfully!")
    else:
        print("Metadata upload failed", upload_response.text)

def create_and_add_annotations(s, collection_id):
    items = get_collection_items_by_handle(collection_id, None)
    for it in items:
        item_uuid = it["uuid"]
        name = it["name"]
        xml_url = it["xml_content"]["content"]
        article = parse_jats_xml(xml_url)
        #print(article)
        
        title = article['title']
        abstract = article['abstract']
        sections = article['sections']
        print('title', title)
        print(abstract)
        keywords = []
        if len(title) !=0 and len(abstract)!=0:
            ltp = label_ltp(title, abstract[0], keywords)
            print(ltp)
        else:
            continue
        print('----------')
        ltp = True #remove it 
        threshold_entities = []
        if len(title) !=0 and len(abstract)!=0 and ltp:
            text = title+" "+abstract[0]
            results = annotate_class_wise_text(text)
            print(type(results))

            per_token = best_label_per_token(results)

            # Example: print token → (class, tag, score)
            #for t in per_token:
                #print(f"{t['token']:<15} {t['class'] or 'O':<22} {t['tag']:<18} {t['score']:.3f}")

                # Collapse BIO into entity spans
                
            entities = bio_spans(per_token)
            for e in entities:
                print(f"[{e['class']}] {e['text']}  (score={e['score']:.3f}, idx={e['start']}..{e['end']-1})")
                
                if e['score'] > 0.80 and e['class'] != 'Unit' :
                    threshold_entities.append(e['text'])

            threshold_entities = list(set(threshold_entities))
            print(threshold_entities)
        
            payload = [
                        {
                            "op": "add",
                            "path": "/metadata/dc.subject",
                            "value": {"value": entity}
                        }
                            for entity in threshold_entities
                    ]
            updated_item_metadata(item_uuid, payload, s)
            #for key, value in results.items():
                #if key.startswith("Label_"):
                    #label_name = value
                    #label_key = key.split("Label_")[1]
                    #score_key = f"Score_{label_key}"
                    #score = results.get(score_key)
                    #print(f"{label_name}: {score}")
                    
                    
            annotated_xml = build_from_bio_dict(results,
            meta={"doi": doi ,"title": title},
            run_meta={"annotator":"BERT-CRF"},
            out_path=name+".ppann.xml"
    
    
    #retrieve_high_quality_annotations(results)
    #upload_xml_to_renate(s, annotated_xml, bundle_uuid, name)
    
    #find high quality metadata
    

def nfc(s): return unicodedata.normalize("NFC", s)

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
            text_parts.append(" "); cur += 1

        start = cur
        text_parts.append(tok)
        cur += len(tok)
        end = cur
        spans.append((start, end))
        prev_token = tok

    text = "".join(text_parts)
    return nfc(text), spans  # normalize for stability

# ---------------- BIO → spans per type ----------------

def bio_to_spans(labels, token_spans, raw_text, entity_type, confidences=None, agg="min"):
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
            end   = token_spans[j-1][1]
            span_conf = (min(confidences[i:j]) if agg == "min"
                         else sum(confidences[i:j]) / max(1, j - i))
            out.append({
                "start": start,
                "end": end,
                "text": raw_text[start:end],
                "type": entity_type,
                "confidence": round(float(span_conf), 4)  # <— NEW
            })
            i = j
        else:
            i += 1
    return out

# ---------------- Build PPAnn XML ----------------
PP = "https://example.org/ppann/1.0"
XL = "http://www.w3.org/1999/xlink"

def build_ppann_xml(raw_text, spans_by_type):
    pp = "{%s}" % PP
    xl = "{%s}" % XL

    root = etree.Element(f"{pp}document", nsmap={"ppann": PP, "xlink": XL})
    root.set("version", "1.0")

    # source meta
  
    src = etree.SubElement(root, f"{pp}source")
    if "doi" in meta: etree.SubElement(src, f"{pp}doi").text = meta["doi"]
    if "title" in meta: etree.SubElement(src, f"{pp}title").text = meta["title"]

    # provenance
   
    prov = etree.SubElement(root, f"{pp}provenance")
    run = etree.SubElement(prov, f"{pp}run", id="r1", timestamp=datetime.utcnow().isoformat()+"Z")
    etree.SubElement(run, f"{pp}annotator", kind="model").text = run_meta.get("annotator","BERT-CRF (per-class)")
    ver = etree.SubElement(run, f"{pp}version")
    if "version" in run_meta: ver.set("code-commit", run_meta["version"])

    # section with the exact text you used to label
    secs = etree.SubElement(root, f"{pp}sections")
    sec = etree.SubElement(secs, f"{pp}section", id="sec-main", type="main", title="MainText")
    etree.SubElement(sec, f"{pp}text").text = raw_text

    # annotations
    ann = etree.SubElement(root, f"{pp}annotations", **{"run-ref":"r1"})
    ents = etree.SubElement(ann, f"{pp}entities")
    etree.SubElement(ann, f"{pp}entity-attributes")   # kept for future use
    etree.SubElement(ann, f"{pp}relations")           # kept for future use
    etree.SubElement(ann, f"{pp}section-labels")      # kept for future use
    etree.SubElement(ann, f"{pp}sentences")           # optional

    all_spans = []
    for etype, spans in spans_by_type.items():
        for sp in spans:
            sp = dict(sp)
            sp["type"] = etype
            all_spans.append(sp)

    all_spans.sort(key=lambda s: (int(s["start"]), int(s["end"])))
    # emit entities

    eid = 0
    for sp in all_spans:
        eid += 1
        e = etree.SubElement(ents, f"{pp}entity")
        e.set("id", f"e{eid}")
        e.set("type", sp["type"])
        e.set("section", sp.get("section", "sec-main"))
        e.set("start", str(sp["start"]))
        e.set("end", str(sp["end"]))
        e.set("text", sp["text"])
        if "confidence" in sp:
            e.set("confidence", f'{float(sp["confidence"]):.4f}')

    # vocab
    vocab = etree.SubElement(root, f"{pp}vocab")
    for etype in sorted(spans_by_type.keys()):
        etree.SubElement(vocab, f"{pp}entity-type", id=etype, iri=f"https://example.org/plasma#{etype}")

    xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", pretty_print=True)
    return xml_bytes

def build_from_bio_dict(bio_dict, out_path="annotations.ppann.xml"):
    tokens = bio_dict["Tokens"]
    raw_text, token_spans = detok_with_offsets(tokens)

    # NEW: collect confidences per entity type
    confidences_by_type = {
        k.replace("Score_", "", 1): v
        for k, v in bio_dict.items()
        if k.startswith("Score_")
    }

    spans_by_type = {}
    for key, labels in bio_dict.items():
        if not key.startswith("Label_"):
            continue
        etype = key.replace("Label_", "", 1)
        confs = confidences_by_type.get(etype)  # may be None
        spans = bio_to_spans(labels, token_spans, raw_text, etype, confs)  # <— pass confs
        spans_by_type[etype] = spans

    return build_ppann_xml(raw_text, spans_by_type)

#when physics papers will be added, someone will invoke the script. The script should load the physics files and generate their annotations.

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
        "subsections": children
    }

# --- Main parser -------------------------------------------------------------
def _t(el):
    return "" if el is None else "".join(el.itertext())

def parse_jats_xml(xml_url):
    r = requests.get(xml_url, stream=True)
    r.raise_for_status()
    r.raw.decode_content = True

    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True, huge_tree=True)
    tree = etree.parse(r.raw, parser)
    root = tree.getroot()

    # Detect default ns (JATS commonly uses a default)
    jats_ns = root.nsmap.get(None)  # default namespace URI if present
    ns = {
        "j": jats_ns or "http://jats.nlm.nih.gov",           # JATS
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
    captions_nodes = root.xpath(".//j:fig//j:caption | .//j:table-wrap//j:caption", namespaces=ns)
    captions = [" ".join(_t(c).split()) for c in captions_nodes if _t(c).strip()]

    # Acknowledgments
    ack_ps = root.findall(".//j:back/j:ack//j:p", namespaces=ns)
    acknowledgments = [" ".join(_t(p).split()) for p in ack_ps if _t(p).strip()]

    return {
        "title": title,
        "abstract": abstract,
        "sections": sections,
        "captions": captions,
        "acknowledgments": acknowledgments,
    }


def get_xml():
    #get xml from renate
    #open it
    read_jats_xml("test.xml")

def read_jats_xml(file_path):
    art = parse_jats_xml(file_path)
    print("TITLE:", art["title"])
    print('abstract', art['abstract'][0])
    #print('sections', art['sections'])
    annotate_class_wise_text(art['abstract'][0])

def annotate_class_wise_text(text):
    #load each class model
    #annotate text
    text = clean_and_split_text(text)
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt',
                   return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping").cpu().numpy().tolist()


    entity_labels = {f"Label_{entity}": [] for entity in entities}
    entity_scores = {f"Score_{entity}": [] for entity in entities}


    results = {"Tokens": []}

    for entity in entities:
        tag = entity
        print(f"🚀 Processing entity: {tag}")
        id2label = [
            'O',
            'B-'+tag,        'I-'+tag
        ]

        with torch.no_grad():
            model = BertCRF.from_pretrained(f'trained_models/{tag}', num_labels=3)
            outputs = model(**inputs)

            emissions = model.classifier(model.bert(**inputs)[0])           # (B, T, 3)
            probs = torch.softmax(emissions, dim=-1)                        # (B, T, 3)

            # probability of the CRF-decoded tag at each token (simple & aligned)
            probs_for_crf_tag = probs.gather(-1, outputs[1].unsqueeze(-1)).squeeze(-1)  # (B, T)

        data = decode(
            outputs[1].numpy().tolist(),
            inputs['input_ids'].numpy().tolist(),
            offset_mapping,
            id2label,
            probs_for_crf_tag.cpu().numpy().tolist()   # NEW: pass confidences
        )
        #print(data)
        #predicted_entities = predicted_entities[0]
        #print(data)


        if not results["Tokens"]: 
            for sentence_data in data:
                results["Tokens"].extend(sentence_data["words"])

        # Append corresponding labels
        for i, sentence_data in enumerate(data):
            entity_labels[f"Label_{tag}"].extend(sentence_data["labels"])
            entity_scores[f"Score_{tag}"].extend(sentence_data["confidences"])
    
        del model, outputs, data  # Delete large objects
        torch.cuda.empty_cache()  # Clear CUDA memory (if using GPU)
        gc.collect()

    results.update(entity_labels)
    results.update(entity_scores)
    return results


def retrieve_high_quality_annotations():
    print()

def enrich_paper_metadata():
    print()

def create_annotated_xml():
    print()

def decode(label_ids, input_ids, offset_mapping, id2label, token_confidences):
    result = []
    for k in range(len(label_ids)):  # batch
        words = []
        labels = []
        confs = []  # NEW

        for i in range(len(label_ids[k])):  # tokens
            start_ind, end_ind = offset_mapping[k][i]
            word = tokenizer.convert_ids_to_tokens([int(input_ids[k][i])])[0]
            is_subword = end_ind - start_ind != len(word)

            if is_subword:
                if word.startswith('##') and words:
                    # merge subword and aggregate confidence (min over pieces)
                    words[-1] += word[2:]
                    confs[-1] = min(confs[-1], float(token_confidences[k][i]))
                # else: ignore stray subword at start
            else:
                words.append(word)
                labels.append(id2label[int(label_ids[k][i])])
                confs.append(float(token_confidences[k][i]))  # NEW
        words[len(words)-1] = words[len(words)-1]+"."
        result.append({'words': words, 'labels': labels, 'confidences': confs})  # NEW
    return result

def clean_and_split_text(text):
    """Cleans text by removing unwanted characters and splits into sentences."""
    # Remove specific unwanted characters
    text = text.replace('\n', ' ')  # Replace newlines with space to avoid joining words

    # Split text into sentences by '.'
    sentences = text.split('.')
    
    # Strip whitespace from each sentence
    cleaned_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    
    return cleaned_sentences