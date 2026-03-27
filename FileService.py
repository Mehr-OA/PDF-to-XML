# from dotenv import load_dotenv
import os
from lxml import etree
import xml.etree.ElementTree as ET
from lxml.builder import E
import re
import requests
import os.path

# load_dotenv()
import os

# import yaml
from urllib.parse import urljoin
import json
from datetime import datetime, timezone
from urllib.parse import quote

# Load config.yml once
from config_loader import CONFIG
from io import BytesIO
from tqdm import tqdm


directory = "paper-pdfs"
GROBID_API_URL = CONFIG.GROBID_API_URL

CURSOR_FILE = "last_run.json"


def load_last_run(collection_id):
    """Return last run timestamp for a collection, or None if not found."""
    if not os.path.exists(CURSOR_FILE):
        return None

    with open(CURSOR_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if collection_id in data:
        return data[collection_id].get("last_run")
    return None


def save_last_run(collection_id):
    """Save last run timestamp for a given collection."""
    # Load previous runs if the file exists
    file_data = {}
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE, "r", encoding="utf-8") as f:
            file_data = json.load(f)

    # Update this collection's timestamp
    file_data[collection_id] = {"last_run": datetime.now(timezone.utc).isoformat()}

    # Save all runs back to file
    with open(CURSOR_FILE, "w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2)

    print(
        f"[INFO] Saved last run for {collection_id}: {file_data[collection_id]['last_run']}"
    )


def pdf_to_xml(pdf_url, name, doi, renate_doi, license):
    pdf_file = download_item_content(pdf_url)
    pdf_file_content = {"input": ("file.pdf", BytesIO(pdf_file), "application/pdf")}
    response = requests.post(GROBID_API_URL, files=pdf_file_content)
    if response.status_code == 200:
        xml_tree = etree.fromstring(response.content)
        xml = convert_tei_to_jats(xml_tree, name, doi, renate_doi, license)
        return xml


def pdf_to_txt():
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            # print(file_path)
            output_file_path = os.path.join(
                "textfiles", os.path.splitext(filename)[0] + ".txt"
            )
            with open(file_path, "rb") as file, open(
                output_file_path, "w", encoding="utf-8"
            ) as output_file:
                # print(file)
                files = {"input": file}
                response = requests.post(GROBID_API_URL, files=files)
                if response.status_code == 200:
                    print(True)
                    xml_tree = etree.fromstring(response.content)
                    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
                    paper_content = extract_paper_content(xml_tree, ns)
                    paper_metadata = extract_paper_metadata(xml_tree, ns)

                    textcontent = ""
                    textcontent = textcontent + "Abstract\n"
                    textcontent = textcontent + paper_metadata["abstract"] + "\n\n"
                    for title, content in paper_content.items():
                        textcontent = textcontent + title + "\n"
                        textcontent = textcontent + content + "\n\n"
                    output_file.write(textcontent)

    return ""


def extract_paper_metadata(tree, namespaces):
    details = {}
    # Extracting paper title
    title_elem = tree.find('.//tei:title[@type="main"]', namespaces)
    details["title"] = title_elem.text if title_elem is not None else "Not available"

    # Extracting authors
    authors = []
    for author_elem in tree.findall(".//tei:teiHeader//tei:author", namespaces):
        author = {}
        pers_name = author_elem.find(".//tei:persName", namespaces)
        if pers_name is not None:
            author["name"] = (
                pers_name.findtext(
                    ".//tei:forename", namespaces=namespaces, default="Not available"
                )
                + " "
                + pers_name.findtext(
                    ".//tei:surname", namespaces=namespaces, default="Not available"
                )
            )
        author["orcid"] = author_elem.findtext(
            './/tei:idno[@type="orcid"]', namespaces=namespaces, default="Not available"
        )
        author["affiliation"] = author_elem.findtext(
            ".//tei:affiliation/tei:orgName",
            namespaces=namespaces,
            default="Not available",
        )
        authors.append(author)
    details["authors"] = authors

    abstract_elem = tree.find(".//tei:abstract", namespaces)
    abstract_text = (
        etree.tostring(abstract_elem, method="text", encoding="unicode").strip()
        if abstract_elem is not None
        else "Not available"
    )
    details["abstract"] = abstract_text
    # Extracting DOI
    details["doi"] = tree.findtext(
        './/tei:idno[@type="DOI"]', namespaces=namespaces, default="Not available"
    )

    # Extracting date published
    publication_date = tree.findtext(
        './/tei:date[@type="published"]', namespaces=namespaces, default="Not available"
    )
    # Removing any non-date characters
    details["date_published"] = re.sub("[^0-9-]", "", publication_date)
    return details


def extract_paper_content(tree, ns):
    sections = {}
    # Iterate over each head tag in the body
    for head in tree.xpath("//tei:body/tei:div/tei:head", namespaces=ns):
        heading = head.text.strip() if head.text else "Unnamed Section"
        content = []

        # Find all the subsequent p tags until the next head tag
        for sibling in head.itersiblings():
            if sibling.tag == "{http://www.tei-c.org/ns/1.0}p":
                paragraph = sibling.xpath("string(.)", namespaces=ns).strip()
                content.append(paragraph)
            elif sibling.tag == "{http://www.tei-c.org/ns/1.0}head":
                break

        sections[heading] = "\n".join(content)

    return sections


def convert_tei_to_jats(tei_xml, name, doi, renate_doi, license):
    with open("teixml.xml", "rb") as file:
        xslt = etree.XSLT(etree.XML(file.read()))

    print(name, doi, renate_doi, license)
    try:
        jats_xml = xslt(
            tei_xml,
            title=etree.XSLT.strparam(name),
            dc_identifier_uri=etree.XSLT.strparam(doi),
            dc_identifier_renate=etree.XSLT.strparam(renate_doi),
            dc_rights_license=etree.XSLT.strparam(license),
            xml_created=etree.XSLT.strparam(datetime.today().strftime("%d-%m-%Y"))
        )
        return etree.tostring(jats_xml, pretty_print=True)

    except etree.XSLTApplyError as e:
        print("XSLTApplyError:", e)
        print("---- XSLT error log ----")
        print(xslt.error_log)
        raise


def get_items_for_collection(collection_id, since=None):
    print(f"Retrieving items from a collection {collection_id}")

    items = []
    page = 0
    page_size = 100

    base_url = f"{CONFIG.COLLECTION_ITEMS_ENDPOINT}{collection_id}&size={page_size}&page={page}"
    next_url = base_url

    pbar = tqdm(desc="Retrieving items", unit=" item")

    while next_url:
        response = requests.get(next_url)
        response.raise_for_status()
        data = response.json()

        search_result = data.get("_embedded", {}).get("searchResult", {})
        collection_items = search_result.get("_embedded", {}).get("objects", [])

        if not collection_items:
            break

        for item in collection_items:
            item_obj = (item.get("_embedded") or {}).get("indexableObject") or {}
            if not item_obj:
                continue

            if since:
                lm = item_obj.get("lastModified")
                if not lm:
                    continue
                if lm < since:
                    continue

            items.append(item_obj)

            pbar.update(1)  # ← progress increases

        next_url = search_result.get("_links", {}).get("next", {}).get("href")

    pbar.close()

    print(f"Retrieved {len(items)} items")
    return items


def pick_pdf_and_xml(bitstreams_json):
    bits = (bitstreams_json.get("_embedded") or {}).get("bitstreams") or []

    def content_of(b):
        return (b.get("_links") or {}).get("content", {}).get("href")

    pdf = next(
        (
            b
            for b in bits
            if ((b.get("format") or {}).get("mimeType") == "application/pdf")
            or (b.get("name", "").lower().endswith(".pdf"))
        ),
        None,
    )

    xmls = [b for b in bits if b.get("name", "").lower().endswith(".xml")]

    def xml_priority(name: str) -> int:
        n = name.lower()
        if n.endswith(".xml"):
            return 2
        return 1

    xml = max(
        xmls,
        key=lambda b: (xml_priority(b.get("name", "")), b.get("sequenceId", 0)),
        default=None,
    )

    def create_object(b):
        if not b:
            return None
        return {"uuid": b.get("uuid"), "name": b.get("name"), "content": content_of(b)}

    return {"pdf": create_object(pdf), "xml": create_object(xml)}


def get_item_information(item_id):
    response = requests.get(f"{CONFIG.ITEMS_ENDPOINT}/{item_id}")
    response.raise_for_status()
    item_info = response.json()
    return item_info


def get_item_content(bundle_links):
    item_content = requests.get(bundle_links).json()
    bitstream_url = ""
    bundle_uuid = ""
    # print(item_content['_embedded']['bundles'])
    for item in item_content["_embedded"]["bundles"]:
        if item["name"] == "ORIGINAL":
            bundle_uuid = item["uuid"]
            bitstream_url = item["_links"]["bitstreams"]["href"]
            # break

    item_content = requests.get(bitstream_url).json()
    r = pick_pdf_and_xml(item_content)
    item_name = item_content["_embedded"]["bitstreams"][0]["name"]
    return {"name": item_name, "bundle_uuid": bundle_uuid, "content": r}
    # content_url = item_content['_embedded']['bitstreams'][0]['_links']['content']['href']
    # return {'name': item_name, 'bundle_uuid': bundle_uuid, 'content': content_url}


def download_item_content(pdf_url):
    pdf_response = requests.get(pdf_url, stream=True)
    pdf_response.raise_for_status()
    return pdf_response.content


def upload_xml_to_renate(s, xml, bundle_uuid, name):
    files = {"file": (f"{name}.xml", BytesIO(xml), "application/json")}

    url = CONFIG.UPLOAD_BITSTREAMS_ENDPOINT.format(bundle_uuid=bundle_uuid)
    upload_response = s.post(url, files=files)
    if upload_response.ok:
        print("File uploaded successfully!")
    else:
        print("File upload failed", upload_response.text)


def get_collection_items_by_handle(collection_id: str, since: None):
    items = get_items_for_collection(collection_id, since)
    # remove this code
    item_s = []
    for item in items:
        if item["uuid"] == "d81476a5-146e-4edd-97f0-124edc83a9ac":
            item_s = item
    print('item')
    print(item_s)
    items = [item_s]
    # remove till here
    results = []
    print("Retreiving papers metadata...")

    for item in tqdm(items, desc="PDFs to XML"):
        metadata = item["metadata"]
        subjects = [x["value"] for x in metadata.get("dc.subject", [])]
        doi = [x["value"] for x in metadata.get("dc.relation.doi", [])]
        renate_doi = [
            x["value"]
            for x in metadata.get("dc.identifier.uri", [])
            if "doi.org/10." in x["value"]
        ]

        license = metadata.get("dc.rights.license", [])
        bundles_url = item["_links"]["bundles"]["href"]

        item_content = get_item_content(bundles_url)
        pdf = item_content.get("content", {}).get("pdf")

        if not pdf:
            continue

        # print(info)
        results.append(
            {
                "uuid": item["uuid"],
                "title": item["name"],
                "name":  item_content["name"],
                "doi": doi[0] if len(doi) > 0 else "",
                "license": license[0]["value"] if len(license) > 0 else "",
                "renate_doi": renate_doi[0] if len(renate_doi) > 0 else "",
                "keywords": subjects,
                "bundles_url": bundles_url,
                "pdf_content": item_content["content"]["pdf"],
                "xml_content": item_content["content"]["xml"],
                "bundle_uuid": item_content["bundle_uuid"],
            }
        )

    print(f"[INFO] Collected {len(results)} candidate items (metadata only).")
    return results


def add_xmls_in_renate(s: requests.Session, collection_id: str, page_size: int = 100):
    last_run = load_last_run(collection_id)
    if last_run:
        print(f"Last run for {collection_id}: {last_run}")
    else:
        print(f"No previous run found for {collection_id}")

    items = get_collection_items_by_handle(collection_id, since=last_run)

    processed = 0
    for it in tqdm(items[:1], desc="PDFs to XML"):
        item_uuid = it["uuid"]
        name = it["name"]
        name = name.split(".")[0] + "-jats"
        pdf_url = it["pdf_content"]["content"]

        xml = pdf_to_xml(pdf_url, it["title"], it["doi"], it["renate_doi"], it["license"])

        upload_xml_to_renate(s, xml, it["bundle_uuid"], name)
        processed += 1
        print(f"[OK] {item_uuid} → XML uploaded")

    save_last_run(collection_id)
    print(f"[INFO] Done. Uploaded XML for {processed}/{len(items)} items.")
