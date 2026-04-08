# from dotenv import load_dotenv
import os
from lxml import etree
import xml.etree.ElementTree as ET
from lxml.builder import E
import re
import requests
import os.path
from datetime import datetime
from io import BytesIO
from tqdm import tqdm

from config_loader import CONFIG

GROBID_API_URL = CONFIG.GROBID_API_URL


def pdf_to_xml(pdf_url, name, doi, renate_doi, license):
    pdf_file = download_item_content(pdf_url)
    pdf_file_content = {"input": ("file.pdf", BytesIO(pdf_file), "application/pdf")}
    response = requests.post(GROBID_API_URL, files=pdf_file_content)
    if response.status_code == 200:
        xml_tree = etree.fromstring(response.content)
        xml = convert_tei_to_jats(xml_tree, name, doi, renate_doi, license)
        return xml


def convert_tei_to_jats(tei_xml, name, doi, renate_doi, license):
    with open("teixml.xml", "rb") as file:
        xslt = etree.XSLT(etree.XML(file.read()))

    try:
        jats_xml = xslt(
            tei_xml,
            title=etree.XSLT.strparam(name),
            dc_identifier_uri=etree.XSLT.strparam(doi),
            dc_identifier_renate=etree.XSLT.strparam(renate_doi),
            dc_rights_license=etree.XSLT.strparam(license),
            xml_created=etree.XSLT.strparam(datetime.today().strftime("%d-%m-%Y")),
        )
        return etree.tostring(jats_xml, pretty_print=True)

    except etree.XSLTApplyError as e:
        print("XSLTApplyError:", e)
        print("---- XSLT error log ----")
        print(xslt.error_log)
        raise


def get_items_for_collection(collection_id):
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

            items.append(item_obj)
            pbar.update(1)

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

    jats_xml = next(
        (b for b in bits if b.get("name", "").lower().endswith("-jats.xml")),
        None,
    )

    annotation_xml = next(
        (b for b in bits if b.get("name", "").lower().endswith("-annotations.xml")),
        None,
    )

    def create_object(b):
        if not b:
            return None
        return {
            "uuid": b.get("uuid"),
            "name": b.get("name"),
            "content": content_of(b),
        }

    return {
        "pdf": create_object(pdf),
        "jats_xml": create_object(jats_xml),
        "annotation_xml": create_object(annotation_xml),
    }


def get_item_information(item_id):
    response = requests.get(f"{CONFIG.ITEMS_ENDPOINT}/{item_id}")
    response.raise_for_status()
    item_info = response.json()
    return item_info


def get_item_content(bundle_links):
    item_content = requests.get(bundle_links).json()
    bitstream_url = ""
    bundle_uuid = ""

    for item in item_content["_embedded"]["bundles"]:
        if item["name"] == "ORIGINAL":
            bundle_uuid = item["uuid"]
            bitstream_url = item["_links"]["bitstreams"]["href"]

    item_content = requests.get(bitstream_url).json()
    files = pick_pdf_and_xml(item_content)
    item_name = item_content["_embedded"]["bitstreams"][0]["name"]
    return {"name": item_name, "bundle_uuid": bundle_uuid, "content": files}


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


def get_collection_items_by_handle(collection_id: str):
    items = get_items_for_collection(collection_id)

    # remove this code
    item_s = []
    for item in items:
        if item["uuid"] == "d81476a5-146e-4edd-97f0-124edc83a9ac":
            item_s = item
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

        results.append(
            {
                "uuid": item["uuid"],
                "title": item["name"],
                "name": item_content["name"],
                "doi": doi[0] if len(doi) > 0 else "",
                "license": license[0]["value"] if len(license) > 0 else "",
                "renate_doi": renate_doi[0] if len(renate_doi) > 0 else "",
                "keywords": subjects,
                "bundles_url": bundles_url,
                "pdf_content": item_content["content"]["pdf"],
                "jats_xml_content": item_content["content"]["jats_xml"],
                "annotation_xml_content": item_content["content"]["annotation_xml"],
                "bundle_uuid": item_content["bundle_uuid"],
            }
        )

    print(f"[INFO] Collected {len(results)} candidate items (metadata only).")
    return results


def add_xmls_in_renate(s: requests.Session, collection_id: str):
    items = get_collection_items_by_handle(collection_id)

    processed = 0
    for it in tqdm(items[:1], desc="PDFs to XML"):
        item_uuid = it["uuid"]
        name = it["name"]
        name = name.split(".")[0] + "-jats"
        pdf_url = it["pdf_content"]["content"]
        existing_jats = it["jats_xml_content"]
        if existing_jats:
            print(f"[SKIP] {item_uuid} → JATS already exists")
            continue

        xml = pdf_to_xml(
            pdf_url, it["title"], it["doi"], it["renate_doi"], it["license"]
        )

        upload_xml_to_renate(s, xml, it["bundle_uuid"], name)
        processed += 1
        print(f"[OK] {item_uuid} → XML uploaded")

    print(f"[INFO] Done. Uploaded XML for {processed}/{len(items)} items.")
