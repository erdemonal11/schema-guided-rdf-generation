import json
import html
import sys
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET

DBLP_XML_PATH = Path("data/real/dblp.xml")
OUTPUT_DIR = Path("data/processed")
TRIPLES_FILE = OUTPUT_DIR / "kg_triples_ids.txt"
MAPPINGS_FILE = OUTPUT_DIR / "kg_mappings.json"

MAX_PUBLICATIONS = 500_000

PUBLICATION_TAGS = {
    "article", "inproceedings", "incollection", "proceedings",
    "book", "phdthesis", "mastersthesis", "informal"
}

def _local(tag: str) -> str:
    return tag.split("}")[1] if "}" in tag else tag

def _text(elem) -> str:
    return (elem.text or "").strip() if elem is not None else ""

def main():
    if not DBLP_XML_PATH.exists():
        raise FileNotFoundError(f"Missing: {DBLP_XML_PATH}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    str_to_id: Dict[str, str] = {}
    id_to_str: Dict[str, str] = {}
    
    counts = {"author": 0, "pub": 0, "venue": 0, "year": 0}

    def get_id(text: str, type_prefix: str) -> str:
        key = f"{type_prefix}:{text}"
        if key not in str_to_id:
            new_id = f"{type_prefix}_{counts[type_prefix]}"
            str_to_id[key] = new_id
            id_to_str[new_id] = text
            counts[type_prefix] += 1
            return new_id
        return str_to_id[key]

    pub_count = 0
    total_scanned = 0

    print(f"[INFO] Parsing {DBLP_XML_PATH}...")
    
    parser = ET.XMLParser()
    for name, value in html.entities.entitydefs.items():
        parser.entity[name] = value

    context = ET.iterparse(str(DBLP_XML_PATH), events=("start", "end"), parser=parser)
    _, root = next(context)

    with open(TRIPLES_FILE, "w", encoding="utf-8") as f_out:
        
        for event, elem in context:
            if event != "end": continue
            
            total_scanned += 1
            if total_scanned % 50000 == 0:
                sys.stdout.write(f"\r[STATUS] Scanned: {total_scanned:,} | Publications: {pub_count:,}")
                sys.stdout.flush()

            tag = _local(elem.tag)
            if tag not in PUBLICATION_TAGS:
                continue

            try:
                title_text = _text(elem.find("title"))
                year_text = _text(elem.find("year"))
                venue_text = _text(elem.find("journal")) or _text(elem.find("booktitle"))

                if not title_text or not year_text:
                    elem.clear(); root.clear(); continue

                pub_id = get_id(title_text, "pub")
                year_id = get_id(year_text, "year")
                venue_id = get_id(venue_text, "venue") if venue_text else None

                author_elems = elem.findall("author") or elem.findall("editor")
                author_ids = []
                for a in author_elems:
                    aname = _text(a)
                    if aname:
                        author_ids.append(get_id(aname, "author"))
                
                if not author_ids:
                    author_ids.append(get_id("Unknown Author", "author"))

                batch = []
                for aid in author_ids:
                    batch.append(f"{aid}\tdblp:wrote\t{pub_id}\n")
                    batch.append(f"{pub_id}\tdblp:hasAuthor\t{aid}\n")
                
                if venue_id:
                    batch.append(f"{pub_id}\tdblp:publishedIn\t{venue_id}\n")
                batch.append(f"{pub_id}\tdblp:inYear\t{year_id}\n")
                
                f_out.writelines(batch)
                
                pub_count += 1
                elem.clear(); root.clear()

                if pub_count >= MAX_PUBLICATIONS:
                    break

            except Exception:
                elem.clear(); root.clear(); continue

    print(f"\n[INFO] Saving Mappings to {MAPPINGS_FILE}...")
    with open(MAPPINGS_FILE, "w", encoding="utf-8") as f_map:
        json.dump(id_to_str, f_map, indent=2)

    print(f"[SUCCESS] Done.")
    print(f" - Triples (GAN Input): {TRIPLES_FILE}")
    print(f" - Decoder Map (Website): {MAPPINGS_FILE}")

if __name__ == "__main__":
    main()