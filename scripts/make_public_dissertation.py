"""Create a privacy-edited reading edition of the supplied 39-page submission.

Requires the optional `report` dependency. Preserves research content and
bibliography; removes the candidate identifier, proposal and signed forms.
The original input is never overwritten.
"""
import argparse
from pathlib import Path
import re
import pymupdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("dissertation/dissertation-public.pdf"))
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("The public edition must not overwrite the submitted PDF.")
    original = pymupdf.open(args.source)
    if len(original) != 39:
        raise ValueError("Expected the supplied 39-page dissertation; review a different source manually.")
    doc = pymupdf.open()
    doc.insert_pdf(original, from_page=0, to_page=32)
    title = doc[0]
    candidate_line = next((line.strip() for line in title.get_text().splitlines()
                           if re.search(r"candidate\s+no\.", line, re.I)), None)
    if not candidate_line:
        raise ValueError("Candidate identifier not located; refusing an unverified public copy.")
    for rect in title.search_for(candidate_line):
        title.add_redact_annot(rect, fill=(1, 1, 1))
    title.apply_redactions()
    last = doc[-1]
    headings = last.search_for("Original Project Proposal")
    if len(headings) != 1:
        raise ValueError("Cannot locate the proposal boundary after the bibliography.")
    last.add_redact_annot(pymupdf.Rect(0, headings[0].y0 - 2, last.rect.width, last.rect.height), fill=(1, 1, 1))
    last.apply_redactions()
    for page in doc:
        for annotation in list(page.annots() or []):
            page.delete_annot(annotation)
        for link in page.get_links():
            page.delete_link(link)
    doc.set_metadata({})
    doc.del_xml_metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(args.output, garbage=4, deflate=True, clean=True)
    check = pymupdf.open(args.output)
    text = "\n".join(page.get_text() for page in check)
    assert len(check) == 33 and candidate_line not in text
    assert "By Nicholas Suciu" not in check[-1].get_text()
    assert "6.8 Final Remarks" in text
    assert all(not list(page.widgets() or []) for page in check)
    print("Public reading edition verified: 33 pages; research text and bibliography retained.")


if __name__ == "__main__":
    main()
