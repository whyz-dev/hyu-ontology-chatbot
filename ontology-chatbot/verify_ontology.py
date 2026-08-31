"""Validate the canonical profile plus page ontology package."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDF

HYU = Namespace("urn:hyu-chatbot:vocab:")
RESOURCE = Namespace("urn:hyu-chatbot:resource:")
PAGE_FILE = re.compile(r"page-(\d{3})\.ttl")


def verify(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    profile = root / "profile.ttl"
    pages = sorted((root / "pages").glob("page-*.ttl"))
    if not profile.is_file() or len(pages) != 79:
        raise ValueError(
            f"Expected profile.ttl and 79 page TTLs under {root}, got {len(pages)} pages"
        )

    full = Graph()
    component_total = 0
    page_document_iris: set[URIRef] = set()
    for path in (profile, *pages):
        component = Graph().parse(path, format="turtle")
        component_total += len(component)
        for triple in component:
            full.add(triple)
        if path == profile:
            continue
        match = PAGE_FILE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid page component filename: {path.name}")
        page_iri = URIRef(RESOURCE[f"page-{match.group(1)}"])
        if (page_iri, RDF.type, HYU.DocumentPage) not in component:
            raise ValueError(f"Missing DocumentPage in {path.name}: {page_iri}")
        page_document_iris.add(page_iri)
        for subject in set(component.subjects(HYU.sourcePage, None)):
            source_pages = set(component.objects(subject, HYU.sourcePage))
            if source_pages != {page_iri}:
                raise ValueError(
                    f"Page-owned subject has mismatched sourcePage in {path.name}: {subject}"
                )

    if component_total != len(full):
        raise ValueError(
            f"Duplicate triples across components: total={component_total}, union={len(full)}"
        )
    all_document_iris = set(full.subjects(RDF.type, HYU.DocumentPage))
    if all_document_iris != page_document_iris:
        raise ValueError(
            "DocumentPage set does not match the 79 page filenames: "
            f"graph={len(all_document_iris)}, files={len(page_document_iris)}"
        )

    declared_properties = {
        subject
        for kind in (OWL.ObjectProperty, OWL.DatatypeProperty)
        for subject in full.subjects(RDF.type, kind)
    }
    undefined_predicates = sorted(
        {
            predicate
            for predicate in full.predicates()
            if str(predicate).startswith(str(HYU))
            and predicate not in declared_properties
        },
        key=str,
    )
    if undefined_predicates:
        raise ValueError(
            "Undefined local predicates: " + ", ".join(map(str, undefined_predicates))
        )

    declared_classes = set(full.subjects(RDF.type, OWL.Class))
    undefined_classes = sorted(
        {
            value
            for value in full.objects(None, RDF.type)
            if str(value).startswith(str(HYU)) and value not in declared_classes
        },
        key=str,
    )
    if undefined_classes:
        raise ValueError(
            "Undefined local classes: " + ", ".join(map(str, undefined_classes))
        )

    local_subjects = {
        subject
        for subject in full.subjects()
        if str(subject).startswith(str(RESOURCE))
    }
    dangling = sorted(
        {
            value
            for value in full.objects()
            if str(value).startswith(str(RESOURCE)) and value not in local_subjects
        },
        key=str,
    )
    if dangling:
        raise ValueError("Dangling local resources: " + ", ".join(map(str, dangling)))

    return {
        "root": str(root),
        "components": 80,
        "profile_triples": len(Graph().parse(profile, format="turtle")),
        "page_triples": component_total
        - len(Graph().parse(profile, format="turtle")),
        "triples": len(full),
        "document_pages": len(page_document_iris),
        "duplicate_triples": 0,
        "undefined_local_predicates": 0,
        "undefined_local_classes": 0,
        "dangling_local_resources": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
