from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.ontology import OntologyStore, ontology_files


class SplitOntologyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        pages = self.root / "pages"
        pages.mkdir()
        (self.root / "profile.ttl").write_text(
            "<urn:test:profile> <urn:test:name> \"profile\" .\n",
            encoding="utf-8",
        )
        for number in (1, 2):
            (pages / f"page-{number:03d}.ttl").write_text(
                f"<urn:test:page-{number:03d}> <urn:test:name> \"page\" .\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_profile_and_pages_are_loaded_in_order(self) -> None:
        files = ontology_files(self.root)
        store = OntologyStore(self.root)

        self.assertEqual(
            [path.name for path in files],
            ["profile.ttl", "page-001.ttl", "page-002.ttl"],
        )
        self.assertEqual(store.files, files)
        self.assertEqual(store.triple_count, 3)

    def test_page_numbers_must_be_contiguous(self) -> None:
        (self.root / "pages" / "page-002.ttl").rename(
            self.root / "pages" / "page-003.ttl"
        )

        with self.assertRaisesRegex(ValueError, "contiguous"):
            ontology_files(self.root)

    def test_unexpected_turtle_is_rejected(self) -> None:
        (self.root / "published.ttl").write_text(
            "<urn:test:old> <urn:test:name> \"old\" .\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "Unexpected Turtle"):
            ontology_files(self.root)


if __name__ == "__main__":
    unittest.main()
