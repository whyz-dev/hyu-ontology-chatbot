from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable


BLOCK_TAGS = {"div", "p", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


@dataclass
class Node:
    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)
    parent: Node | None = None

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def descendants(self, tag: str | None = None) -> Iterable[Node]:
        for child in self.children:
            if isinstance(child, Node):
                if tag is None or child.tag == tag:
                    yield child
                yield from child.descendants(tag)

    def first_with_class(self, class_name: str) -> Node | None:
        if class_name in self.classes():
            return self
        return next(
            (node for node in self.descendants() if class_name in node.classes()),
            None,
        )

    def text(self, preserve_blocks: bool = False) -> str:
        parts: list[str] = []

        def visit(node: Node) -> None:
            for child in node.children:
                if isinstance(child, str):
                    parts.append(child)
                else:
                    if preserve_blocks and child.tag in BLOCK_TAGS:
                        parts.append("\n")
                    visit(child)
                    if preserve_blocks and child.tag in BLOCK_TAGS:
                        parts.append("\n")

        visit(self)
        joined = "".join(parts).replace("\xa0", " ")
        if preserve_blocks:
            lines = [re.sub(r"\s+", " ", line).strip() for line in joined.splitlines()]
            return "\n".join(line for line in lines if line)
        return re.sub(r"\s+", " ", joined).strip()


class DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, {key: value or "" for key, value in attrs}, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def _nearest_ancestor(node: Node, tag: str) -> Node | None:
    parent = node.parent
    while parent is not None:
        if parent.tag == tag:
            return parent
        parent = parent.parent
    return None


def _table_grid(table: Node) -> list[list[str]]:
    rows = [
        row
        for row in table.descendants("tr")
        if _nearest_ancestor(row, "table") is table
    ]
    grid: list[list[str]] = []
    carried: dict[tuple[int, int], str] = {}
    width = 0

    for row_index, row in enumerate(rows):
        values: list[str | None] = []
        for (future_row, column), value in carried.items():
            if future_row == row_index:
                while len(values) <= column:
                    values.append(None)
                values[column] = value

        cells = [
            cell
            for cell in row.descendants()
            if cell.tag in {"td", "th"} and _nearest_ancestor(cell, "tr") is row
        ]
        column = 0
        for cell in cells:
            while column < len(values) and values[column] is not None:
                column += 1
            rowspan = max(1, int(cell.attrs.get("rowspan", "1") or "1"))
            colspan = max(1, int(cell.attrs.get("colspan", "1") or "1"))
            value = cell.text()
            for offset in range(colspan):
                target = column + offset
                while len(values) <= target:
                    values.append(None)
                values[target] = value
                for future_row in range(row_index + 1, row_index + rowspan):
                    carried[(future_row, target)] = value
            column += colspan

        width = max(width, len(values))
        grid.append([value or "" for value in values])

    return [row + [""] * (width - len(row)) for row in grid]


def parse_notice(path: Path, source_url: str, source_hash: str) -> dict[str, object]:
    parser = DocumentParser()
    parser.feed(path.read_text(encoding="utf-8"))
    view = parser.root.first_with_class("board-default-view")
    if view is None:
        raise ValueError("Notice DOM does not contain .board-default-view")

    subject = view.first_with_class("subject")
    datetime_node = view.first_with_class("datetime")
    content = view.first_with_class("content")
    if subject is None or datetime_node is None or content is None:
        raise ValueError("Notice DOM schema changed: subject/datetime/content missing")

    title_node = next(subject.descendants("strong"), None)
    tables = []
    for index, table in enumerate(content.descendants("table"), start=1):
        if _nearest_ancestor(table, "table") is not None:
            continue
        grid = _table_grid(table)
        tables.append(
            {
                "table_id": f"notice-2341-table-{index:02d}",
                "grid": grid,
                "row_count": len(grid),
                "column_count": max((len(row) for row in grid), default=0),
                "source": {
                    "url": source_url,
                    "locator": f"table[{index}]",
                    "sha256": source_hash,
                },
                "verification": {"status": "pending"},
            }
        )

    return {
        "document_id": "policy-notice-2341",
        "title": title_node.text() if title_node is not None else subject.text(),
        "published_at": datetime_node.text(),
        "body": content.text(preserve_blocks=True),
        "tables": tables,
        "source": {"url": source_url, "sha256": source_hash},
        "verification": {"status": "pending"},
    }
