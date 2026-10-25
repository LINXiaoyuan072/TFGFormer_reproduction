"""Read selected cells from the paper's XLSX without adding a spreadsheet dependency."""
import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

MAIN = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG_REL = 'http://schemas.openxmlformats.org/package/2006/relationships'


def cell_value(cell, shared):
    kind = cell.get('t')
    if kind == 'inlineStr':
        return ''.join(node.text or '' for node in cell.findall(f'.//{{{MAIN}}}t'))
    node = cell.find(f'{{{MAIN}}}v')
    if node is None:
        return None
    value = node.text
    if kind == 's':
        return shared[int(value)]
    if kind == 'b':
        return value == '1'
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def column_index(reference):
    letters = re.match(r'[A-Z]+', reference).group(0)
    result = 0
    for char in letters:
        result = result * 26 + ord(char) - 64
    return result - 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workbook', type=Path)
    parser.add_argument('--sheets', nargs='+')
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.workbook) as archive:
        shared_root = ET.fromstring(archive.read('xl/sharedStrings.xml'))
        shared = [''.join(t.text or '' for t in item.findall(f'.//{{{MAIN}}}t'))
                  for item in shared_root.findall(f'{{{MAIN}}}si')]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        rels = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        targets = {node.get('Id'): node.get('Target') for node in rels.findall(f'{{{PKG_REL}}}Relationship')}
        sheet_paths = {node.get('name'): 'xl/' + targets[node.get(f'{{{REL}}}id')]
                       for node in workbook.findall(f'.//{{{MAIN}}}sheet')}
        selected = args.sheets or list(sheet_paths)
        result = {}
        for name in selected:
            root = ET.fromstring(archive.read(sheet_paths[name]))
            rows = []
            for row in root.findall(f'.//{{{MAIN}}}row'):
                cells = row.findall(f'{{{MAIN}}}c')
                if not cells:
                    continue
                width = max(column_index(cell.get('r')) for cell in cells) + 1
                values = [None] * width
                for cell in cells:
                    values[column_index(cell.get('r'))] = cell_value(cell, shared)
                while values and values[-1] is None:
                    values.pop()
                rows.append(values)
                if args.max_rows is not None and len(rows) >= args.max_rows:
                    break
            result[name] = rows
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
