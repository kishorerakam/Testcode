import os
import json
from collections import defaultdict
from sqlfluff.core import Linter

def clean_table_name(name: str) -> str:
    name = name.strip()
    for char in ('"', "'", '`'):
        name = name.replace(char, '')
    return name.upper()

def extract_tables_from_segment(segment, ctes):
    """
    Recursively extract source/target tables from SQLFluff AST segment,
    skipping CTEs properly from both sources and targets.
    """
    sources = set()
    targets = set()

    if hasattr(segment, 'type') and segment.type in (
        'with_compound_statement', 'insert_statement', 'update_statement', 'merge_statement', 'select_statement', 'create_table_statement'
    ):
        with_clause = segment.get_child('with_clause')
        print(f"segment.type:{segment.type},with_clause,{with_clause}")
        if segment.type == 'with_compound_statement':
            for cte_def in segment.recursive_crawl('common_table_expression'):
                identifier = cte_def.get_child('identifier')
                if identifier:
                    ctes.add(clean_table_name(identifier.raw))
        else:
            with_clause = segment.get_child('with_clause')
            if with_clause:
                for cte_def in with_clause.recursive_crawl('common_table_expression'):
                    identifier = cte_def.get_child('identifier')
                    if identifier:
                        ctes.add(clean_table_name(identifier.raw))

        select_part = (
            segment.get_child('select_statement') or
            segment.get_child('set_expression') or
            segment.get_child('from_clause')
        )
        if select_part:
            for child in select_part.recursive_crawl('table_reference'):
                identifiers = child.recursive_crawl('identifier')
                if identifiers:
                    full_name = ".".join([id.raw for id in identifiers])
                    clean_name = clean_table_name(full_name)
                    if clean_name and clean_name not in ctes:
                        sources.add(clean_name)

        if segment.type in ('insert_statement', 'update_statement', 'merge_statement', 'create_table_statement'):
            possible_targets = segment.recursive_crawl('table_reference')
            for table_ref in possible_targets:
                identifiers = table_ref.recursive_crawl('identifier')
                if identifiers:
                    full_name = ".".join([id.raw for id in identifiers])
                    clean_name = clean_table_name(full_name)
                    if clean_name and clean_name not in ctes:
                        targets.add(clean_name)

    for child in segment.segments if hasattr(segment, 'segments') else []:
        sub_sources, sub_targets = extract_tables_from_segment(child, ctes)
        sources.update(sub_sources)
        targets.update(sub_targets)

    return sources, targets

def process_sql_files(folder_paths):
    linter = Linter(dialect="snowflake")
    lineage_graph = defaultdict(list)

    for folder_path in folder_paths:
        print(f"\U0001F4C2 Processing: {folder_path}")
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".sql"):
                file_path = os.path.join(folder_path, file)
                print(f"\U0001F50E {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql = f.read()

                result = linter.parse_string(sql)
                if result.tree is None:
                    continue

                ctes = set()
                for segment in result.tree.recursive_crawl('statement'):
                    sources, targets = extract_tables_from_segment(segment, ctes)
                    print(f"sources:{sources}")
                    print(f"targets:{targets}")
                    for target in targets:
                        if target in ctes:
                            continue
                        for source in sources:
                            if source in ctes or source == target:
                                continue
                            if target not in lineage_graph[source]:
                                lineage_graph[source].append(target)

    return lineage_graph

def save_lineage_to_json(graph, output_file='lineage_output.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({k: sorted(v) for k, v in graph.items()}, f, indent=4)
    print(f"\u2705 Saved to {output_file}")

if __name__ == "__main__":
    folder_paths = ["/Users/ryakamkishore/Lineage/20250429/SQL_path"]
    graph = process_sql_files(folder_paths)
    save_lineage_to_json(graph)
