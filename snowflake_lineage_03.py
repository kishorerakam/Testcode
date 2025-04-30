import os
import json
from collections import defaultdict, namedtuple
from typing import Optional
from sqlfluff.core import Linter
from sqlfluff.core.parser import BaseSegment

# ---------------- TUPLES ---------------- #
ColumnQualifierTuple = namedtuple("ColumnQualifierTuple", ["column", "parent"])
SubQueryTuple = namedtuple("SubQueryTuple", ["segment", "alias"])

# ---------------- UTILITY FUNCTIONS ---------------- #
def is_negligible(segment: BaseSegment) -> bool:
    return (
        segment.is_whitespace
        or segment.is_comment
        or bool(segment.is_meta)
        or (segment.type == "symbol" and segment.raw != "*")
    )

def is_subquery(segment: BaseSegment) -> bool:
    if segment.type == "from_expression_element" or segment.type == "bracketed":
        token = extract_innermost_bracketed(
            segment if segment.type == "bracketed" else segment.segments[0]
        )
        if token.get_child("select_statement", "set_expression", "with_compound_statement"):
            return True
        elif expression := token.get_child("expression"):
            if expression.get_child("select_statement"):
                return True
    return False

def find_table_identifier(segment: BaseSegment) -> Optional[BaseSegment]:
    if segment.type in ["table_reference", "file_reference", "object_reference"]:
        return segment
    for sub_segment in segment.segments:
        if identifier := find_table_identifier(sub_segment):
            return identifier
    return None

def list_subqueries(segment: BaseSegment) -> list[SubQueryTuple]:
    subquery = []
    if segment.type == "from_expression_element":
        as_segment, target = extract_as_and_target_segment(segment)
        if is_subquery(target):
            subquery = [SubQueryTuple(extract_innermost_bracketed(target),
                         extract_identifier(as_segment) if as_segment else None)]
    elif segment.type in ["from_clause", "from_expression"]:
        for seg in segment.segments:
            if seg.type == "from_expression_element":
                subquery += list_subqueries(seg)
            elif seg.type == "join_clause":
                for join_seg in seg.segments:
                    if join_seg.type == "from_expression_element":
                        subquery += list_subqueries(join_seg)
    return subquery

def extract_identifier(col_segment: BaseSegment) -> str:
    identifiers = list_child_segments(col_segment)
    col_identifier = identifiers[-1]
    return str(col_identifier.raw)

def extract_as_and_target_segment(segment: BaseSegment) -> tuple[Optional[BaseSegment], BaseSegment]:
    as_segment = segment.get_child("alias_expression")
    sublist = list_child_segments(segment, False)
    target = sublist[0]
    if target.type == "keyword" and target.raw_upper == "LATERAL":
        target = sublist[1]
    table_expr = target if is_subquery(target) else target.segments[0]
    return as_segment, table_expr

def list_child_segments(segment: BaseSegment, check_bracketed: bool = True) -> list[BaseSegment]:
    if segment.type == "bracketed" and check_bracketed:
        return [seg for seg in segment.segments if seg.type == "set_expression"]
    return [seg for seg in segment.segments if not is_negligible(seg)]

def extract_innermost_bracketed(bracketed_segment: BaseSegment) -> BaseSegment:
    while True:
        sub_bracketed_segments = [
            bs.get_child("bracketed")
            for bs in bracketed_segment.segments
            if bs.get_child("bracketed")
        ]
        sub_paren = bracketed_segment.get_child("bracketed") or (
            sub_bracketed_segments[0] if sub_bracketed_segments else None
        )
        if sub_paren:
            bracketed_segment = sub_paren
        else:
            break
    return bracketed_segment

def safe_get_table_reference(segment: BaseSegment) -> Optional[BaseSegment]:
    # Try direct child access first
    table_ref = segment.get_child("table_reference")
    if table_ref:
        return table_ref
    # Fallback: search immediate children
    for seg in segment.segments:
        if seg.type == "table_reference":
            return seg
    return None

def extract_table_sources_from_segment(segment: BaseSegment) -> set[str]:
    sources = set()
    for seg in segment.recursive_crawl("table_reference"):
        sources.add(seg.raw)
    return sources

# ---------------- TABLE EXTRACTOR FOR SEGMENT ---------------- #
def extract_tables_from_segment(segment: BaseSegment, ctes: set):
    sources = set()
    targets = set()
    #print({segment})
    print(segment.to_tuple(show_raw=True))
    if segment.type == "with_compound_statement":
        if cte_clause := segment.get_child("common_table_expression"):
            for identifier in cte_clause.recursive_crawl("identifier"):
                ctes.add(identifier.raw)
            # Look for embedded insert_statement inside with_compound_statement
            for seg in segment.segments:
                if seg.type == "insert_statement":
                    sources_inner, targets_inner = extract_tables_from_segment(seg, ctes)
                    sources |= sources_inner
                    targets |= targets_inner
    print(f"segment.type:{segment.type}")            
    if segment.type == "insert_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
    elif segment.type == "create_table_as_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
    elif segment.type == "merge_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw) 
            seen_using = False  
            for seg in segment.segments:
                if seg.raw_upper == "USING":
                    seen_using = True
                elif seen_using and seg.type == "table_reference":
                    sources.add(seg.raw)
    elif segment.type == "update_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
                    
    for elem in segment.recursive_crawl("from_expression_element"):
        if is_subquery(elem):
            subqueries = list_subqueries(elem)
            for sq in subqueries:
                if table := find_table_identifier(sq.segment):
                    sources.add(table.raw)
        else:
            if table := find_table_identifier(elem):
                sources.add(table.raw)

    return sources, targets

# ---------------- PROCESS SQL FILES ---------------- #
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
                #print(f"result,{result}")
                ctes = set()
                for segment in result.tree.recursive_crawl("statement"):
                    print("Segment type:", segment.type)
                    real_stmt = segment.segments[0] 
                    print("actual statement type:", real_stmt.type)
                    for s in real_stmt.recursive_crawl():
                             print(f" HHHH - {s.type}: {s.raw}")
                    sources, targets = extract_tables_from_segment(real_stmt, ctes)
                    print(f"  - Sources: {sources}")
                    print(f"  - Targets: {targets}")
                    for target in targets:
                        if target in ctes:
                            continue
                        for source in sources:
                            if source in ctes or source == target:
                                continue
                            if target not in lineage_graph[source]:
                                lineage_graph[source].append(target)

    return lineage_graph

# ---------------- SAVE TO JSON ---------------- #
def save_lineage_to_json(graph, output_file='lineage_output.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({k: sorted(v) for k, v in graph.items()}, f, indent=4)
    print(f"\u2705 Saved to {output_file}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    folder_paths = ["/Users/ryakamkishore/Lineage/20250429/SQL_path"]  # Update this path
    graph = process_sql_files(folder_paths)
    save_lineage_to_json(graph)
