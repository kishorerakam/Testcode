import os
import json
from collections import defaultdict, namedtuple, OrderedDict
from typing import Optional
from sqlfluff.core import Linter
from sqlfluff.core.parser import BaseSegment

DEBUG = False

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
    if DEBUG:
        print(f"Inside function :safe_get_table_reference before :table_ref : {table_ref}")
    if table_ref:
        return table_ref
    # Fallback: search immediate children
    print(f"Inside function :safe_get_table_reference :table_ref after IF: {table_ref}")
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
    from collections import defaultdict
    sources = defaultdict(set)
    targets = set()
    if DEBUG:
        print({segment})
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
    if DEBUG:
        print(f"segment.type:{segment.type}")            
    if segment.type == "insert_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
    elif segment.type == "create_table_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
    elif segment.type == "merge_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw) 
        if DEBUG:
            print("Target Table:", tgt.raw)
            print("Line:", tgt.pos_marker.line_no)
            print("Char Pos:", tgt.pos_marker.line_pos)
            #seen_using = False  
            #for seg in segment.segments:
            #    if seg.raw_upper == "USING":
            #        seen_using = True
            #    elif seen_using and seg.type == "table_reference":
            #        sources.add(seg.raw)
        # Look for USING clause's source table
        for seg in segment.recursive_crawl("table_reference"):
            if tgt is None or seg.raw != tgt.raw:
                sources[seg.raw].add("USING")
        
    elif segment.type == "update_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
                    
    for join_clause in segment.recursive_crawl("join_clause"):
        if DEBUG:
            print("join_clause",join_clause.to_tuple(show_raw=True))
        # More robust join type extraction
        join_type = "INNER"  # Default
        for seg in join_clause.segments:
            if seg.type == "keyword" and seg.raw_upper in ["LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS"]:
                join_type = seg.raw_upper
                # If OUTER follows LEFT/RIGHT/FULL, keep the original join_type
                if join_type == "OUTER":
                    join_type = "INNER"
                break

        for sub in join_clause.segments:
            if sub.type == "from_expression_element":
                if table := find_table_identifier(sub):
                    sources[table.raw].add(join_type)

    for elem in segment.recursive_crawl("from_expression_element"):
        if is_subquery(elem):
            subqueries = list_subqueries(elem)
            for sq in subqueries:
                if table := find_table_identifier(sq.segment):
                    sources[table.raw].add("FROM")
        else:
            if table := find_table_identifier(elem):
                sources[table.raw].add("FROM")

    return sources, targets

# ---------------- PROCESS SQL FILES ---------------- #
def process_sql_files(folder_paths):
    linter = Linter(dialect="snowflake")
    lineage_graph = defaultdict(list)
    node_metadata = {}

    for folder_path in folder_paths:
        if DEBUG:
            print(f"\U0001F4C2 Processing: {folder_path}")
        if not os.path.exists(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".sql"):
                current_sql_filename = file
                file_path = os.path.join(folder_path, file)
                if DEBUG:
                    print(f"\U0001F50E {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    sql = f.read()
                result = linter.parse_string(sql)
                if result.tree is None:
                    continue
                ctes = set()
                for segment in result.tree.recursive_crawl("statement"):
                    real_stmt = segment.segments[0] 
                    sources, targets = extract_tables_from_segment(real_stmt, ctes)
                    if DEBUG:
                        print(f"  - Sources: {sources}")
                        print(f"  - Targets: {targets}")

                    operation = ""
                    is_temp = False
                    if real_stmt.type == "insert_statement":
                        operation = "INSERT"
                    elif real_stmt.type == "merge_statement":
                        operation = "MERGE"
                    elif real_stmt.type == "create_table_as_statement":
                        operation = "CREATE TABLE AS"
                    elif real_stmt.type == "update_statement":
                        operation = "UPDATE"
                    elif real_stmt.type == "delete_statement":
                        operation = "DELETE"
                    elif real_stmt.type == "create_table_statement":
                        operation = "CREATE"
                        is_temp = "TEMP" in sql.upper() or "TEMPORARY" in sql.upper()

                    for target in targets:
                        target_str = target if isinstance(target, str) else str(target)
                        if target_str not in node_metadata:
                            node_metadata[target_str] = {
                                "type": "target",
                                "operation": [],
                                "is_temp": "TEMP" in target_str.upper() or "TEMPORARY" in target_str.upper(),
                                "file": current_sql_filename
                            }
                        if operation and operation not in node_metadata[target_str]["operation"]:
                            node_metadata[target_str]["operation"].append(operation)
                    for source in sources:
                        if isinstance(source, tuple):
                            source_name, _ = source
                        else:
                            source_name = source

                        source_name_str = source_name if isinstance(source_name, str) else str(source_name)
                        if source_name_str not in node_metadata:
                            node_metadata[source_name_str] = {
                                "type": "source",
                                "operation": [],
                                "is_temp": "TEMP" in source_name_str.upper() or "TEMPORARY" in source_name_str.upper(),
                                "file": current_sql_filename
                            }

                    # Consolidate multiple join types per source into the most specific one
                    join_priority = {
                        "CROSS": 1,
                        "LEFT": 2,
                        "RIGHT": 2,
                        "FULL": 2,
                        "INNER": 3,
                        "FROM": 4,
                        "UNKNOWN": 5
                    }
                    consolidated_sources = {}
                    for name, join_type_set in sources.items():
                        best = min(join_type_set, key=lambda jt: join_priority.get(jt, 99))
                        consolidated_sources[name] = best

                    for target in targets:
                        target_str = target if isinstance(target, str) else str(target)
                        if target_str in ctes:
                            continue
                        for source_name, join_type in consolidated_sources.items():
                            source_str = source_name if isinstance(source_name, str) else str(source_name)
                            if source_str != target_str:
                                existing_links = lineage_graph[source_str]
                                has_known = any(t == target_str and jt != "UNKNOWN" for t, jt in existing_links)
                                if not has_known:
                                    if (target_str, join_type) not in existing_links:
                                        lineage_graph[source_str].append((target_str, join_type))

    return lineage_graph, node_metadata

# ---------------- SAVE TO JSON ---------------- #
def save_lineage_to_json(graph, metadata, output_file='lineage_output.json'):
    with open(output_file, 'w', encoding='utf-8') as f:
        # Ensure deterministic node order using OrderedDict and sorting
        node_dict = OrderedDict()
        for source in graph.keys():
            node_dict[str(source)] = metadata.get(str(source), {})
        for targets in graph.values():
            for tgt, _ in targets:
                node_dict[str(tgt)] = metadata.get(str(tgt), {})

        nodes = []
        for node_id, meta in sorted(node_dict.items()):
            op_map = {"INSERT": "I", "UPDATE": "U", "MERGE": "M", "CREATE": "C", "DELETE": "D", "CREATE TABLE AS": "C"}
            # If meta.get("operation", []) is empty, set operation to "" instead of an empty list
            nodes.append({
                "id": node_id,
                "type": meta.get("type", "transform"),
                "operation": (
                    [op_map.get(op, op) for op in meta["operation"]] if meta.get("operation") else ""
                ),
                "is_temp": meta.get("is_temp", False),
                "file": meta.get("file", None)
            })
        seen_links = {}
        link_list = []

        def better_join_type(existing, new):
            # Always resolve to the more specific join type if duplicates exist
            join_priority = {
                "CROSS": 1,
                "LEFT": 2,
                "RIGHT": 2,
                "FULL": 2,
                "INNER": 3,
                "FROM": 4,
                "UNKNOWN": 5
            }
            if existing == new:
                return existing
            return existing if join_priority.get(existing, 99) < join_priority.get(new, 99) else new

        for source, targets in graph.items():
            source_str = source if isinstance(source, str) else str(source)
            for target, join_type in targets:
                target_str = target if isinstance(target, str) else str(target)
                key = (source_str, target_str)
                if key in seen_links:
                    seen_links[key] = better_join_type(seen_links[key], join_type)
                else:
                    seen_links[key] = join_type

        for (source, target), join_type in seen_links.items():
            link_list.append({"source": source, "target": target, "join_type": join_type})

        output = {
            "nodes": nodes,
            "links": link_list
        }
        f.write('{\n')
        f.write('  "nodes": [\n')
        for i, node in enumerate(output["nodes"]):
            comma = ',' if i < len(output["nodes"]) - 1 else ''
            f.write(f'    {json.dumps(node)}{comma}\n')
        f.write('  ],\n')
        f.write('  "links": [\n')
        for i, link in enumerate(output["links"]):
            comma = ',' if i < len(output["links"]) - 1 else ''
            f.write(f'    {json.dumps(link)}{comma}\n')
        f.write('  ]\n')
        f.write('}\n')
    print(f"\u2705 Saved to {output_file}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    DEBUG = True
    folder_paths = ["/Users/ryakamkishore/Lineage/20250429/SQL_path"]  # Update this path
    graph, metadata = process_sql_files(folder_paths)
    save_lineage_to_json(graph, metadata)
