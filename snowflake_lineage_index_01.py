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
    sources = defaultdict(set)
    targets = set()
    if DEBUG:
        print({segment})
        print(segment.to_tuple(show_raw=True))
    # Handle WITH clause: collect CTEs and extract sources/targets for embedded statements
    if segment.type == "with_compound_statement":
        if cte_clause := segment.get_child("common_table_expression"):
            for identifier in cte_clause.recursive_crawl("identifier"):
                ctes.add(identifier.raw)
            # Find the statement after the WITH clause (e.g., INSERT, SELECT, etc.)
            for seg in segment.segments:
                if seg.type in ("insert_statement", "update_statement", "delete_statement", "merge_statement", "create_table_as_statement"):
                    sources_inner, targets_inner = extract_tables_from_segment(seg, ctes)
                    sources |= sources_inner
                    targets |= targets_inner
        # Also extract sources from CTE definitions
        for cte_def in segment.recursive_crawl("common_table_expression"):
            for select in cte_def.recursive_crawl("select_statement"):
                cte_sources, _ = extract_tables_from_segment(select, ctes)
                sources |= cte_sources
    if DEBUG:
        print(f"segment.type:{segment.type}")
    # Handle INSERT INTO ... SELECT ... (including CREATE TEMP + INSERT)
    if segment.type == "insert_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
        # Extract sources from SELECT statement in the insert
        # Extract sources from all set expressions (e.g., UNION, UNION ALL)
        for set_expr in segment.recursive_crawl("set_expression"):
            for sel in set_expr.recursive_crawl("select_statement"):
                sel_sources, _ = extract_tables_from_segment(sel, ctes)
                sources |= sel_sources
            # Also handle any direct FROMs at set level
            for from_elem in set_expr.recursive_crawl("from_expression_element"):
                if is_subquery(from_elem):
                    subqueries = list_subqueries(from_elem)
                    for sq in subqueries:
                        if table := find_table_identifier(sq.segment):
                            if table.raw not in ctes:
                                sources[table.raw].add("FROM")
                else:
                    if table := find_table_identifier(from_elem):
                        if table.raw not in ctes:
                            sources[table.raw].add("FROM")

        # Fallback: extract sources from any direct select_statement if set_expression was not used
        for sel in segment.recursive_crawl("select_statement"):
            sel_sources, _ = extract_tables_from_segment(sel, ctes)
            sources |= sel_sources

        # Also include from_expression_element outside of select/set_expression
        for from_elem in segment.recursive_crawl("from_expression_element"):
            if is_subquery(from_elem):
                subqueries = list_subqueries(from_elem)
                for sq in subqueries:
                    if table := find_table_identifier(sq.segment):
                        if table.raw not in ctes:
                            sources[table.raw].add("FROM")
            else:
                if table := find_table_identifier(from_elem):
                    if table.raw not in ctes:
                        sources[table.raw].add("FROM")
    # Handle CREATE TABLE ... AS SELECT ...
    elif segment.type == "create_table_as_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
        for sel in segment.recursive_crawl("select_statement"):
            sel_sources, _ = extract_tables_from_segment(sel, ctes)
            sources |= sel_sources
    # Handle CREATE TEMP TABLE
    elif segment.type == "create_table_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
    # Handle MERGE INTO ... USING ...
    elif segment.type == "merge_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
        # Find USING clause's source table
        using_found = False
        for seg in segment.segments:
            if seg.type == "keyword" and seg.raw_upper == "USING":
                using_found = True
            elif using_found:
                # Could be a subquery or table reference
                if seg.type == "table_reference":
                    sources[seg.raw].add("USING")
                    break
                elif is_subquery(seg):
                    # Subquery: extract sources recursively
                    sub_sources, _ = extract_tables_from_segment(seg, ctes)
                    for s in sub_sources:
                        sources[s].add("USING")
                    break
    # Handle TRUNCATE TABLE
    elif segment.type == "truncate_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
            # Mark this as TRUNCATE operation elsewhere
    # Handle UPDATE ... SET ... FROM ...
    elif segment.type == "update_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
        # Extract sources from FROM clause if present
        for from_clause in segment.recursive_crawl("from_clause"):
            for elem in from_clause.recursive_crawl("from_expression_element"):
                if is_subquery(elem):
                    subqueries = list_subqueries(elem)
                    for sq in subqueries:
                        if table := find_table_identifier(sq.segment):
                            sources[table.raw].add("FROM")
                else:
                    if table := find_table_identifier(elem):
                        sources[table.raw].add("FROM")
    # Handle DELETE ... FROM ...
    elif segment.type == "delete_statement":
        if tgt := safe_get_table_reference(segment):
            targets.add(tgt.raw)
        # Extract sources from USING clause if present
        for using_clause in segment.recursive_crawl("using_clause"):
            for elem in using_clause.recursive_crawl("from_expression_element"):
                if is_subquery(elem):
                    subqueries = list_subqueries(elem)
                    for sq in subqueries:
                        if table := find_table_identifier(sq.segment):
                            sources[table.raw].add("USING")
                else:
                    if table := find_table_identifier(elem):
                        sources[table.raw].add("USING")

    # Handle joins (for select, merge, etc.)
    for join_clause in segment.recursive_crawl("join_clause"):
        if DEBUG:
            print("join_clause", join_clause.to_tuple(show_raw=True))
        join_type = "INNER"  # Default
        for seg in join_clause.segments:
            if seg.type == "keyword" and seg.raw_upper in ["LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS"]:
                join_type = seg.raw_upper
                if join_type == "OUTER":
                    join_type = "INNER"
                break
        for sub in join_clause.segments:
            if sub.type == "from_expression_element":
                if table := find_table_identifier(sub):
                    sources[table.raw].add(join_type)

    # For selects and others: collect all FROM sources (skip CTEs)
    for elem in segment.recursive_crawl("from_expression_element"):
        if is_subquery(elem):
            subqueries = list_subqueries(elem)
            for sq in subqueries:
                if table := find_table_identifier(sq.segment):
                    if table.raw not in ctes:
                        sources[table.raw].add("FROM")
        else:
            if table := find_table_identifier(elem):
                if table.raw not in ctes:
                    sources[table.raw].add("FROM")

    return sources, targets

# ---------------- PROCESS SQL FILES ---------------- #
def process_sql_files(folder_paths):
    linter = Linter(dialect="snowflake")
    lineage_graph = defaultdict(list)
    node_metadata = {}
    table_map = defaultdict(set)

    for folder_path in folder_paths:
        if DEBUG:
            print(f"\U0001F4C2 Processing: {folder_path}")
        if not os.path.exists(folder_path):
            continue

        for file in sorted(os.listdir(folder_path)):
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
                    # --- Patch: also include direct FROM sources for insert_statement and create_table_as_statement ---
                    if real_stmt.type == "insert_statement":
                        # After extracting sources from select_statement, also include direct FROMs (e.g., top-level UNION ALL)
                        for sel in real_stmt.recursive_crawl("select_statement"):
                            sel_sources, _ = extract_tables_from_segment(sel, ctes)
                            sources |= sel_sources
                        # Also include sources from direct FROM clauses at top level (e.g., UNION ALL components not wrapped in select_statement)
                        for from_elem in real_stmt.recursive_crawl("from_expression_element"):
                            if is_subquery(from_elem):
                                subqueries = list_subqueries(from_elem)
                                for sq in subqueries:
                                    if table := find_table_identifier(sq.segment):
                                        if table.raw not in ctes:
                                            sources[table.raw].add("FROM")
                            else:
                                if table := find_table_identifier(from_elem):
                                    if table.raw not in ctes:
                                        sources[table.raw].add("FROM")
                    elif real_stmt.type == "create_table_as_statement":
                        for sel in real_stmt.recursive_crawl("select_statement"):
                            sel_sources, _ = extract_tables_from_segment(sel, ctes)
                            sources |= sel_sources
                        # Also include sources from direct FROM clauses at top level (e.g., UNION ALL components not wrapped in select_statement)
                        for from_elem in real_stmt.recursive_crawl("from_expression_element"):
                            if is_subquery(from_elem):
                                subqueries = list_subqueries(from_elem)
                                for sq in subqueries:
                                    if table := find_table_identifier(sq.segment):
                                        if table.raw not in ctes:
                                            sources[table.raw].add("FROM")
                            else:
                                if table := find_table_identifier(from_elem):
                                    if table.raw not in ctes:
                                        sources[table.raw].add("FROM")
                    if DEBUG:
                        print(f"  - Sources: {sources}")
                        print(f"  - Targets: {targets}")

                    # Determine operation type for target table(s)
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
                    elif real_stmt.type == "truncate_statement":
                        operation = "TRUNCATE"

                    # Merge sources and targets for table_map and node_metadata
                    combined_tables = set(targets)
                    combined_tables.update(sources.keys())
                    for table in combined_tables:
                        table_str = table.upper() if isinstance(table, str) else str(table).upper()
                        # Determine node type
                        if table_str in (t.upper() if isinstance(t, str) else str(t).upper() for t in targets):
                            ntype = "target"
                            is_temp_val = "TEMP" in table_str.upper() or "TEMPORARY" in table_str.upper()
                        else:
                            ntype = "source"
                            is_temp_val = "TEMP" in table_str.upper() or "TEMPORARY" in table_str.upper()
                        if table_str not in node_metadata:
                            node_metadata[table_str] = {
                                "type": ntype,
                                "operation": [],
                                "is_temp": is_temp_val,
                                "file": current_sql_filename
                            }
                        # Allow multiple operations (e.g. CREATE and INSERT for temp tables)
                        if ntype == "target" and operation and operation not in node_metadata[table_str]["operation"]:
                            node_metadata[table_str]["operation"].append(operation)
                        # Ensure current_sql_filename is always added for both source and target roles
                        if current_sql_filename not in table_map[table_str.upper()]:
                            table_map[table_str.upper()].add(current_sql_filename)

                    # Only track which files a table appears in (as source or target)
                    for table in combined_tables:
                        table_str = table.upper() if isinstance(table, str) else str(table).upper()
                        if current_sql_filename not in table_map[table_str]:
                            table_map[table_str].add(current_sql_filename)

                    # Consolidate multiple join types per source into the most specific one
                    join_priority = {
                        "CROSS": 1,
                        "LEFT": 2,
                        "RIGHT": 2,
                        "FULL": 2,
                        "INNER": 3,
                        "FROM": 4,
                        "USING": 4,
                        "UNKNOWN": 5
                    }
                    consolidated_sources = {}
                    for name, join_type_set in sources.items():
                        best = min(join_type_set, key=lambda jt: join_priority.get(jt, 99))
                        name_upper = name.upper() if isinstance(name, str) else str(name).upper()
                        consolidated_sources[name_upper] = best

                    for target in targets:
                        target_str = target.upper() if isinstance(target, str) else str(target).upper()
                        if target_str in (ct.upper() for ct in ctes):
                            continue
                        for source_name, join_type in consolidated_sources.items():
                            source_str = source_name.upper() if isinstance(source_name, str) else str(source_name).upper()
                            if source_str != target_str:
                                existing_links = lineage_graph[source_str]
                                has_known = any(t == target_str and jt != "UNKNOWN" for t, jt in existing_links)
                                if not has_known:
                                    if (target_str, join_type) not in existing_links:
                                        lineage_graph[source_str].append((target_str, join_type))

    # After processing all files, sort the sets in table_map for deterministic output
    for key in table_map:
        table_map[key] = sorted(table_map[key])
    return lineage_graph, node_metadata, table_map

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
            op_map = {"INSERT": "I", "UPDATE": "U", "MERGE": "M", "CREATE": "C", "DELETE": "D", "CREATE TABLE AS": "C", "TRUNCATE": "T"}
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


# ---------------- SAVE PARTITIONED JSON PER SQL FILE ---------------- #
def save_partitioned_json(graph, metadata, output_dir='output_partitioned'):
    os.makedirs(output_dir, exist_ok=True)
    file_map = defaultdict(lambda: {"nodes": [], "links": []})
    op_map = {"INSERT": "I", "UPDATE": "U", "MERGE": "M", "CREATE": "C", "DELETE": "D", "CREATE TABLE AS": "C", "TRUNCATE": "T"}

    # Aggregate nodes by file
    for node_id, meta in metadata.items():
        #print(f"node_id:{node_id}")
        entry = {
            "id": node_id,
            "type": meta.get("type", "transform"),
            "operation": (
                [op_map.get(op, op) for op in meta["operation"]] if meta.get("operation") else ""
            ),
            "is_temp": meta.get("is_temp", False),
            "file": meta.get("file")
        }
        if meta.get("file"):
            file_map[meta["file"]]["nodes"].append(entry)

    # Aggregate links by file (save in both source and target files)
    for src, tgt_list in graph.items():
        for tgt, join_type in tgt_list:
            source_meta = metadata.get(src, {})
            target_meta = metadata.get(tgt, {})
            source_file = source_meta.get("file")
            target_file = target_meta.get("file")
            for file in {source_file, target_file}:
                if file:
                    file_map[file]["links"].append({
                        "source": str(src),
                        "target": str(tgt),
                        "join_type": join_type
                    })

    # Save per-file JSONs
    for file, data in file_map.items():
        out_path = os.path.join(output_dir, file.replace(".sql", ".json"))
        # Sort nodes by id and links by (source, target) for deterministic output
        data["nodes"] = sorted(data["nodes"], key=lambda x: x["id"])
        data["links"] = sorted(data["links"], key=lambda x: (x["source"], x["target"]))
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('{\n')
            f.write('  "nodes": [\n')
            for i, node in enumerate(data["nodes"]):
                comma = ',' if i < len(data["nodes"]) - 1 else ''
                f.write(f'    {json.dumps(node)}{comma}\n')
            f.write('  ],\n')
            f.write('  "links": [\n')
            for i, link in enumerate(data["links"]):
                comma = ',' if i < len(data["links"]) - 1 else ''
                f.write(f'    {json.dumps(link)}{comma}\n')
            f.write('  ]\n')
            f.write('}\n')
    print(f"✅ Partitioned JSON saved to: {output_dir}")


# ---------------- BUILD TABLE-TO-FILE INDEX ---------------- #
def save_table_to_file_index(table_map, index_path='table_file_index.json'):
    # Normalize table names (uppercase) before saving
    normalized_index = {}
    for k, v in table_map.items():
        key_upper = k.upper()
        if key_upper not in normalized_index:
            normalized_index[key_upper] = set()
        normalized_index[key_upper].update(v)

    # Convert sets to sorted lists for JSON output, and sort table names for deterministic output
    table_index = dict(sorted((k, sorted(list(v))) for k, v in normalized_index.items()))

    with open(index_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        last_key = list(table_index.keys())[-1]
        for key, value in table_index.items():
            comma = ',' if key != last_key else ''
            f.write(f'  {json.dumps(key)}: {json.dumps(value)}{comma}\n')
        f.write('}\n')
    print(f"✅ Table-to-file index saved to: {index_path}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    # Delete old JSON files if they exist
    for f in ["lineage_output.json", "table_file_index.json"]:
        try:
            os.remove(f)
            print(f"🗑️ Deleted old {f}")
        except FileNotFoundError:
            pass

    import shutil
    if os.path.exists("output_partitioned"):
        shutil.rmtree("output_partitioned")
        print("🗑️ Deleted old output_partitioned directory")


    DEBUG = False
    folder_paths = ["/Users/ryakamkishore/Lineage/20250429/SQL_path"]  # Update this path
    graph, metadata, table_map = process_sql_files(folder_paths)
    save_lineage_to_json(graph, metadata)
    save_partitioned_json(graph, metadata)
    save_table_to_file_index(table_map)
