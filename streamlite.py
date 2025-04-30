import streamlit as st
from pyvis.network import Network
import json
import tempfile
import os

st.set_page_config(layout="wide")
st.title("🕸️ Interactive SQL Lineage DAG Viewer")

# Load JSON lineage file
with open("lineage_output.json") as f:
    lineage = json.load(f)

# Target Table
target_table = st.text_input("🎯 Enter Target Table (exact name):", "REPORTING_DB.HIGH_VALUE_CUSTOMER_SUMMARY")

direction = st.radio("📍 Lineage Direction", ["Upstream", "Downstream", "Both"])

# Build directed graph from lineage
net = Network(height="1000px", directed=True, bgcolor="#ffffff", font_color="black")
net.set_options("""
{
  "layout": {
    "hierarchical": {
      "enabled": false,
      "direction": "LR",
      "sortMethod": "directed",
      "nodeSpacing": 120,
      "levelSeparation": 100
    }
  },
  "physics": {
    "enabled": false
  },
  "edges": {
    "smooth": {
      "type": "cubicBezier",
      "forceDirection": "vertical",
      "roundness": 0.4
    },
    "arrows": {
      "to": {
        "enabled": true
      }
    }
  }
}
""")
added_nodes = set()

def add_upstream(current, lineage_map, visited):
    for src, tgts in lineage_map.items():
        if current in tgts and (src, current) not in visited:
            net.add_node(src, label=src.split('.')[-1], title=src, shape="box", color="#ADD8E6")
            net.add_node(current, label=current.split('.')[-1], title=current, shape="box", color="#90EE90")
            net.add_edge(src, current)
            visited.add((src, current))
            add_upstream(src, lineage_map, visited)

def add_downstream(current, lineage_map, visited):
    if current in lineage_map:
        for tgt in lineage_map[current]:
            if (current, tgt) not in visited:
                net.add_node(current, label=current.split('.')[-1], title=current, shape="box", color="#90EE90")
                net.add_node(tgt, label=tgt.split('.')[-1], title=tgt, shape="box", color="#ADD8E6")
                net.add_edge(current, tgt)
                visited.add((current, tgt))
                add_downstream(tgt, lineage_map, visited)

def add_bidirectional(current, lineage_map, visited):
    add_upstream(current, lineage_map, visited)
    add_downstream(current, lineage_map, visited)

if direction == "Upstream":
    add_upstream(target_table, lineage, set())
elif direction == "Downstream":
    add_downstream(target_table, lineage, set())
else:
    add_bidirectional(target_table, lineage, set())

# Save graph to temp HTML file
with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
    net.save_graph(tmp_file.name)
    tmp_path = tmp_file.name

# Render in Streamlit
st.components.v1.html(open(tmp_path).read(), height=600, scrolling=True)

# Clean up temp file after rendering
os.unlink(tmp_path)
