from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
import networkx as nx
import matplotlib.pyplot as plt



def extract_text_from_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=200) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    return chunks

def summarize_chunks(chunks,model) -> list[str]:
  summary_list = []
  for chunk in chunks:
    prompt_summary = f"Summarize the following text: {chunk.page_content}"
    summary = model.invoke(prompt_summary)
    summary_list.append(summary.content)
  return summary_list

def create_json_from_text(text):
  prompt = f"""
  Extract named entities and their relationships from the following text. 
  Format the output as a JSON object with two keys: 
  - "entities": a list of unique entity names
  - "relations": a list of relationships, where each item is an object with:
    - "source" (the first entity)
    - "target" (the second entity)
    - "relation" (the type of relationship)

  Text:
  {text[:4000]}  # จำกัดไม่เกิน 4,000 ตัวอักษร (GPT-4o-mini รับ input ได้ประมาณนี้)

  Output should be in JSON format:
  """

  client = OpenAI()
  response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "system", "content": "You are an AI that extracts structured knowledge from text."},
              {"role": "user", "content": prompt}]
  )

  output_json = response.choices[0].message.content
  clean_json_text = output_json.strip("```json").strip("```").strip()
  data = json.loads(clean_json_text)
  return data
  
def create_knowledge_graph(data):
    # สร้าง Knowledge Graph ด้วย NetworkX
    G = nx.DiGraph()

    # เพิ่ม Nodes (Entity)
    for entity in data["entities"]:
        G.add_node(entity)

    # เพิ่ม Relationships (Edges)
    for relation in data["relations"]:
        G.add_edge(relation["source"], relation["target"], label=relation["relation"])

        # วาดกราฟ
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=8)

    # แสดงชื่อความสัมพันธ์บนเส้นเชื่อม
    edge_labels = {(rel["source"], rel["target"]): rel["relation"] for rel in data["relations"]}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    # plt.title("Knowledge Graph from PDF (Using GPT-4o-mini)")
    plt.savefig("knowledge_graph.png", format="png")
    print("Knowledge Graph saved as knowledge_graph.png")