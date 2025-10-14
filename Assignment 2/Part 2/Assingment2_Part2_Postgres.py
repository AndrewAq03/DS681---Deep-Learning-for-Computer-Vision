#Andrew Aquino 
#My attempt to implement Postgres for similarity search

import os
import torch
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

ASSET_DIR = "/workspaces/eng-ai-agents/assignments/assignment-2/patchcore_assets"

DB_CONFIG = {
    "dbname": "template1",
    "user": "postgres",
    "password": "1234",
    "host": "host.docker.internal",
    "port": "5432"
}

FEATURE_DIMENSION = 1536

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
register_vector(conn)

cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute(f"""
CREATE TABLE IF NOT EXISTS patchcore_embeddings (
    id SERIAL PRIMARY KEY,
    category TEXT,
    image_name TEXT UNIQUE,
    embedding VECTOR({FEATURE_DIMENSION}),
    is_anomalous BOOLEAN DEFAULT FALSE
);
""")
conn.commit()

def insert_embeddings():
    for file in os.listdir(ASSET_DIR):
        if not file.endswith("_memory_bank.pt"):
            continue
        category = file.replace("_memory_bank.pt", "")
        path = os.path.join(ASSET_DIR, file)
        memory_bank = torch.load(path, map_location="cpu")
        features = memory_bank.get("features") if isinstance(memory_bank, dict) else memory_bank
        if features is None:
            continue
        for i, emb in enumerate(features):
            emb_list = emb.cpu().numpy().astype(np.float32).tolist()
            image_name = f"{category}_train_coreset_{i}"
            cur.execute("""
                INSERT INTO patchcore_embeddings (category, image_name, embedding, is_anomalous)
                VALUES (%s, %s, %s::vector, %s)
                ON CONFLICT (image_name) DO NOTHING;
            """, (category, image_name, emb_list, False))
        conn.commit()
    print("Inserted all embeddings.")
    
def insert_fake_anomalies(n=5):
    cur.execute("SELECT embedding, category FROM patchcore_embeddings LIMIT %s;", (n,))
    rows = cur.fetchall()
    for i, (emb, cat) in enumerate(rows):
        image_name = f"{cat}_anomaly_{i}"
        cur.execute("""
            INSERT INTO patchcore_embeddings (category, image_name, embedding, is_anomalous)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (image_name) DO NOTHING;
        """, (cat, image_name, emb, True))
    conn.commit()
    print(f"Inserted {n} fake anomalies.")
    
def insert_anomalies(n=5):
    cur.execute("SELECT embedding, category FROM patchcore_embeddings LIMIT %s;", (n,))
    rows = cur.fetchall()
    for i, (emb, cat) in enumerate(rows):
        image_name = f"{cat}_anomaly_{i}"
        cur.execute("""
            INSERT INTO patchcore_embeddings (category, image_name, embedding, is_anomalous)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (image_name) DO NOTHING;
        """, (cat, image_name, emb, True))
    conn.commit()
    print(f"Inserted {n} fake anomalies.")

def get_neighbors(embedding, top_k=10, only_anomalous=None):
    emb_str = "[" + ",".join(map(str, embedding.tolist())) + "]"
    query = """
        SELECT image_name, category, is_anomalous, 1 - (embedding <=> %s::vector) AS similarity
        FROM patchcore_embeddings
    """
    if only_anomalous is not None:
        query += f" WHERE is_anomalous = {str(only_anomalous).lower()}"
    query += " ORDER BY embedding <=> %s::vector LIMIT %s;"
    cur.execute(query, (emb_str, emb_str, top_k))
    return cur.fetchall()

def find_similar_or_anomaly():
    cur.execute("SELECT embedding, category, image_name FROM patchcore_embeddings WHERE is_anomalous=FALSE ORDER BY random() LIMIT 1;")
    row = cur.fetchone()
    if not row:
        print("No embeddings found.")
        return
    embedding = np.array(row[0], dtype=np.float32)
    print(f"\nQuery: {row[2]} ({row[1]})")

    results = get_neighbors(embedding, 10, only_anomalous=False)
    best_sim = results[0][3] if results else 0.0
    THRESHOLD = 0.9999

    if best_sim < THRESHOLD:
        print("\nAnomaly detected — showing similar anomalies:")
        results = get_neighbors(embedding, 10, only_anomalous=True)
    else:
        print("\nNormal image — showing top similar normals:")
    for img, cat, anom, sim in results:
        print(f"{img:<25} | {cat:<10} | {'Anom' if anom else 'Norm'} | Sim={sim:.4f}")


#insert_fake_anomalies()
insert_anomalies() 
find_similar_or_anomaly()
cur.close()
conn.close()
