from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import pandas as pd
from index_mapping import mappings
import json
import warnings
warnings.filterwarnings("ignore")


# Create an instance of Elasticsearch and connect to Elasticsearch cluster. If you have a remote cluster, 
# you can use the following format:
# client = Elasticsearch("https://my-elasticsearch-project-ce982d.es.us-east-1.aws.elastic.cloud:443", api_key="YOUR_API_KEY")
# Here, I am using a local Elasticsearch instance.
try:
    es = Elasticsearch("https://localhost:9200", basic_auth=('elastic', 'mvsFaXzEkUiSOhanK2lS'), verify_certs=False)
except ConnectionError as e:
    print(f"Connection Error: {e}")
    
if es.ping():
    print("Successfully connected to Elasticsearch!")
else:
    print("Oops! Cannot connect to Elasticsearch!")


# Initialize the SentenceTransformer model (SBERT)
model = SentenceTransformer('all-mpnet-base-v2')

# Get embeddings for the description column using SBERT model
def get_embeddings(text):
    try:
        text = text.replace('\n', ' ')
        return model.encode(text)
    except Exception as e:
        print("Error getting embeddings:", e)

# Define the index name
index_name = "hybrid_search_index"


# Create index if it doesn't exist
def create_index():
    try:
        if es.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
        else:
            mapping = mappings  # We'll use the mappings defined in index_mapping.py
            es.indices.create(index=index_name, mappings=mapping)
            print(f"Index '{index_name}' created successfully.")
    except Exception as e:
            print(f"Error creating index '{index_name}': {e}")


# Load data
df = pd.read_csv('./data.csv')
def load_data():
    df = pd.read_csv('./data.csv')
    return df

# Convert data to Elasticsearch bulk format
actions = [
    {
        "_index": index_name,
        "_source": {
            "product_name": row["ProductName"],
            "description": row["Description"],
            "price": row["Price"],
            "tags": row["Tags"],
            "description_vector": get_embeddings(row["Description"])
        }
    }
    for _, row in df.iterrows()
]

# print(actions)

# Bulk index data
helpers.bulk(es, actions)
print("Data indexed successfully.")

# Perform lexical search
def lexical_search(query: str, top_k: int):
    # Useing Elasticsearch's built-in multi_match query to search across product names and descriptions based on the query terms.
    lexical_results = es.search(
        index=index_name,
        body={
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["product_name", "description"]
                }
            }
        },
        size=top_k,
        source_excludes=["description_vector"]
    )

    lexical_hits = lexical_results["hits"]["hits"]

    # Best scoring hit among lexical hits
    max_bm25_score = max([hit["_score"] for hit in lexical_hits], default=1.0)
    
    # Normalize scores
    for hit in lexical_hits:
        hit["_normalized_score"] = hit["_score"] / max_bm25_score

    return lexical_hits


# Perform semantic search
def semantic_search(query: str, top_k: int):
    query_embedding = get_embeddings(query)
    semantic_results = es.search(
        index=index_name,
        body={
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_embedding, 'description_vector') + 1.0", # Uses the cosine similarity between the query embedding and the description embeddings stored in Elasticsearch.
                        "params": {
                            "query_embedding": query_embedding
                        }
                    }
                }
            }
        },
        size=top_k,
        source_excludes=["description_vector"]
    )

    semantic_hits = semantic_results["hits"]["hits"]
    max_semantic_score = max([hit["_score"] for hit in semantic_hits], default=1.0)

    for hit in semantic_hits:
        hit["_normalized_score"] = hit["_score"] / max_semantic_score
    return semantic_hits

# Combines the lexical and semantic search results using Reciprocal Rank Fusion (RRF).
# Assigns higher scores to documents that appear earlier in the ranking, and it merges results based on rank
def reciprocal_rank_fusion(lexical_hits, semantic_hits, k=60):
    rrf_scores = {}

    # Process lexical hits
    for rank, hit in enumerate(lexical_hits, start=1):
        doc_id = hit["_id"]
        score = 1 / (k + rank)

        if doc_id in rrf_scores:
            rrf_scores[doc_id]["rrf_score"] += score
        else:
            rrf_scores[doc_id] = {
                "product_name": hit["_source"]["product_name"],
                "description": hit["_source"]["description"],
                "lexical_score": hit["_normalized_score"],
                "semantic_score": 0,
                "rrf_score": score}

    # Process semantic hits
    for rank, hit in enumerate(semantic_hits, start=1):
        doc_id = hit["_id"]
        score = 1 / (k + rank)
        if doc_id in rrf_scores:
            rrf_scores[doc_id]["semantic_score"] += score
        else:
            rrf_scores[doc_id] = {
                "product_name": hit["_source"]["product_name"],
                "description": hit["_source"]["description"],
                "lexical_score": 0,
                "semantic_score": hit["_normalized_score"],
                "rrf_score": score}

    # Sort the results by RRF score
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    return sorted_results


def remove_duplicates_and_rerank(rrf_hits, rerank=False):
    # Create a dictionary to store unique results based on product ID
    combined_results = {}

    # Loop through RRF hits and process them
    for hit in rrf_hits:
        doc_id = hit.get("_id")

        # If the product already exists in the results, we update the RRF score
        if doc_id in combined_results:
            # Sum the RRF scores of the duplicate results (both lexical and semantic scores)
            combined_results[doc_id]["rrf_score"] += hit["rrf_score"]
        else:
            # Add the current result as a new entry
            combined_results[doc_id] = {
                "product_name": hit["product_name"],
                "description": hit["description"],
                "lexical_score": hit.get("lexical_score", 0),  # Use 0 if lexical_score doesn't exist
                "semantic_score": hit.get("semantic_score", 0),  # Use 0 if semantic_score doesn't exist
                "rrf_score": hit["rrf_score"]  # RRF score from the current hit
            }

    # Convert the dictionary values back to a list of results
    results_list = list(combined_results.values())

    # If rerank is True, we sort the results based on the combined RRF score
    if rerank:
        # Sort results in descending order of the RRF score
        results_list = sorted(results_list, key=lambda x: x["rrf_score"], reverse=True)

    return results_list


# Hybrid search combines lexical and semantic search results using RRF
def hybrid_search(query: str, lexical_top_k, semantic_top_k, rerank=False):
    lexical_hits = lexical_search(query, lexical_top_k)
    semantic_hits = semantic_search(query, semantic_top_k)
    
    # Combine results using Reciprocal Rank Fusion (RRF)
    rrf_results = reciprocal_rank_fusion(lexical_hits, semantic_hits, k=60)
    
    # Remove duplicates and rerank if rerank is True
    final_results = remove_duplicates_and_rerank(rrf_results, rerank=rerank)
    
    # Print results in a readable format
    print(json.dumps(final_results, indent=4))
    return final_results


if __name__ == "__main__":
    input_query = "Headphones for travel"
    hybrid_search(input_query, 10, 10, rerank=True)
    print(es.count(index=index_name))