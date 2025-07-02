#!/usr/bin/env python3
"""
Inspect ANTIQUE Dataset Qrels
==============================

This script inspects the ANTIQUE qrels to ensure we understand the format correctly
and are doing evaluation properly.
"""

import ir_datasets
import pandas as pd
from collections import defaultdict, Counter

def inspect_antique_qrels():
    """Inspect ANTIQUE qrels format and statistics"""
    print("🔍 Inspecting ANTIQUE qrels format...")
    
    # Load ANTIQUE test dataset
    dataset = ir_datasets.load('antique/test')
    
    # Collect qrels
    qrels_list = []
    for qrel in dataset.qrels_iter():
        qrels_list.append({
            'query_id': qrel.query_id,
            'doc_id': qrel.doc_id,
            'relevance': qrel.relevance
        })
    
    print(f"📊 Total qrels: {len(qrels_list)}")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(qrels_list)
    
    # Relevance level distribution
    print(f"\n📈 Relevance level distribution:")
    relevance_counts = df['relevance'].value_counts().sort_index()
    for level, count in relevance_counts.items():
        print(f"   Level {level}: {count:,} judgments")
    
    # Queries with judgments
    unique_queries = df['query_id'].nunique()
    print(f"\n📊 Unique queries with judgments: {unique_queries}")
    
    # Documents with judgments  
    unique_docs = df['doc_id'].nunique()
    print(f"📊 Unique documents with judgments: {unique_docs}")
    
    # Judgments per query statistics
    judgments_per_query = df.groupby('query_id').size()
    print(f"\n📋 Judgments per query:")
    print(f"   Mean: {judgments_per_query.mean():.1f}")
    print(f"   Median: {judgments_per_query.median():.1f}")
    print(f"   Min: {judgments_per_query.min()}")
    print(f"   Max: {judgments_per_query.max()}")
    
    # Relevant documents per query
    relevant_df = df[df['relevance'] > 0]
    if len(relevant_df) > 0:
        relevant_per_query = relevant_df.groupby('query_id').size()
        print(f"\n📋 Relevant documents per query:")
        print(f"   Mean: {relevant_per_query.mean():.1f}")
        print(f"   Median: {relevant_per_query.median():.1f}")
        print(f"   Min: {relevant_per_query.min()}")
        print(f"   Max: {relevant_per_query.max()}")
        print(f"   Queries with relevant docs: {len(relevant_per_query)}")
    
    # Show example qrels
    print(f"\n📝 Example qrels (first 10):")
    print(df.head(10).to_string(index=False))
    
    # Show example query with all its judgments
    example_query = df['query_id'].iloc[0]
    query_qrels = df[df['query_id'] == example_query]
    print(f"\n📝 All judgments for query '{example_query}':")
    print(query_qrels.to_string(index=False))
    
    return df

def inspect_antique_queries():
    """Inspect ANTIQUE test queries"""
    print(f"\n🔍 Inspecting ANTIQUE test queries...")
    
    dataset = ir_datasets.load('antique/test')
    
    queries = []
    for query in dataset.queries_iter():
        queries.append({
            'query_id': query.query_id,
            'text': query.text,
            'length': len(query.text),
            'word_count': len(query.text.split())
        })
    
    df = pd.DataFrame(queries)
    print(f"📊 Total test queries: {len(df)}")
    
    print(f"\n📋 Query length statistics:")
    print(f"   Mean length: {df['length'].mean():.1f} chars")
    print(f"   Mean word count: {df['word_count'].mean():.1f} words")
    print(f"   Min word count: {df['word_count'].min()}")
    print(f"   Max word count: {df['word_count'].max()}")
    
    print(f"\n📝 Example queries (first 5):")
    for _, row in df.head(5).iterrows():
        print(f"   {row['query_id']}: '{row['text']}'")
    
    return df

def check_evaluation_correctness():
    """Check if our evaluation approach is correct"""
    print(f"\n✅ Evaluation Correctness Check:")
    print(f"=" * 50)
    
    # Load data
    dataset = ir_datasets.load('antique/test')
    
    # Count queries and qrels
    queries = list(dataset.queries_iter())
    qrels = list(dataset.qrels_iter())
    
    query_ids_in_queries = set(q.query_id for q in queries)
    query_ids_in_qrels = set(q.query_id for q in qrels)
    
    # Check overlap
    overlap = query_ids_in_queries & query_ids_in_qrels
    queries_only = query_ids_in_queries - query_ids_in_qrels
    qrels_only = query_ids_in_qrels - query_ids_in_queries
    
    print(f"📊 Query ID analysis:")
    print(f"   Queries in test set: {len(query_ids_in_queries)}")
    print(f"   Queries with qrels: {len(query_ids_in_qrels)}")
    print(f"   Overlap: {len(overlap)}")
    print(f"   Queries without qrels: {len(queries_only)}")
    print(f"   Qrels without queries: {len(qrels_only)}")
    
    if len(overlap) > 0:
        print(f"✅ Good: {len(overlap)} queries can be evaluated")
    else:
        print(f"❌ Problem: No overlap between queries and qrels")
    
    # Check if we should clean queries
    print(f"\n🧹 Query cleaning considerations:")
    print(f"   • ANTIQUE queries are natural language questions")
    print(f"   • For proper evaluation, we should test both:")
    print(f"     - Raw queries (as users would type them)")
    print(f"     - Cleaned queries (as system processes them)")
    print(f"   • This tells us if cleaning helps or hurts performance")
    
    # Show MAP calculation example
    print(f"\n📊 MAP Calculation reminder:")
    print(f"   • For each query, calculate AP (Average Precision)")
    print(f"   • AP = (1/R) * Σ(P(k) * rel(k)) where:")
    print(f"     - R = total relevant documents for the query")
    print(f"     - P(k) = precision at rank k")
    print(f"     - rel(k) = 1 if document at rank k is relevant, 0 otherwise")
    print(f"   • MAP = mean of all AP scores across queries")

def main():
    """Main inspection function"""
    print("🎯 ANTIQUE Dataset Inspection")
    print("=" * 50)
    
    # Inspect qrels
    qrels_df = inspect_antique_qrels()
    
    # Inspect queries
    queries_df = inspect_antique_queries()
    
    # Check evaluation correctness
    check_evaluation_correctness()
    
    print(f"\n💡 Key Takeaways:")
    print(f"   ✅ Use ANTIQUE test queries and qrels")
    print(f"   ✅ Test both cleaned and raw query processing")
    print(f"   ✅ Only evaluate queries that have relevance judgments")
    print(f"   ✅ Relevance > 0 means relevant document")
    print(f"   ✅ Calculate AP per query, then average for MAP")

if __name__ == "__main__":
    main()
