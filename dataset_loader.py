#!/usr/bin/env python3
"""
Proper IR Dataset Loader for Academic Search Engine
Loads datasets with >200K documents and includes test queries with relevance judgments
"""

import ir_datasets
import asyncio
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class IRDatasetLoader:
    """Loads proper IR datasets with documents, queries, and relevance judgments"""
    
    def __init__(self):
        # Datasets with >200K documents and test data
        self.available_datasets = {
            'msmarco-passage': {
                'name': 'MS MARCO Passage Ranking',
                'docs': 'msmarco-passage',
                'queries': 'msmarco-passage/dev/small',
                'qrels': 'msmarco-passage/dev/small',
                'description': '8.8M passages from web pages',
                'doc_count': 8841823
            },
            'trec-covid': {
                'name': 'TREC-COVID',
                'docs': 'cord19/trec-covid',
                'queries': 'cord19/trec-covid',
                'qrels': 'cord19/trec-covid',
                'description': 'COVID-19 research papers',
                'doc_count': 171332
            },
            'robust04': {
                'name': 'TREC Robust 2004',
                'docs': 'trec-robust04',
                'queries': 'trec-robust04',
                'qrels': 'trec-robust04',
                'description': 'News articles collection',
                'doc_count': 528155
            },
            'clueweb09b': {
                'name': 'ClueWeb09 Category B',
                'docs': 'clueweb09b',
                'queries': 'clueweb09b/trec-web-2009',
                'qrels': 'clueweb09b/trec-web-2009',
                'description': 'Web crawl data',
                'doc_count': 50220423
            }
        }
    
    def list_available_datasets(self) -> Dict:
        """List all available datasets with their metadata"""
        return {
            dataset_id: {
                'name': info['name'],
                'description': info['description'],
                'document_count': info['doc_count'],
                'has_queries': True,
                'has_relevance_judgments': True
            }
            for dataset_id, info in self.available_datasets.items()
        }
    
    async def load_dataset(self, dataset_id: str, limit_docs: Optional[int] = None, limit_queries: int = 100) -> Dict:
        """
        Load a complete IR dataset with documents, queries, and relevance judgments
        
        Args:
            dataset_id: Dataset identifier
            limit_docs: Limit number of documents to load (None for all)
            limit_queries: Limit number of queries to load
            
        Returns:
            Dictionary with documents, queries, and relevance judgments
        """
        if dataset_id not in self.available_datasets:
            raise ValueError(f"Dataset {dataset_id} not available. Choose from: {list(self.available_datasets.keys())}")
        
        dataset_info = self.available_datasets[dataset_id]
        logger.info(f"Loading {dataset_info['name']} dataset...")
        
        try:
            # Load documents
            logger.info("Loading documents...")
            docs_dataset = ir_datasets.load(dataset_info['docs'])
            documents = []
            
            for i, doc in enumerate(docs_dataset.docs_iter()):
                if limit_docs and i >= limit_docs:
                    break
                    
                documents.append({
                    'doc_id': doc.doc_id,
                    'title': getattr(doc, 'title', ''),
                    'content': getattr(doc, 'text', getattr(doc, 'body', '')),
                    'url': getattr(doc, 'url', ''),
                    'metadata': {
                        'dataset': dataset_id,
                        'source': dataset_info['name']
                    }
                })
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"Loaded {i + 1:,} documents...")
            
            logger.info(f"✅ Loaded {len(documents):,} documents")
            
            # Load queries
            logger.info("Loading queries...")
            queries_dataset = ir_datasets.load(dataset_info['queries'])
            queries = []
            
            for i, query in enumerate(queries_dataset.queries_iter()):
                if i >= limit_queries:
                    break
                    
                queries.append({
                    'query_id': query.query_id,
                    'text': query.text,
                    'title': getattr(query, 'title', ''),
                    'description': getattr(query, 'description', ''),
                    'narrative': getattr(query, 'narrative', '')
                })
            
            logger.info(f"✅ Loaded {len(queries)} queries")
            
            # Load relevance judgments (qrels)
            logger.info("Loading relevance judgments...")
            qrels_dataset = ir_datasets.load(dataset_info['qrels'])
            qrels = {}
            
            for qrel in qrels_dataset.qrels_iter():
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = {}
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
            
            logger.info(f"✅ Loaded relevance judgments for {len(qrels)} queries")
            
            return {
                'dataset_id': dataset_id,
                'dataset_name': dataset_info['name'],
                'documents': documents,
                'queries': queries,
                'qrels': qrels,
                'stats': {
                    'total_documents': len(documents),
                    'total_queries': len(queries),
                    'queries_with_judgments': len(qrels)
                }
            }
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            raise
    
    async def save_dataset(self, dataset_data: Dict, output_dir: str):
        """Save dataset to disk for future use"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_id = dataset_data['dataset_id']
        
        # Save documents
        docs_file = output_path / f"{dataset_id}_documents.jsonl"
        with open(docs_file, 'w') as f:
            for doc in dataset_data['documents']:
                f.write(json.dumps(doc) + '\n')
        
        # Save queries
        queries_file = output_path / f"{dataset_id}_queries.json"
        with open(queries_file, 'w') as f:
            json.dump(dataset_data['queries'], f, indent=2)
        
        # Save relevance judgments
        qrels_file = output_path / f"{dataset_id}_qrels.json"
        with open(qrels_file, 'w') as f:
            json.dump(dataset_data['qrels'], f, indent=2)
        
        # Save metadata
        metadata_file = output_path / f"{dataset_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'dataset_id': dataset_data['dataset_id'],
                'dataset_name': dataset_data['dataset_name'],
                'stats': dataset_data['stats']
            }, f, indent=2)
        
        logger.info(f"✅ Dataset saved to {output_path}")

# Example usage functions
async def load_two_datasets_for_project():
    """Load two datasets as required by the project"""
    loader = IRDatasetLoader()
    
    # Dataset 1: MS MARCO (large web passages)
    dataset1 = await loader.load_dataset('msmarco-passage', limit_docs=250000, limit_queries=100)
    
    # Dataset 2: TREC Robust 2004 (news articles)
    dataset2 = await loader.load_dataset('robust04', limit_docs=200000, limit_queries=100)
    
    return dataset1, dataset2

async def main():
    """Demo function"""
    loader = IRDatasetLoader()
    
    print("Available IR Datasets:")
    datasets = loader.list_available_datasets()
    for dataset_id, info in datasets.items():
        print(f"- {dataset_id}: {info['name']} ({info['document_count']:,} docs)")
    
    # Load a small sample
    print("\nLoading sample dataset...")
    sample = await loader.load_dataset('msmarco-passage', limit_docs=1000, limit_queries=5)
    print(f"Sample loaded: {sample['stats']}")
    
    # Save to disk
    await loader.save_dataset(sample, 'data/datasets')

if __name__ == "__main__":
    asyncio.run(main())
