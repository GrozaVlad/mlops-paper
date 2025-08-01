#!/usr/bin/env python3
"""
Model Lineage Tracking System

Tracks model ancestry, relationships, and dependencies throughout the model lifecycle.
Provides comprehensive audit trails and visualization of model evolution.
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run, Experiment
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import sqlite3
from contextlib import contextmanager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LineageRelationType(Enum):
    """Types of relationships between models"""
    DERIVED_FROM = "derived_from"
    FINE_TUNED_FROM = "fine_tuned_from"
    ENSEMBLE_MEMBER = "ensemble_member"
    RETRAINED_FROM = "retrained_from"
    EXPERIMENT_VARIANT = "experiment_variant"
    A_B_TEST_VARIANT = "a_b_test_variant"
    ROLLBACK_FROM = "rollback_from"
    MERGED_FROM = "merged_from"


class DataLineageType(Enum):
    """Types of data lineage relationships"""
    TRAINED_ON = "trained_on"
    VALIDATED_ON = "validated_on"
    TESTED_ON = "tested_on"
    AUGMENTED_FROM = "augmented_from"
    SUBSET_OF = "subset_of"
    TRANSFORMED_FROM = "transformed_from"


@dataclass
class ModelNode:
    """Represents a model in the lineage graph"""
    model_id: str
    model_name: str
    version: str
    mlflow_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    stage: str = "None"
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.model_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelNode':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class DataNode:
    """Represents a dataset in the lineage graph"""
    data_id: str
    name: str
    version: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    size_bytes: Optional[int] = None
    num_samples: Optional[int] = None
    features: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    checksum: Optional[str] = None
    
    def __hash__(self):
        return hash(self.data_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataNode':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class LineageEdge:
    """Represents a relationship in the lineage graph"""
    source_id: str
    target_id: str
    relation_type: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relation_type': self.relation_type,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


class ModelLineageTracker:
    """Comprehensive model lineage tracking system"""
    
    def __init__(self, 
                 db_path: str = "model_lineage.db",
                 mlflow_tracking_uri: Optional[str] = None):
        """
        Initialize lineage tracker
        
        Args:
            db_path: Path to SQLite database for lineage storage
            mlflow_tracking_uri: MLflow tracking URI
        """
        self.db_path = Path(db_path)
        self.mlflow_client = MlflowClient(mlflow_tracking_uri) if mlflow_tracking_uri else None
        
        # Initialize graphs
        self.model_graph = nx.DiGraph()
        self.data_graph = nx.DiGraph()
        self.model_data_graph = nx.DiGraph()
        
        # Initialize database
        self._init_database()
        
        # Load existing lineage
        self._load_lineage_from_db()
        
    def _init_database(self):
        """Initialize SQLite database for lineage storage"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    mlflow_run_id TEXT,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    metrics TEXT,
                    parameters TEXT,
                    tags TEXT,
                    artifacts TEXT,
                    environment TEXT
                )
            """)
            
            # Datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    data_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    size_bytes INTEGER,
                    num_samples INTEGER,
                    features TEXT,
                    statistics TEXT,
                    location TEXT,
                    checksum TEXT
                )
            """)
            
            # Model relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (source_id) REFERENCES models(model_id),
                    FOREIGN KEY (target_id) REFERENCES models(model_id)
                )
            """)
            
            # Model-data relationships table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_data_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    data_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (model_id) REFERENCES models(model_id),
                    FOREIGN KEY (data_id) REFERENCES datasets(data_id)
                )
            """)
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection context manager"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()
    
    def _load_lineage_from_db(self):
        """Load existing lineage from database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Load models
            cursor.execute("SELECT * FROM models")
            for row in cursor.fetchall():
                model_data = {
                    'model_id': row[0],
                    'model_name': row[1],
                    'version': row[2],
                    'mlflow_run_id': row[3],
                    'created_at': row[4],
                    'created_by': row[5],
                    'stage': row[6],
                    'metrics': json.loads(row[7]) if row[7] else {},
                    'parameters': json.loads(row[8]) if row[8] else {},
                    'tags': json.loads(row[9]) if row[9] else {},
                    'artifacts': json.loads(row[10]) if row[10] else [],
                    'environment': json.loads(row[11]) if row[11] else {}
                }
                node = ModelNode.from_dict(model_data)
                self.model_graph.add_node(node.model_id, data=node)
            
            # Load datasets
            cursor.execute("SELECT * FROM datasets")
            for row in cursor.fetchall():
                data_data = {
                    'data_id': row[0],
                    'name': row[1],
                    'version': row[2],
                    'created_at': row[3],
                    'size_bytes': row[4],
                    'num_samples': row[5],
                    'features': json.loads(row[6]) if row[6] else [],
                    'statistics': json.loads(row[7]) if row[7] else {},
                    'location': row[8],
                    'checksum': row[9]
                }
                node = DataNode.from_dict(data_data)
                self.data_graph.add_node(node.data_id, data=node)
            
            # Load model relationships
            cursor.execute("SELECT * FROM model_relationships")
            for row in cursor.fetchall():
                self.model_graph.add_edge(
                    row[1], row[2],
                    relation_type=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {}
                )
            
            # Load model-data relationships
            cursor.execute("SELECT * FROM model_data_relationships")
            for row in cursor.fetchall():
                self.model_data_graph.add_edge(
                    row[1], row[2],
                    relation_type=row[3],
                    created_at=datetime.fromisoformat(row[4]),
                    metadata=json.loads(row[5]) if row[5] else {}
                )
    
    def add_model(self, model: ModelNode) -> None:
        """Add a model to the lineage graph"""
        self.model_graph.add_node(model.model_id, data=model)
        
        # Save to database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO models 
                (model_id, model_name, version, mlflow_run_id, created_at, 
                 created_by, stage, metrics, parameters, tags, artifacts, environment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model.model_id,
                model.model_name,
                model.version,
                model.mlflow_run_id,
                model.created_at,
                model.created_by,
                model.stage,
                json.dumps(model.metrics),
                json.dumps(model.parameters),
                json.dumps(model.tags),
                json.dumps(model.artifacts),
                json.dumps(model.environment)
            ))
            conn.commit()
        
        logger.info(f"Added model {model.model_id} to lineage")
    
    def add_dataset(self, dataset: DataNode) -> None:
        """Add a dataset to the lineage graph"""
        self.data_graph.add_node(dataset.data_id, data=dataset)
        
        # Save to database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO datasets 
                (data_id, name, version, created_at, size_bytes, num_samples,
                 features, statistics, location, checksum)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset.data_id,
                dataset.name,
                dataset.version,
                dataset.created_at,
                dataset.size_bytes,
                dataset.num_samples,
                json.dumps(dataset.features),
                json.dumps(dataset.statistics),
                dataset.location,
                dataset.checksum
            ))
            conn.commit()
        
        logger.info(f"Added dataset {dataset.data_id} to lineage")
    
    def add_model_relationship(self,
                             source_model_id: str,
                             target_model_id: str,
                             relation_type: LineageRelationType,
                             metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a relationship between models"""
        if source_model_id not in self.model_graph:
            raise ValueError(f"Source model {source_model_id} not found")
        if target_model_id not in self.model_graph:
            raise ValueError(f"Target model {target_model_id} not found")
        
        edge = LineageEdge(
            source_id=source_model_id,
            target_id=target_model_id,
            relation_type=relation_type.value,
            metadata=metadata or {}
        )
        
        self.model_graph.add_edge(
            source_model_id,
            target_model_id,
            relation_type=relation_type.value,
            created_at=edge.created_at,
            metadata=edge.metadata
        )
        
        # Save to database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_relationships 
                (source_id, target_id, relation_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                source_model_id,
                target_model_id,
                relation_type.value,
                edge.created_at,
                json.dumps(edge.metadata)
            ))
            conn.commit()
        
        logger.info(f"Added relationship {relation_type.value} between {source_model_id} and {target_model_id}")
    
    def add_model_data_relationship(self,
                                  model_id: str,
                                  data_id: str,
                                  relation_type: DataLineageType,
                                  metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a relationship between model and dataset"""
        if model_id not in self.model_graph:
            raise ValueError(f"Model {model_id} not found")
        if data_id not in self.data_graph:
            raise ValueError(f"Dataset {data_id} not found")
        
        edge = LineageEdge(
            source_id=model_id,
            target_id=data_id,
            relation_type=relation_type.value,
            metadata=metadata or {}
        )
        
        self.model_data_graph.add_edge(
            model_id,
            data_id,
            relation_type=relation_type.value,
            created_at=edge.created_at,
            metadata=edge.metadata
        )
        
        # Save to database
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_data_relationships 
                (model_id, data_id, relation_type, created_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_id,
                data_id,
                relation_type.value,
                edge.created_at,
                json.dumps(edge.metadata)
            ))
            conn.commit()
        
        logger.info(f"Added data relationship {relation_type.value} between model {model_id} and dataset {data_id}")
    
    def track_mlflow_model(self, run_id: str, model_name: str, version: str) -> ModelNode:
        """Track a model from MLflow"""
        if not self.mlflow_client:
            raise ValueError("MLflow client not configured")
        
        # Get run information
        run = self.mlflow_client.get_run(run_id)
        
        # Create model node
        model = ModelNode(
            model_id=f"{model_name}_v{version}_{run_id[:8]}",
            model_name=model_name,
            version=version,
            mlflow_run_id=run_id,
            created_at=datetime.fromtimestamp(run.info.start_time / 1000),
            created_by=run.data.tags.get("mlflow.user", "unknown"),
            stage="None",
            metrics=run.data.metrics,
            parameters=run.data.params,
            tags=run.data.tags,
            artifacts=[artifact.path for artifact in self.mlflow_client.list_artifacts(run_id)],
            environment={
                "mlflow.source.name": run.data.tags.get("mlflow.source.name", ""),
                "mlflow.source.type": run.data.tags.get("mlflow.source.type", "")
            }
        )
        
        # Add to lineage
        self.add_model(model)
        
        # Track parent run if exists
        parent_run_id = run.data.tags.get("mlflow.parentRunId")
        if parent_run_id:
            parent_model_id = f"{model_name}_parent_{parent_run_id[:8]}"
            self.add_model_relationship(
                parent_model_id,
                model.model_id,
                LineageRelationType.DERIVED_FROM,
                metadata={"parent_run_id": parent_run_id}
            )
        
        return model
    
    def get_model_ancestry(self, model_id: str, max_depth: Optional[int] = None) -> List[str]:
        """Get all ancestors of a model"""
        if model_id not in self.model_graph:
            raise ValueError(f"Model {model_id} not found")
        
        ancestors = []
        visited = set()
        queue = [(model_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if max_depth and depth > max_depth:
                continue
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            if current_id != model_id:
                ancestors.append(current_id)
            
            # Get predecessors (models this model was derived from)
            for pred in self.model_graph.predecessors(current_id):
                queue.append((pred, depth + 1))
        
        return ancestors
    
    def get_model_descendants(self, model_id: str, max_depth: Optional[int] = None) -> List[str]:
        """Get all descendants of a model"""
        if model_id not in self.model_graph:
            raise ValueError(f"Model {model_id} not found")
        
        descendants = []
        visited = set()
        queue = [(model_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if max_depth and depth > max_depth:
                continue
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            if current_id != model_id:
                descendants.append(current_id)
            
            # Get successors (models derived from this model)
            for succ in self.model_graph.successors(current_id):
                queue.append((succ, depth + 1))
        
        return descendants
    
    def get_model_data_lineage(self, model_id: str) -> Dict[str, List[str]]:
        """Get all datasets used by a model"""
        if model_id not in self.model_graph:
            raise ValueError(f"Model {model_id} not found")
        
        data_lineage = defaultdict(list)
        
        # Direct data relationships
        for _, data_id, edge_data in self.model_data_graph.edges(model_id, data=True):
            relation_type = edge_data['relation_type']
            data_lineage[relation_type].append(data_id)
        
        # Inherited data relationships from ancestors
        ancestors = self.get_model_ancestry(model_id)
        for ancestor_id in ancestors:
            for _, data_id, edge_data in self.model_data_graph.edges(ancestor_id, data=True):
                relation_type = edge_data['relation_type']
                data_lineage[f"inherited_{relation_type}"].append(data_id)
        
        return dict(data_lineage)
    
    def generate_lineage_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive lineage report for a model"""
        if model_id not in self.model_graph:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.model_graph.nodes[model_id]['data']
        
        report = {
            'model_info': model.to_dict(),
            'ancestry': {
                'immediate_parents': list(self.model_graph.predecessors(model_id)),
                'all_ancestors': self.get_model_ancestry(model_id),
                'immediate_children': list(self.model_graph.successors(model_id)),
                'all_descendants': self.get_model_descendants(model_id)
            },
            'relationships': {},
            'data_lineage': self.get_model_data_lineage(model_id),
            'statistics': {
                'total_ancestors': len(self.get_model_ancestry(model_id)),
                'total_descendants': len(self.get_model_descendants(model_id)),
                'generation': self._get_model_generation(model_id),
                'related_experiments': self._get_related_experiments(model_id)
            }
        }
        
        # Group relationships by type
        for source, target, edge_data in self.model_graph.edges(data=True):
            if source == model_id or target == model_id:
                rel_type = edge_data['relation_type']
                if rel_type not in report['relationships']:
                    report['relationships'][rel_type] = []
                
                report['relationships'][rel_type].append({
                    'source': source,
                    'target': target,
                    'created_at': edge_data['created_at'].isoformat(),
                    'metadata': edge_data.get('metadata', {})
                })
        
        return report
    
    def _get_model_generation(self, model_id: str) -> int:
        """Get the generation number of a model (distance from root)"""
        # Find root models (no predecessors)
        roots = [n for n in self.model_graph.nodes() if self.model_graph.in_degree(n) == 0]
        
        if not roots:
            return 0
        
        # Calculate shortest path from any root
        min_distance = float('inf')
        for root in roots:
            try:
                path = nx.shortest_path(self.model_graph, root, model_id)
                min_distance = min(min_distance, len(path) - 1)
            except nx.NetworkXNoPath:
                continue
        
        return min_distance if min_distance != float('inf') else 0
    
    def _get_related_experiments(self, model_id: str) -> List[str]:
        """Get all experiments related to a model and its lineage"""
        related_models = set([model_id])
        related_models.update(self.get_model_ancestry(model_id))
        related_models.update(self.get_model_descendants(model_id))
        
        experiments = set()
        for mid in related_models:
            if mid in self.model_graph:
                model = self.model_graph.nodes[mid]['data']
                if model.mlflow_run_id and self.mlflow_client:
                    try:
                        run = self.mlflow_client.get_run(model.mlflow_run_id)
                        experiments.add(run.info.experiment_id)
                    except:
                        pass
        
        return list(experiments)
    
    def visualize_model_lineage(self, 
                              model_id: Optional[str] = None,
                              output_file: str = "model_lineage.png",
                              include_data: bool = False,
                              max_depth: Optional[int] = None) -> None:
        """Visualize model lineage graph"""
        plt.figure(figsize=(20, 16))
        
        # Create subgraph based on parameters
        if model_id:
            # Get relevant nodes
            nodes = set([model_id])
            if max_depth:
                nodes.update(self.get_model_ancestry(model_id, max_depth))
                nodes.update(self.get_model_descendants(model_id, max_depth))
            else:
                nodes.update(self.get_model_ancestry(model_id))
                nodes.update(self.get_model_descendants(model_id))
            
            subgraph = self.model_graph.subgraph(nodes)
        else:
            subgraph = self.model_graph
        
        # Layout
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Draw nodes with different colors based on stage
        stage_colors = {
            'None': '#E0E0E0',
            'Staging': '#FFE082',
            'Production': '#81C784',
            'Archived': '#BCAAA4'
        }
        
        for node_id, (x, y) in pos.items():
            model = subgraph.nodes[node_id]['data']
            color = stage_colors.get(model.stage, '#E0E0E0')
            
            # Draw node
            circle = plt.Circle((x, y), 0.08, color=color, ec='black', linewidth=2)
            plt.gca().add_patch(circle)
            
            # Add label
            plt.text(x, y-0.12, f"{model.model_name}\nv{model.version}", 
                    ha='center', va='top', fontsize=8, weight='bold')
            
            # Add metrics if available
            if model.metrics:
                key_metric = list(model.metrics.keys())[0]
                value = model.metrics[key_metric]
                plt.text(x, y-0.16, f"{key_metric}: {value:.3f}", 
                        ha='center', va='top', fontsize=6)
        
        # Draw edges with different styles based on relationship type
        edge_styles = {
            'derived_from': {'style': 'solid', 'color': 'blue', 'width': 2},
            'fine_tuned_from': {'style': 'dashed', 'color': 'green', 'width': 2},
            'retrained_from': {'style': 'dotted', 'color': 'orange', 'width': 2},
            'a_b_test_variant': {'style': 'dashdot', 'color': 'red', 'width': 1.5},
            'ensemble_member': {'style': 'solid', 'color': 'purple', 'width': 1.5}
        }
        
        for source, target, edge_data in subgraph.edges(data=True):
            rel_type = edge_data['relation_type']
            style = edge_styles.get(rel_type, {'style': 'solid', 'color': 'gray', 'width': 1})
            
            x1, y1 = pos[source]
            x2, y2 = pos[target]
            
            plt.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', 
                                      linestyle=style['style'],
                                      color=style['color'],
                                      linewidth=style['width']))
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=color, label=stage) 
            for stage, color in stage_colors.items()
        ]
        plt.legend(handles=legend_elements, title="Model Stage", loc='upper left')
        
        # Add relationship legend
        rel_legend = [
            plt.Line2D([0], [0], color=style['color'], 
                      linestyle=style['style'], 
                      linewidth=style['width'],
                      label=rel_type.replace('_', ' ').title())
            for rel_type, style in edge_styles.items()
        ]
        plt.legend(handles=rel_legend, title="Relationships", loc='upper right')
        
        plt.title(f"Model Lineage Graph" + (f" - Focus: {model_id}" if model_id else ""), 
                 fontsize=16, weight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Lineage visualization saved to {output_file}")
    
    def export_lineage(self, output_file: str, format: str = 'json') -> None:
        """Export complete lineage to file"""
        lineage_data = {
            'models': {},
            'datasets': {},
            'model_relationships': [],
            'data_relationships': [],
            'metadata': {
                'export_date': datetime.utcnow().isoformat(),
                'total_models': len(self.model_graph),
                'total_datasets': len(self.data_graph),
                'total_relationships': len(self.model_graph.edges()) + len(self.model_data_graph.edges())
            }
        }
        
        # Export models
        for node_id in self.model_graph.nodes():
            model = self.model_graph.nodes[node_id]['data']
            lineage_data['models'][node_id] = model.to_dict()
        
        # Export datasets
        for node_id in self.data_graph.nodes():
            dataset = self.data_graph.nodes[node_id]['data']
            lineage_data['datasets'][node_id] = dataset.to_dict()
        
        # Export model relationships
        for source, target, edge_data in self.model_graph.edges(data=True):
            lineage_data['model_relationships'].append({
                'source': source,
                'target': target,
                'relation_type': edge_data['relation_type'],
                'created_at': edge_data['created_at'].isoformat(),
                'metadata': edge_data.get('metadata', {})
            })
        
        # Export data relationships
        for source, target, edge_data in self.model_data_graph.edges(data=True):
            lineage_data['data_relationships'].append({
                'model_id': source,
                'data_id': target,
                'relation_type': edge_data['relation_type'],
                'created_at': edge_data['created_at'].isoformat(),
                'metadata': edge_data.get('metadata', {})
            })
        
        # Save to file
        output_path = Path(output_file)
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(lineage_data, f, indent=2)
        elif format == 'yaml':
            with open(output_path, 'w') as f:
                yaml.dump(lineage_data, f, default_flow_style=False)
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(lineage_data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Lineage exported to {output_file}")
    
    def import_lineage(self, input_file: str, format: str = 'json') -> None:
        """Import lineage from file"""
        input_path = Path(input_file)
        
        # Load data
        if format == 'json':
            with open(input_path, 'r') as f:
                lineage_data = json.load(f)
        elif format == 'yaml':
            with open(input_path, 'r') as f:
                lineage_data = yaml.safe_load(f)
        elif format == 'pickle':
            with open(input_path, 'rb') as f:
                lineage_data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Import models
        for model_id, model_data in lineage_data['models'].items():
            model = ModelNode.from_dict(model_data)
            self.add_model(model)
        
        # Import datasets
        for data_id, data_data in lineage_data['datasets'].items():
            dataset = DataNode.from_dict(data_data)
            self.add_dataset(dataset)
        
        # Import model relationships
        for rel in lineage_data['model_relationships']:
            self.model_graph.add_edge(
                rel['source'],
                rel['target'],
                relation_type=rel['relation_type'],
                created_at=datetime.fromisoformat(rel['created_at']),
                metadata=rel.get('metadata', {})
            )
        
        # Import data relationships
        for rel in lineage_data['data_relationships']:
            self.model_data_graph.add_edge(
                rel['model_id'],
                rel['data_id'],
                relation_type=rel['relation_type'],
                created_at=datetime.fromisoformat(rel['created_at']),
                metadata=rel.get('metadata', {})
            )
        
        logger.info(f"Lineage imported from {input_file}")


def main():
    """Example usage of model lineage tracking"""
    
    # Initialize tracker
    tracker = ModelLineageTracker(
        db_path="model_lineage.db",
        mlflow_tracking_uri="http://localhost:5000"
    )
    
    # Example: Add base model
    base_model = ModelNode(
        model_id="drugban_base_v1.0",
        model_name="DrugBAN",
        version="1.0",
        created_by="research_team",
        stage="Production",
        metrics={"auc": 0.92, "precision": 0.85},
        parameters={"learning_rate": 0.001, "batch_size": 32},
        tags={"framework": "pytorch", "dataset": "biosnap"}
    )
    tracker.add_model(base_model)
    
    # Example: Add training dataset
    train_data = DataNode(
        data_id="biosnap_train_v1",
        name="BIOSNAP Training Set",
        version="1.0",
        num_samples=100000,
        features=["smiles", "target_sequence", "interaction_label"],
        location="s3://mlops-data/biosnap/train/"
    )
    tracker.add_dataset(train_data)
    
    # Link model to training data
    tracker.add_model_data_relationship(
        base_model.model_id,
        train_data.data_id,
        DataLineageType.TRAINED_ON
    )
    
    # Example: Add fine-tuned model
    finetuned_model = ModelNode(
        model_id="drugban_finetuned_v1.1",
        model_name="DrugBAN",
        version="1.1",
        created_by="ml_engineer",
        stage="Staging",
        metrics={"auc": 0.94, "precision": 0.88},
        parameters={"learning_rate": 0.0001, "batch_size": 16}
    )
    tracker.add_model(finetuned_model)
    
    # Add relationship
    tracker.add_model_relationship(
        base_model.model_id,
        finetuned_model.model_id,
        LineageRelationType.FINE_TUNED_FROM,
        metadata={"improvement": "domain_specific_data"}
    )
    
    # Generate report
    report = tracker.generate_lineage_report(finetuned_model.model_id)
    print(json.dumps(report, indent=2, default=str))
    
    # Visualize lineage
    tracker.visualize_model_lineage(
        model_id=finetuned_model.model_id,
        output_file="model_lineage_example.png"
    )
    
    # Export lineage
    tracker.export_lineage("model_lineage_export.json")


if __name__ == "__main__":
    main()