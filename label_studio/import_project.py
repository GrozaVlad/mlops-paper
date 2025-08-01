#!/usr/bin/env python3
"""
Label Studio Project Import Script for Drug-Target Interactions
"""

import json
import requests
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def import_project_to_label_studio():
    """Import the drug-target annotation project to Label Studio."""
    
    # Label Studio API configuration
    LABEL_STUDIO_URL = "http://localhost:8080"
    API_TOKEN = "your_api_token_here"  # Get this from Label Studio UI
    
    # Load project configuration
    with open("label_studio/drug_target_label_config.xml", "r") as f:
        label_config = f.read()
    
    with open("label_studio/annotation_dataset.json", "r") as f:
        tasks_data = json.load(f)
    
    # Create project
    project_data = {
        "title": "Drug-Target Interaction Annotation",
        "description": "Annotate drug-target interactions for ML model training",
        "label_config": label_config,
        "expert_instruction": "Follow the annotation guidelines for consistent labeling",
        "show_instruction": True,
        "show_skip_button": True,
        "enable_empty_annotation": False
    }
    
    headers = {
        "Authorization": f"Token {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    try:
        # Create project
        response = requests.post(
            f"{LABEL_STUDIO_URL}/api/projects/",
            headers=headers,
            json=project_data
        )
        
        if response.status_code == 201:
            project = response.json()
            project_id = project["id"]
            logger.info(f"Project created successfully with ID: {project_id}")
            
            # Import tasks
            tasks_response = requests.post(
                f"{LABEL_STUDIO_URL}/api/projects/{project_id}/tasks/bulk/",
                headers=headers,
                json=tasks_data
            )
            
            if tasks_response.status_code == 201:
                logger.info(f"Successfully imported {len(tasks_data)} tasks")
                print(f"✅ Project setup complete!")
                print(f"🌐 Access your project at: {LABEL_STUDIO_URL}/projects/{project_id}")
            else:
                logger.error(f"Failed to import tasks: {tasks_response.text}")
        else:
            logger.error(f"Failed to create project: {response.text}")
            
    except Exception as e:
        logger.error(f"Error importing project: {e}")
        print("⚠️  Manual import required:")
        print("1. Open Label Studio in your browser")
        print("2. Create a new project")
        print("3. Import label_studio/drug_target_label_config.xml as labeling config")
        print("4. Import label_studio/annotation_dataset.json as tasks")

if __name__ == "__main__":
    import_project_to_label_studio()
