#!/usr/bin/env python3
"""
Label Studio Setup Script for MLOps Drug Repurposing Project

This script sets up Label Studio for annotating drug-target interactions.
"""

import os
import json
import pandas as pd
from pathlib import Path
import logging
import shutil
from datetime import datetime

# Set OpenMP environment variable to avoid conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_label_studio_config():
    """Create Label Studio configuration directory and files."""
    logger.info("Creating Label Studio configuration...")
    
    # Create Label Studio directory structure
    label_studio_dir = Path("label_studio")
    label_studio_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (label_studio_dir / "projects").mkdir(exist_ok=True)
    (label_studio_dir / "exports").mkdir(exist_ok=True)
    (label_studio_dir / "uploads").mkdir(exist_ok=True)
    
    # Create label config for drug-target interaction annotation
    label_config = """
    <View>
      <Header value="Drug-Target Interaction Annotation"/>
      
      <!-- Drug Information -->
      <View style="background-color: #f0f8ff; padding: 10px; margin: 10px 0; border-radius: 5px;">
        <Header value="Drug Information"/>
        <Text name="drug_id" value="$drug_id" style="font-weight: bold;"/>
        <Text name="drug_name" value="$drug_name"/>
        <Text name="smiles" value="$smiles" style="font-family: monospace; font-size: 12px;"/>
        <Text name="molecular_weight" value="Molecular Weight: $molecular_weight"/>
        <Text name="logp" value="LogP: $logp"/>
      </View>
      
      <!-- Target Information -->
      <View style="background-color: #f0fff0; padding: 10px; margin: 10px 0; border-radius: 5px;">
        <Header value="Target Information"/>
        <Text name="target_id" value="$target_id" style="font-weight: bold;"/>
        <Text name="target_name" value="$target_name"/>
        <Text name="uniprot_id" value="UniProt ID: $uniprot_id"/>
        <Text name="organism" value="Organism: $organism"/>
        <Text name="target_class" value="Class: $target_class"/>
      </View>
      
      <!-- Interaction Annotation -->
      <View style="background-color: #fff8f0; padding: 10px; margin: 10px 0; border-radius: 5px;">
        <Header value="Interaction Annotation"/>
        
        <!-- Interaction Type -->
        <Choices name="interaction_type" toName="drug_id" choice="single-radio" showInLine="true">
          <Header value="Interaction Type"/>
          <Choice value="binding" background="#4CAF50"/>
          <Choice value="inhibition" background="#FF9800"/>
          <Choice value="activation" background="#2196F3"/>
          <Choice value="modulation" background="#9C27B0"/>
          <Choice value="unknown" background="#757575"/>
        </Choices>
        
        <!-- Confidence Level -->
        <Rating name="confidence" toName="drug_id" maxRating="5" icon="star" size="medium" defaultValue="3">
          <Header value="Confidence Level (1-5 stars)"/>
        </Rating>
        
        <!-- Binding Affinity -->
        <Number name="binding_affinity" toName="drug_id" min="0" max="15" step="0.1" defaultValue="5.0">
          <Header value="Binding Affinity (pKd/pIC50)"/>
        </Number>
        
        <!-- Evidence Quality -->
        <Choices name="evidence_quality" toName="drug_id" choice="single-radio" showInLine="true">
          <Header value="Evidence Quality"/>
          <Choice value="high" background="#4CAF50"/>
          <Choice value="medium" background="#FF9800"/>
          <Choice value="low" background="#F44336"/>
          <Choice value="predicted" background="#9E9E9E"/>
        </Choices>
        
        <!-- Additional Notes -->
        <TextArea name="notes" toName="drug_id" placeholder="Additional notes about this interaction..." rows="3"/>
        
        <!-- Validation Flags -->
        <Choices name="validation_flags" toName="drug_id" choice="multiple" showInLine="true">
          <Header value="Validation Flags"/>
          <Choice value="reviewed" background="#4CAF50"/>
          <Choice value="needs_verification" background="#FF9800"/>
          <Choice value="conflicting_data" background="#F44336"/>
          <Choice value="novel_interaction" background="#2196F3"/>
        </Choices>
      </View>
    </View>
    """
    
    # Save label config
    config_file = label_studio_dir / "drug_target_label_config.xml"
    with open(config_file, 'w') as f:
        f.write(label_config.strip())
    
    logger.info(f"Label configuration saved to: {config_file}")
    
    return label_studio_dir, config_file

def prepare_annotation_data():
    """Prepare data for annotation by combining drug and target information."""
    logger.info("Preparing annotation data...")
    
    try:
        # Load existing data
        data_dir = Path("data/raw/sample")
        
        # Load drug-target interactions
        interactions_df = pd.read_csv(data_dir / "drug_target_interactions.csv")
        drug_metadata_df = pd.read_csv(data_dir / "drug_metadata.csv")
        target_metadata_df = pd.read_csv(data_dir / "target_metadata.csv")
        
        # Merge data for comprehensive annotation view
        annotation_data = interactions_df.merge(
            drug_metadata_df, on='drug_id', how='left'
        ).merge(
            target_metadata_df, on='target_id', how='left'
        )
        
        # Add annotation fields with defaults
        annotation_data['annotation_date'] = datetime.now().strftime('%Y-%m-%d')
        annotation_data['annotator'] = 'system'
        annotation_data['annotation_status'] = 'pending'
        
        # Create annotation dataset
        annotation_file = Path("label_studio/annotation_dataset.json")
        
        # Convert to Label Studio format
        label_studio_data = []
        for idx, row in annotation_data.iterrows():
            task = {
                "id": idx + 1,
                "data": {
                    "drug_id": row['drug_id'],
                    "drug_name": row['drug_name'],
                    "smiles": row['smiles'],
                    "molecular_weight": row.get('molecular_weight', 'N/A'),
                    "logp": row.get('logp', 'N/A'),
                    "target_id": row['target_id'],
                    "target_name": row['target_name'],
                    "uniprot_id": row['uniprot_id'],
                    "organism": row.get('organism', 'N/A'),
                    "target_class": row.get('target_class', 'N/A'),
                    "existing_interaction_type": row['interaction_type'],
                    "existing_binding_affinity": row.get('binding_affinity', 'N/A'),
                    "existing_confidence": row.get('confidence_score', 'N/A')
                },
                "meta": {
                    "original_source": "sample_data",
                    "preparation_date": datetime.now().isoformat()
                }
            }
            label_studio_data.append(task)
        
        # Save annotation dataset
        with open(annotation_file, 'w') as f:
            json.dump(label_studio_data, f, indent=2)
        
        logger.info(f"Annotation dataset created: {annotation_file}")
        logger.info(f"Total annotation tasks: {len(label_studio_data)}")
        
        return annotation_file, len(label_studio_data)
        
    except Exception as e:
        logger.error(f"Failed to prepare annotation data: {e}")
        return None, 0

def create_annotation_guidelines():
    """Create comprehensive annotation guidelines."""
    logger.info("Creating annotation guidelines...")
    
    guidelines = """
# Drug-Target Interaction Annotation Guidelines

## Overview
This guide provides instructions for annotating drug-target interactions in the MLOps Drug Repurposing project. The goal is to create high-quality, consistent annotations that can be used for training machine learning models.

## Annotation Interface Components

### 1. Drug Information Section
- **Drug ID**: Unique identifier for the drug compound
- **Drug Name**: Common or trade name of the drug
- **SMILES**: Chemical structure representation in SMILES format
- **Molecular Weight**: Molecular weight in Daltons
- **LogP**: Lipophilicity measure (octanol-water partition coefficient)

### 2. Target Information Section
- **Target ID**: Unique identifier for the protein target
- **Target Name**: Name of the protein target
- **UniProt ID**: UniProt database identifier
- **Organism**: Source organism (usually Homo sapiens)
- **Target Class**: Functional classification (e.g., enzyme, receptor, transporter)

### 3. Interaction Annotation Section

#### Interaction Type (Required)
Select the primary type of interaction:

- **Binding**: Drug physically binds to the target without necessarily modulating activity
- **Inhibition**: Drug decreases target activity or function
- **Activation**: Drug increases target activity or function
- **Modulation**: Drug alters target activity in a complex manner (allosteric effects)
- **Unknown**: Interaction is confirmed but mechanism is unclear

#### Confidence Level (Required)
Rate your confidence in the annotation (1-5 stars):
- **5 stars**: Very high confidence, multiple reliable sources
- **4 stars**: High confidence, well-documented interaction
- **3 stars**: Medium confidence, some supporting evidence
- **2 stars**: Low confidence, limited evidence
- **1 star**: Very low confidence, speculative or conflicting data

#### Binding Affinity (Optional)
Enter the binding affinity value if known:
- Use pKd, pIC50, or pEC50 values (typically 0-15 range)
- If multiple values are available, use the most reliable one
- Leave blank if no quantitative data is available

#### Evidence Quality (Required)
Assess the quality of underlying evidence:
- **High**: Peer-reviewed publications, clinical trials, validated assays
- **Medium**: Preliminary studies, cell-based assays, computational predictions with validation
- **Low**: Single studies, limited validation, older literature
- **Predicted**: Computational predictions without experimental validation

#### Additional Notes (Optional)
Provide any relevant additional information:
- Experimental conditions
- Contradictory evidence
- Specificity issues
- Dosage or concentration considerations

#### Validation Flags (Optional)
Select applicable flags:
- **Reviewed**: Annotation has been quality-checked
- **Needs Verification**: Requires additional validation
- **Conflicting Data**: Multiple sources show conflicting results
- **Novel Interaction**: Previously unreported interaction

## Quality Standards

### Consistency Requirements
1. **Terminology**: Use consistent terminology across all annotations
2. **Evidence Hierarchy**: Prioritize experimental data over computational predictions
3. **Source Reliability**: Prefer peer-reviewed literature over preprints or databases
4. **Completeness**: Fill all required fields; mark uncertain data appropriately

### Common Mistakes to Avoid
1. **Confusing binding with functional effects**: Binding doesn't always equal functional modulation
2. **Over-interpreting weak evidence**: Be conservative with low-confidence data
3. **Ignoring dosage effects**: Consider concentration-dependent effects
4. **Missing context**: Consider experimental conditions and model systems

### Data Validation Steps
1. **Cross-reference**: Check multiple sources when possible
2. **Unit consistency**: Ensure binding affinity units are consistent
3. **Biological plausibility**: Consider if the interaction makes biological sense
4. **Literature verification**: Verify claims against original sources

## Annotation Workflow

### Step 1: Review Information
- Read all provided drug and target information
- Familiarize yourself with the compounds and proteins involved

### Step 2: Research (if needed)
- Consult additional literature if information is incomplete
- Use reliable databases (ChEMBL, DrugBank, UniProt)

### Step 3: Annotate
- Fill required fields first
- Add optional information when available
- Use notes field for important context

### Step 4: Quality Check
- Review your annotation for consistency
- Ensure all required fields are completed
- Verify numerical values are reasonable

### Step 5: Submit
- Save your annotation
- Mark as complete only when confident in the quality

## Resources

### Recommended Databases
- **ChEMBL**: Bioactivity database (https://www.ebi.ac.uk/chembl/)
- **DrugBank**: Drug information database (https://go.drugbank.com/)
- **UniProt**: Protein sequence and functional information (https://www.uniprot.org/)
- **PubChem**: Chemical information (https://pubchem.ncbi.nlm.nih.gov/)

### Literature Sources
- PubMed for peer-reviewed articles
- Google Scholar for broader literature search
- Specialized pharmacology journals

## Contact Information
For questions about annotation procedures or technical issues:
- Technical Support: MLOps Team
- Scientific Questions: Domain Experts
- Data Quality Issues: Quality Assurance Team

---

**Version**: 1.0  
**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}  
**Reviewed By**: MLOps Team
"""
    
    guidelines_file = Path("label_studio/annotation_guidelines.md")
    with open(guidelines_file, 'w') as f:
        f.write(guidelines)
    
    logger.info(f"Annotation guidelines created: {guidelines_file}")
    
    return guidelines_file

def create_label_studio_startup_script():
    """Create a startup script for Label Studio."""
    logger.info("Creating Label Studio startup script...")
    
    startup_script = """#!/bin/bash
# Label Studio Startup Script for MLOps Drug Repurposing Project

echo "🏷️  Starting Label Studio for Drug-Target Interaction Annotation"

# Set environment variables
export LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true
export LABEL_STUDIO_USERNAME=${LABEL_STUDIO_USERNAME:-admin}
export LABEL_STUDIO_PASSWORD=${LABEL_STUDIO_PASSWORD:-password123}
export LABEL_STUDIO_HOST=${LABEL_STUDIO_HOST:-0.0.0.0}
export LABEL_STUDIO_PORT=${LABEL_STUDIO_PORT:-8080}

# Create data directory if it doesn't exist
mkdir -p label_studio/data

# Initialize Label Studio
echo "🚀 Initializing Label Studio..."
label-studio init drug_repurposing_annotation \\
    --username $LABEL_STUDIO_USERNAME \\
    --password $LABEL_STUDIO_PASSWORD \\
    --host $LABEL_STUDIO_HOST \\
    --port $LABEL_STUDIO_PORT

echo "✅ Label Studio initialized successfully!"
echo "📊 Access the annotation interface at: http://localhost:$LABEL_STUDIO_PORT"
echo "👤 Username: $LABEL_STUDIO_USERNAME"
echo "🔑 Password: $LABEL_STUDIO_PASSWORD"
echo ""
echo "📚 Next steps:"
echo "1. Import the annotation dataset: label_studio/annotation_dataset.json"
echo "2. Use the label config from: label_studio/drug_target_label_config.xml"
echo "3. Follow the annotation guidelines in: label_studio/annotation_guidelines.md"
"""
    
    script_file = Path("label_studio/start_label_studio.sh")
    with open(script_file, 'w') as f:
        f.write(startup_script)
    
    # Make script executable
    script_file.chmod(0o755)
    
    logger.info(f"Startup script created: {script_file}")
    
    return script_file

def create_project_import_script():
    """Create a script to import the project into Label Studio."""
    logger.info("Creating project import script...")
    
    import_script = '''#!/usr/bin/env python3
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
'''
    
    script_file = Path("label_studio/import_project.py")
    with open(script_file, 'w') as f:
        f.write(import_script)
    
    script_file.chmod(0o755)
    
    logger.info(f"Project import script created: {script_file}")
    
    return script_file

def main():
    """Main setup function for Label Studio."""
    logger.info("Setting up Label Studio for drug-target interaction annotation...")
    
    try:
        # Create Label Studio configuration
        label_studio_dir, config_file = create_label_studio_config()
        
        # Prepare annotation data
        annotation_file, task_count = prepare_annotation_data()
        
        # Create annotation guidelines
        guidelines_file = create_annotation_guidelines()
        
        # Create startup script
        startup_script = create_label_studio_startup_script()
        
        # Create import script
        import_script = create_project_import_script()
        
        # Create summary report
        setup_summary = {
            "setup_timestamp": datetime.now().isoformat(),
            "label_studio_directory": str(label_studio_dir),
            "configuration_file": str(config_file),
            "annotation_dataset": str(annotation_file) if annotation_file else None,
            "task_count": task_count,
            "guidelines_file": str(guidelines_file),
            "startup_script": str(startup_script),
            "import_script": str(import_script),
            "setup_status": "completed"
        }
        
        summary_file = label_studio_dir / "setup_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(setup_summary, f, indent=2)
        
        print("\n🏷️  Label Studio Setup Complete!")
        print("=" * 50)
        print(f"📁 Setup Directory: {label_studio_dir}")
        print(f"⚙️  Configuration: {config_file}")
        print(f"📊 Annotation Tasks: {task_count}")
        print(f"📚 Guidelines: {guidelines_file}")
        print(f"🚀 Startup Script: {startup_script}")
        print(f"📥 Import Script: {import_script}")
        print()
        print("🚀 Next Steps:")
        print("1. Run the startup script: ./label_studio/start_label_studio.sh")
        print("2. Open http://localhost:8080 in your browser")
        print("3. Create a project and import the configuration")
        print("4. Import the annotation dataset")
        print("5. Begin annotation following the guidelines")
        
        return 0
        
    except Exception as e:
        logger.error(f"Label Studio setup failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    exit(main())