
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
