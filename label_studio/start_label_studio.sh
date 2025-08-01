#!/bin/bash
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
label-studio init drug_repurposing_annotation \
    --username $LABEL_STUDIO_USERNAME \
    --password $LABEL_STUDIO_PASSWORD \
    --host $LABEL_STUDIO_HOST \
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
