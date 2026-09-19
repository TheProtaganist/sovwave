#!/bin/bash

# Backup script for the entire project
# Creates numbered backups with automatic rotation (max 3 backups)
# Naming format: {project_name}_XXX_Stable.tar.gz or {project_name}_XXX_Development.tar.gz
# Usage: ./backup.sh [--stable|--development]
# Default: --stable

# Define backup directory
BACKUP_DIR="Backups"
MAX_BACKUPS=3

# Parse command line arguments for stability flag
STABILITY="Stable"  # Default to Stable

if [ -n "$1" ]; then
    if [ "$1" == "--development" ] || [ "$1" == "-d" ]; then
        STABILITY="Development"
    elif [ "$1" == "--stable" ] || [ "$1" == "-s" ]; then
        STABILITY="Stable"
    elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        echo "📦 Backup Script - Usage Guide"
        echo ""
        echo "Usage: ./backup.sh [OPTION]"
        echo ""
        echo "Options:"
        echo "  --stable, -s        Create a Stable backup (default)"
        echo "  --development, -d   Create a Development backup"
        echo "  --help, -h          Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./backup.sh                    # Creates Stable backup"
        echo "  ./backup.sh --stable           # Creates Stable backup"
        echo "  ./backup.sh --development      # Creates Development backup"
        echo ""
        echo "Backup naming format:"
        echo "  {ProjectName}_XXX_Stable.tar.gz"
        echo "  {ProjectName}_XXX_Development.tar.gz"
        echo ""
        echo "Note: Each stability level maintains its own backup rotation (max 3 backups each)"
        exit 0
    else
        echo "❌ Error: Invalid option '$1'"
        echo ""
        echo "Valid options:"
        echo "  --stable, -s        Create a Stable backup (default)"
        echo "  --development, -d   Create a Development backup"
        echo "  --help, -h          Show help message"
        echo ""
        echo "Examples:"
        echo "  ./backup.sh                    # Creates Stable backup"
        echo "  ./backup.sh --stable           # Creates Stable backup"
        echo "  ./backup.sh --development      # Creates Development backup"
        echo ""
        echo "Run './backup.sh --help' for more information"
        exit 1
    fi
fi

# Create Backups directory if it doesn't exist
mkdir -p "${BACKUP_DIR}"

# Get the project root directory name (used as project_name)
PROJECT_ROOT=$(basename "$(pwd)")
PROJECT_NAME="${PROJECT_ROOT}"

# Find the highest existing backup number for this stability level
HIGHEST_NUM=0
for backup_file in "${BACKUP_DIR}/${PROJECT_NAME}_"*"_${STABILITY}.tar.gz"; do
    if [ -f "$backup_file" ]; then
        # Extract the number from the filename
        NUM=$(basename "$backup_file" | sed "s/${PROJECT_NAME}_\([0-9]\{3\}\)_${STABILITY}\.tar\.gz/\1/")
        if [ "$NUM" -gt "$HIGHEST_NUM" ]; then
            HIGHEST_NUM=$NUM
        fi
    fi
done

# Calculate next number (wrap around after 999)
NEXT_NUM=$((HIGHEST_NUM + 1))
if [ $NEXT_NUM -gt 999 ]; then
    NEXT_NUM=1
fi

BACKUP_NUM=$(printf "%03d" $NEXT_NUM)
BACKUP_NAME="${PROJECT_NAME}_${BACKUP_NUM}_${STABILITY}.tar.gz"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

# Create the backup (exclude Backups folder itself to avoid recursion)
echo "Creating backup: ${BACKUP_NAME}"
tar -czf "${BACKUP_PATH}" \
    --exclude="${BACKUP_DIR}" \
    --exclude=".git" \
    --exclude="node_modules" \
    --exclude="build" \
    --exclude="*.o" \
    --exclude="*.so" \
    --exclude="*.a" \
    -C .. "${PROJECT_ROOT}"

# Check if backup was successful
if [ $? -eq 0 ]; then
    echo "✅ Backup created successfully: ${BACKUP_PATH}"
    echo "📦 Backup size: $(du -h "${BACKUP_PATH}" | cut -f1)"
    echo "🏷️  Stability: ${STABILITY}"
    
    # Rotate backups - keep only the most recent MAX_BACKUPS for this stability level
    BACKUP_COUNT=$(ls -1 "${BACKUP_DIR}/${PROJECT_NAME}_"*"_${STABILITY}.tar.gz" 2>/dev/null | wc -l)
    
    if [ $BACKUP_COUNT -gt $MAX_BACKUPS ]; then
        echo "🔄 Rotating ${STABILITY} backups (keeping ${MAX_BACKUPS} most recent)..."
        # List backups sorted by modification time, oldest first, and delete excess
        ls -1t "${BACKUP_DIR}/${PROJECT_NAME}_"*"_${STABILITY}.tar.gz" | tail -n +$((MAX_BACKUPS + 1)) | while read old_backup; do
            echo "🗑️  Deleting old backup: $(basename "$old_backup")"
            rm -f "$old_backup"
        done
    fi
    
    echo ""
    echo "📊 Current backups:"
    echo "   Stable: $(ls -1 "${BACKUP_DIR}/${PROJECT_NAME}_"*"_Stable.tar.gz" 2>/dev/null | wc -l)"
    echo "   Development: $(ls -1 "${BACKUP_DIR}/${PROJECT_NAME}_"*"_Development.tar.gz" 2>/dev/null | wc -l)"
else
    echo "❌ Error: Backup failed!"
    exit 1
fi
