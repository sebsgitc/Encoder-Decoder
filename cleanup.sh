#!/bin/bash

# Create a new orphan branch
git checkout --orphan temp_branch

# Add all the files (this will respect your .gitignore)
git add .

# Commit everything
git commit -m "Initial commit without large files"

# Delete the main branch
git branch -D main

# Rename temp branch to main
git branch -m main

# Force push to remote
git push -f origin main
