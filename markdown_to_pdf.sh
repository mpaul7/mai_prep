#!/bin/bash

# Markdown to PDF Converter Script
# Usage: ./markdown_to_pdf.sh input.md [output.pdf]

# Check if pandoc is installed (check PATH and ~/.local/bin)
if ! command -v pandoc &> /dev/null; then
    # Check if pandoc exists in ~/.local/bin
    if [ -f "$HOME/.local/bin/pandoc" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "Error: pandoc is not installed."
        echo ""
        echo "To install pandoc and required LaTeX packages, run:"
        echo "  sudo apt-get update"
        echo "  sudo apt-get install -y pandoc texlive-latex-base texlive-fonts-recommended texlive-extra-utils texlive-latex-extra texlive-latex-recommended texlive-luatex"
        echo ""
        echo "Or if you have conda:"
        echo "  conda install -c conda-forge pandoc"
        exit 1
    fi
fi

# Check if lualatex is available (for math rendering)
if ! command -v lualatex &> /dev/null; then
    echo "Warning: lualatex is not installed. Math rendering may not work properly."
    echo "Install it with: sudo apt-get install texlive-luatex texlive-latex-recommended"
    echo "Attempting conversion anyway..."
fi

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input.md> [output.pdf]"
    echo "Example: $0 README.md README.pdf"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.md}.pdf}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Convert markdown to PDF with proper math rendering
echo "Converting $INPUT_FILE to $OUTPUT_FILE..."
pandoc "$INPUT_FILE" -o "$OUTPUT_FILE" \
    --pdf-engine=lualatex \
    --standalone \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V colorlinks=true \
    --highlight-style=tango \
    -V documentclass=article \
    -V classoption=oneside

if [ $? -eq 0 ]; then
    echo "Success! PDF created: $OUTPUT_FILE"
else
    echo "Error: Conversion failed."
    exit 1
fi

