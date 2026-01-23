# Markdown to PDF Conversion Guide

## Method 1: Using Pandoc (Recommended)

### Installation

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended texlive-extra-utils texlive-latex-extra
```

**On macOS:**
```bash
brew install pandoc basictex
```

### Basic Usage

```bash
pandoc input.md -o output.pdf
```

### Using the Provided Script

```bash
./markdown_to_pdf.sh README.md
# or specify output name
./markdown_to_pdf.sh README.md custom_output.pdf
```

### Advanced Pandoc Options

```bash
# With custom margins and font size
pandoc input.md -o output.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    --highlight-style=tango

# With table of contents
pandoc input.md -o output.pdf --toc

# With custom template
pandoc input.md -o output.pdf --template=template.tex
```

## Method 2: Using Python (markdown2pdf)

### Installation
```bash
pip install markdown2pdf
```

### Usage
```python
from markdown2pdf import convert_markdown_to_pdf
convert_markdown_to_pdf('input.md', 'output.pdf')
```

## Method 3: Using Python (markdown + weasyprint)

### Installation
```bash
pip install markdown weasyprint
```

### Usage
```python
import markdown
from weasyprint import HTML

with open('input.md', 'r') as f:
    html = markdown.markdown(f.read())
    
HTML(string=html).write_pdf('output.pdf')
```

## Method 4: Using VS Code Extension

1. Install "Markdown PDF" extension in VS Code
2. Open your markdown file
3. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
4. Type "Markdown PDF: Export (pdf)" and select it

## Method 5: Online Converters

- [Markdown to PDF](https://www.markdowntopdf.com/)
- [Dillinger](https://dillinger.io/) - Export as PDF
- [StackEdit](https://stackedit.io/) - Export as PDF

## Batch Conversion

To convert all markdown files in a directory:

```bash
for file in *.md; do
    pandoc "$file" -o "${file%.md}.pdf"
done
```

Or using the script:

```bash
for file in *.md; do
    ./markdown_to_pdf.sh "$file"
done
```

## Troubleshooting

### Pandoc LaTeX errors
If you get LaTeX errors, try installing a full LaTeX distribution:
```bash
sudo apt-get install texlive-full
```

### Font issues
For better font support, use XeLaTeX:
```bash
pandoc input.md -o output.pdf --pdf-engine=xelatex
```

### Chinese/Japanese/Korean characters
Install additional fonts:
```bash
sudo apt-get install fonts-noto-cjk
```



