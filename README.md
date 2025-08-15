# Invoice Parser

An AI-powered invoice extraction tool using LayoutLMv3 to extract key information from invoice images and PDFs.

## Features

- Extracts items (name, reference, quantity, price) and total from invoices
- Supports both image files (PNG, JPG, etc.) and PDFs
- Uses OCR (Tesseract) for text extraction
- Powered by LayoutLMv3 model fine-tuned for invoice parsing
- GPU acceleration support

## Requirements

- Python 3.12+
- CUDA-compatible GPU (optional, for faster processing)
- Tesseract OCR

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd invoice-parser
```

2. Create a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
# Install all dependencies from requirements.txt
pip install -r requirements.txt

# OR install manually:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers pillow pymupdf pytesseract safetensors
```

4. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Or use: `winget install -e --id Tesseract-OCR.Tesseract`

## Usage

1. Place your invoice files (images or PDFs) in the `data/` folder
2. Run the parser:
```bash
python main.py
```
3. Check the `parsed_data/` folder for JSON results

## Output Format

Each invoice generates a JSON file with:
- Page information (width, height)
- Extracted items with name, reference, quantity, and price
- Total amount

## Configuration

- Adjust OCR confidence threshold in `main.py`
- Modify model sequence length for speed/accuracy trade-off
- Change DPI settings for PDF rendering

## License

[Add your license here]

