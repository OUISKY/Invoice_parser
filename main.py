from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF
import io
import json
import torch, platform, sys

print("torch", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

from transformers import AutoProcessor, AutoModelForTokenClassification
from pytesseract import Output
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

processor = AutoProcessor.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice")
model = AutoModelForTokenClassification.from_pretrained("Theivaprakasham/layoutlmv3-finetuned-invoice", use_safetensors=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    model = model.half()
model.to(device).eval()

DATA_DIR = Path("data")
OUT_DIR = Path("parsed_data")
VISUALIZATION_DIR = Path("visualized_data")


def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)


def load_pages(path: Path, dpi: int = 300):
    if path.suffix.lower() == ".pdf":
        images = []
        with fitz.open(path) as doc:
            for page in doc:               
                pix = page.get_pixmap(dpi=dpi)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                img = maybe_downscale(img)
                images.append(img)
        return images
    else:
        try:
            return [Image.open(path).convert("RGB")]
        except Exception:
            return []

def maybe_downscale(image, max_side=1800):
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    scale = max_side / max(w, h)
    return image.resize((int(w*scale), int(h*scale)))

def encode_page(image, words, boxes):
	encoding = processor(
		image,
		words,
		boxes=boxes,
		return_tensors="pt",
		truncation=True,
		padding="max_length",
		max_length=384,
	)
	# Get word_ids while it's still a BatchEncoding
	word_ids = encoding.word_ids(batch_index=0)
	# Move only tensors to device (avoid deprecated device argument)
	for k, v in list(encoding.items()):
		if isinstance(v, torch.Tensor):
			encoding[k] = v.to(device)
	return encoding, word_ids

@torch.no_grad()
def predict_labels(encoding):
	if device.type == "cuda":
		with torch.autocast(device_type="cuda", dtype=torch.float16):
			outputs = model(**encoding)
	else:
		outputs = model(**encoding)
	pred_ids = outputs.logits.argmax(-1).squeeze(0).tolist()
	labels = [model.config.id2label[i] for i in pred_ids]
	return labels

def ocr_image_to_words_boxes(image, min_conf: float = 0.0):
    w, h = image.size
    data = pytesseract.image_to_data(image, output_type=Output.DICT, lang="eng", config="--oem 1 --psm 6")

    words, boxes = [], []
    n = len(data["text"])

    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if not text:
            continue
        if conf < min_conf:
            continue

        l = int(data["left"][i]); t = int(data["top"][i])
        wd = int(data["width"][i]); ht = int(data["height"][i])

        x0, y0 = l, t
        x1, y1 = l + wd, t + ht

        nx0 = max(0, min(1000, int(1000 * x0 / w)))
        ny0 = max(0, min(1000, int(1000 * y0 / h)))
        nx1 = max(0, min(1000, int(1000 * x1 / w)))
        ny1 = max(0, min(1000, int(1000 * y1 / h)))

        # Skip invalid tiny/empty boxes if any
        if nx1 <= nx0 or ny1 <= ny0:
            continue

        words.append(text)
        boxes.append([nx0, ny0, nx1, ny1])

    return words, boxes, (w, h)

def aggregate_entities(word_labels):
    entities = {}
    curr_type, curr_words = None, []

    def flush():
        nonlocal curr_type, curr_words
        if curr_type and curr_words:
            entities.setdefault(curr_type, []).append(" ".join(curr_words))
        curr_type, curr_words = None, []

    for word, label in word_labels:
        if label == "O":
            flush()
            continue
        tag, _, base = label.partition("-")  # "B-VENDOR" -> ("B", "-", "VENDOR")
        if tag == "B" or base != curr_type:
            flush()
            curr_type = base
            curr_words = [word]
        else:  # "I" with same base
            curr_words.append(word)
    flush()
    return entities  # dict like {"VENDOR": ["ACME Inc"], "TOTAL": ["$123.45"], ...}

def align_predictions_to_words(labels, word_ids, words):
	prev = None
	aligned = []
	for label, wid in zip(labels, word_ids):
		if wid is None or wid == prev:
			continue
		aligned.append((words[wid], label, wid))  # keep wid
		prev = wid
	return aligned

def _to_float_maybe(text: str):
    """
    Best-effort number parser for quantities/prices.
    Handles currency symbols, spaces, thousands separators, and comma decimals.
    Returns float or None.
    """
    if not text:
        return None
    s = text.strip()
    # Remove currency symbols and common prefixes
    for ch in ["€", "$", "£", "¥", "AUD", "CAD", "USD", "MAD"]:
        s = s.replace(ch, "")
    # Replace non-breaking spaces and regular spaces
    s = s.replace("\u00A0", "").replace(" ", "")
    # Normalize separators
    has_comma = "," in s
    has_dot = "." in s
    if has_comma and not has_dot:
        # Treat comma as decimal separator
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # Remove commas as thousands separators
        s = s.replace(",", "")
    # Remove thousand separators like apostrophes
    s = s.replace("'", "")
    try:
        return float(s)
    except Exception:
        return None

def extract_items_heuristic(words, boxes):
    """
    Heuristic fallback when model labels are not sufficient.
    Groups words into lines by Y center, then per line:
      - price := right-most numeric token
      - quantity := nearest numeric to the left of price that is an integer-like value
      - name := tokens left of quantity (or left of price if no quantity)
    Skips lines that look like totals.
    """
    if not words or not boxes:
        return []

    # Build tokens with geometry
    tokens = []
    for idx, (w, b) in enumerate(zip(words, boxes)):
        yctr = (b[1] + b[3]) // 2
        x0 = b[0]
        tokens.append((yctr, x0, w, idx))
    tokens.sort()

    # Group into lines by Y proximity
    lines = []
    line_tol = 12
    for y, x, w, idx in tokens:
        if not lines or abs(y - lines[-1]["y"]) > line_tol:
            lines.append({"y": y, "tokens": []})
        lines[-1]["tokens"].append((x, w))

    items = []
    for line in lines:
        toks = sorted(line["tokens"], key=lambda t: t[0])
        words_only = [w for _, w in toks]
        joined_lower = " ".join(words_only).lower()
        if "total" in joined_lower:
            continue

        # Identify numeric tokens with their positions
        numeric_positions = []  # (index, value)
        for idx, (_, w) in enumerate(toks):
            val = _to_float_maybe(w)
            if val is not None:
                numeric_positions.append((idx, val))

        if not numeric_positions:
            # No numeric → cannot infer price/qty; treat whole line as name
            name = " ".join(words_only).strip()
            if name:
                items.append({"name": name})
            continue

        # Price: right-most numeric
        price_idx, price_val = numeric_positions[-1]
        qty_val = None
        # Quantity: look left for integer-like small value
        for idx, val in reversed(numeric_positions[:-1]):
            if abs(idx - price_idx) <= 6:  # close by
                # Prefer integers or small decimals as quantities
                if abs(val - round(val)) < 1e-6 and 0 < val < 100000:
                    qty_val = int(round(val))
                    break
                if 0 < val < 10000:
                    qty_val = val
                    break

        # Name: tokens left of quantity if exists else left of price
        cut_idx = qty_val is not None and idx or price_idx
        name_tokens = [w for _, w in toks[:cut_idx]]
        name = " ".join(name_tokens).strip()
        item = {}
        if name:
            item["name"] = name
        if qty_val is not None:
            item["qte"] = qty_val
        if price_val is not None:
            item["price"] = price_val
        if item:
            items.append(item)

    return items

def parse_pages_to_entities(images, source_path: Path):
	pages = []
	for i, img in enumerate(images):
		words, boxes, (w, h) = ocr_image_to_words_boxes(img, min_conf=50.0)

		if not words:
			pages.append({
				"page_index": i,
				"width": w,
				"height": h,
				"items": [],
				"total": None
			})
			continue

		encoding, word_ids = encode_page(img, words, boxes)
		labels = predict_labels(encoding)
		word_triplets = align_predictions_to_words(labels, word_ids, words)
		
		# Create word-level labels list for visualization
		word_labels = ['O'] * len(words)  # Initialize with 'O' (no label)
		for word, label, wid in word_triplets:
			if wid < len(word_labels):
				word_labels[wid] = label
		
		# Create annotated image with model labels (in memory only)
		annotated_img = create_annotated_image(img, words, boxes, word_labels)
		
		# Store the annotated image for PDF creation later
		if 'annotated_images' not in locals():
			annotated_images = []
		annotated_images.append(annotated_img)
		
		# Debug: Print model labels and sample predictions
		if i == 0:  # Only print for first page
			print(f"\nModel labels: {list(model.config.id2label.values())}")
			print(f"Sample word predictions (first 20):")
			for j, (word, label, wid) in enumerate(word_triplets[:20]):
				print(f"  {word} -> {label}")
			print()

		# map model label bases -> desired fields
		def norm(label):
			tag, _, base = label.partition("-")
			base = base.upper().replace("-", "").replace("_", "")
			return tag, base

		# More comprehensive label mapping
		label_to_field = {
			# Item names
			"ITEM": "name", "DESCRIPTION": "name", "PRODUCT": "name", "SERVICE": "name", "GOODS": "name",
			"NAME": "name", "TITLE": "name", "DETAILS": "name",
			
			# References/SKUs
			"REF": "reference", "REFERENCE": "reference", "SKU": "reference", "ITEMNO": "reference", 
			"CODE": "reference", "PARTNO": "reference", "MODEL": "reference", "ID": "reference",
			
			# Quantities
			"QTY": "qte", "QUANTITY": "qte", "QTY": "qte", "AMOUNT": "qte", "UNITS": "qte",
			
			# Prices
			"PRICE": "price", "UNITPRICE": "price", "UNITCOST": "price", "RATE": "price", 
			"COST": "price", "VALUE": "price", "AMOUNT": "price", "LINEPRICE": "price",
			"SUBTOTAL": "price", "LINEAMOUNT": "price",
			
			# Totals
			"TOTAL": "total", "GRANDTOTAL": "total", "TOTALAMOUNT": "total", "AMOUNTDUE": "total",
			"BALANCE": "total", "DUE": "total", "FINAL": "total", "SUM": "total"
		}

		# collect tokens with geometry
		toks = []
		matched_labels = set()
		for word, label, wid in word_triplets:
			tag, base = norm(label)
			field = label_to_field.get(base)
			if not field:
				continue
			matched_labels.add(base)
			x0 = boxes[wid][0]
			yctr = (boxes[wid][1] + boxes[wid][3]) // 2
			toks.append((yctr, x0, field, tag, word, wid))
		toks.sort()  # by y then x
		
		# Debug: Show what labels were matched
		if i == 0:
			print(f"Matched labels: {matched_labels}")
			print(f"Total tokens found: {len(toks)}")

		# extract total (last span labeled as total)
		total_spans, cur, prev_y = [], [], None
		for y, x, field, tag, word, wid in toks:
			if field != "total":
				continue
			if tag == "B" or not cur or (prev_y is not None and abs(y - prev_y) > 8):
				if cur:
					total_spans.append(" ".join(cur))
				cur = [word]
			else:
				cur.append(word)
			prev_y = y
		if cur:
			total_spans.append(" ".join(cur))
		total_value = total_spans[-1] if total_spans else None

		# group line items by y (merge tokens close in y)
		line_tol = 15  # Increased tolerance
		lines = []
		for y, x, field, tag, word, wid in toks:
			if field == "total":
				continue
			if not lines or abs(y - lines[-1]["y"]) > line_tol:
				lines.append({"y": y, "parts": {"name": [], "reference": [], "qte": [], "price": []}})
			lines[-1]["parts"][field].append((x, word))

		items = []
		for line in lines:
			parts = line["parts"]
			item = {}
			for key in ("name", "reference", "qte", "price"):
				if parts[key]:
					item[key] = " ".join(w for x, w in sorted(parts[key], key=lambda t: t[0]))
			if item:
				items.append(item)

		# Heuristic fallback if items look empty or poor
		if not items:
			items = extract_items_heuristic(words, boxes)

		# Compute sum of line prices if available
		total_items_sum = None
		acc = 0.0
		found_any = False
		for it in items:
			price_val = _to_float_maybe(str(it.get("price", "")))
			qty_val = _to_float_maybe(str(it.get("qte", ""))) or 1.0
			if price_val is not None:
				acc += price_val  # if price is already line total
				found_any = True
		if found_any:
			total_items_sum = round(acc, 2)
		
		# Debug: Show extracted items
		if i == 0:
			print(f"Extracted {len(items)} items:")
			for j, item in enumerate(items[:5]):  # Show first 5 items
				print(f"  Item {j+1}: {item}")
			if items:
				print(f"  ... and {len(items)-5} more items" if len(items) > 5 else "")
			print(f"Total: {total_value}")
			print()

		pages.append({
			"page_index": i,
			"width": w,
			"height": h,
			"items": items,
			"total": total_value,
			"total_items_sum": total_items_sum
		})

	# After processing pages, build a single annotated PDF for this source
	try:
		if 'annotated_images' in locals() and annotated_images:
			pdf_out = VISUALIZATION_DIR / f"{source_path.stem}_annotated.pdf"
			create_pdf_from_images_in_memory(annotated_images, pdf_out)
	except Exception as e:
		print(f"Warning: failed to create annotated PDF for {source_path.name}: {e}")

	return {
		"source": str(source_path),
		"pages": pages,
	}

def create_annotated_image(image, words, boxes, word_labels=None):
    """
    Create an annotated image with bounding boxes around detected text.
    If word_labels is provided, color-code boxes by label type.
    """
    # Create a copy of the image to draw on
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)  # macOS
        except:
            font = ImageFont.load_default()
    
    # Color scheme for different label types
    colors = {
        'name': (255, 0, 0),      # Red for item names
        'reference': (0, 255, 0), # Green for references
        'qte': (0, 0, 255),       # Blue for quantities
        'price': (255, 165, 0),   # Orange for prices
        'total': (128, 0, 128),   # Purple for totals
        'default': (0, 0, 0)      # Black for unlabeled
    }
    
    # Convert normalized boxes back to pixel coordinates
    w, h = image.size
    
    for i, (word, box) in enumerate(zip(words, boxes)):
        # Convert from 0-1000 normalized coordinates back to pixels
        x0 = int(box[0] * w / 1000)
        y0 = int(box[1] * h / 1000)
        x1 = int(box[2] * w / 1000)
        y1 = int(box[3] * h / 1000)
        
        # Determine color based on label
        color = colors['default']
        if word_labels and i < len(word_labels):
            label = word_labels[i]
            if label != 'O':
                # Extract base label (remove B-/I- prefix)
                base = label.split('-')[-1].upper()
                if 'ITEM' in base or 'DESCRIPTION' in base or 'PRODUCT' in base:
                    color = colors['name']
                elif 'REF' in base or 'SKU' in base or 'CODE' in base:
                    color = colors['reference']
                elif 'QTY' in base or 'QUANTITY' in base:
                    color = colors['qte']
                elif 'PRICE' in base or 'COST' in base or 'AMOUNT' in base:
                    color = colors['price']
                elif 'TOTAL' in base:
                    color = colors['total']
        
        # Draw bounding box
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        
        # Draw label text above the box
        label_text = f"{word}"
        if word_labels and i < len(word_labels):
            label_text += f" ({word_labels[i]})"
        
        # Calculate text position (above the box)
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x0
        text_y = max(0, y0 - text_height - 2)
        
        # Draw text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                      fill=(255, 255, 255), outline=color)
        draw.text((text_x, text_y), label_text, fill=color, font=font)
    
    return annotated_img

def create_ocr_only_image(image, words, boxes):
    """
    Create a simple annotated image showing only OCR bounding boxes (no model labels).
    """
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except:
        font = ImageFont.load_default()
    
    # Convert normalized boxes back to pixel coordinates
    w, h = image.size
    
    for i, (word, box) in enumerate(zip(words, boxes)):
        # Convert from 0-1000 normalized coordinates back to pixels
        x0 = int(box[0] * w / 1000)
        y0 = int(box[1] * h / 1000)
        x1 = int(box[2] * w / 1000)
        y1 = int(box[3] * h / 1000)
        
        # Draw bounding box in blue
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=1)
        
        # Draw word text above the box
        text_bbox = draw.textbbox((0, 0), word, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = x0
        text_y = max(0, y0 - text_height - 1)
        
        # Draw text background
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                      fill=(255, 255, 255), outline=(0, 0, 255))
        draw.text((text_x, text_y), word, fill=(0, 0, 255), font=font)
    
    return annotated_img

def create_pdf_from_images_in_memory(images, output_pdf_path: Path):
    """
    Combine a list of PIL images in memory into a single multi-page PDF.
    """
    pdf_doc = fitz.open()
    try:
        for img in images:
            width, height = img.size
            page = pdf_doc.new_page(width=width, height=height)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)
            page.insert_image(fitz.Rect(0, 0, width, height), stream=img_bytes.getvalue())
        pdf_doc.save(str(output_pdf_path))
        print(f"Saved annotated PDF: {output_pdf_path}")
    finally:
        pdf_doc.close()

def save_json(result: dict, input_path: Path):
    out_path = OUT_DIR / f"{input_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")


def main():
    ensure_dirs()
    
    for path in DATA_DIR.iterdir():
        if not path.is_file():
            continue
        pages = load_pages(path, dpi=220)
        if not pages:
            print(f"Skipping unsupported or unreadable file: {path.name}")
            continue
        result = parse_pages_to_entities(pages, path)
        save_json(result, path)


if __name__ == "__main__":
    main()