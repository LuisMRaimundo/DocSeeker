import os
import re
import hashlib
from pathlib import Path
from PyPDF2 import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import io
import unicodedata
from difflib import SequenceMatcher
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, messagebox
from collections import defaultdict

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from skimage import filters
import cv2
import pytesseract
import re

class ContentHash:
    def __init__(self):
        self.seen_hashes = {}
        
    def is_duplicate(self, content, filepath):
        """Check if content is duplicate and store hash if not"""
        # Create hash of content
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        
        if content_hash in self.seen_hashes:
            original_file = self.seen_hashes[content_hash]
            print(f"Duplicate content detected: {filepath} matches {original_file}")
            return True
            
        self.seen_hashes[content_hash] = filepath
        return False

class SearchResult:
    def __init__(self, filepath, query, location, context, relevance_score=0.0, snippet_start=0, snippet_end=0):
        self.filepath = filepath
        self.query = query
        self.location = location
        self.context = context
        self.relevance_score = relevance_score
        self.snippet_start = snippet_start
        self.snippet_end = snippet_end

class BooleanSearchParser:
    def __init__(self, query):
        self.original_query = query.strip()
        self.query = self.preprocess_query(query)
        self.search_terms = self.extract_search_terms(query)
        
    def evaluate(self, text):
        """Evaluate if text matches the boolean query and return match status and score"""
        if not self.query:
            return False, 0.0
            
        # First check if there's a match
        terms = self.tokenize(self.query)
        matches = self.evaluate_expression(terms, text)
        
        if matches:
            score = self.calculate_relevance_score(text, 0, len(text))
            return True, score
        return False, 0.0

    def preprocess_query(self, query):
        # Add explicit parentheses and normalize query
        query = query.strip()
        query = self.normalize_text(query)
        
        # Handle AND/OR precedence
        if 'AND' in query.upper() and 'OR' in query.upper() and '(' not in query:
            parts = query.split('AND')
            for i, part in enumerate(parts):
                if 'OR' in part.upper() and '(' not in part:
                    parts[i] = f"({part.strip()})"
            query = ' AND '.join(parts)
        return query

    def normalize_text(self, text):
        """Normalize text for consistent matching"""
        text = unicodedata.normalize('NFKC', text)
        return text
        
    def evaluate_expression(self, terms, text):
        stack = []
        operators = {'AND', 'OR', 'NOT'}
        i = 0
        
        while i < len(terms):
            term = terms[i]
            
            if term == '(':
                # Find matching parenthesis
                count = 1
                j = i + 1
                while j < len(terms) and count > 0:
                    if terms[j] == '(':
                        count += 1
                    elif terms[j] == ')':
                        count -= 1
                    j += 1
                if count == 0:
                    # Evaluate sub-expression
                    result = self.evaluate_expression(terms[i+1:j-1], text)
                    stack.append(result)
                    i = j
                    continue
                    
            elif term == 'NOT':
                # Handle NOT operator
                i += 1
                if i < len(terms):
                    result = self.evaluate_term(terms[i], text)
                    stack.append(not result)
                
            elif term in operators:
                stack.append(term)
                
            else:
                result = self.evaluate_term(term, text)
                stack.append(result)
                
            i += 1
            
        return self.process_stack(stack)

    def evaluate_term(self, term, text):
        # Enhanced term evaluation with better phrase matching
        if (term.startswith('"') and term.endswith('"')) or \
           (term.startswith("'") and term.endswith("'")):
            term = term[1:-1]  # Remove quotes
            # For phrases, require word boundaries and preserve word order
            words = term.split()
            pattern = r'\b' + r'\W+'.join(re.escape(word) for word in words) + r'\b'
        else:
            # For single words, just require word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
        
        return bool(re.search(pattern, text, re.IGNORECASE))

    def process_stack(self, stack):
        if not stack:
            return False
            
        result = stack[0]
        i = 1
        
        while i < len(stack):
            if stack[i] == 'AND':
                result = result and stack[i+1]
                i += 2
            elif stack[i] == 'OR':
                result = result or stack[i+1]
                i += 2
            else:
                result = stack[i]
                i += 1
                
        return result
    def extract_search_terms(self, query):
        """Extract individual search terms for relevance scoring"""
        terms = []
        current_term = ''
        in_quotes = False
        quote_char = None
        
        for char in query:
            if char in '"\'':
                if not in_quotes:
                    if current_term:
                        terms.extend(self._clean_term(current_term))
                    current_term = ''
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    terms.append(current_term)  # Keep quoted terms as is
                    current_term = ''
                    in_quotes = False
                else:
                    current_term += char
            elif in_quotes:
                current_term += char
            elif char.isspace() or char in '()':
                if current_term:
                    terms.extend(self._clean_term(current_term))
                current_term = ''
            else:
                current_term += char
                
        if current_term:
            terms.extend(self._clean_term(current_term))
            
        return [term for term in terms if term]

    def _clean_term(self, term):
        """Clean and validate search terms"""
        term = term.strip('"\'')
        if term.upper() in {'AND', 'OR', 'NOT'}:
            return []
        return [term]

    def tokenize(self, query):
        """Tokenize the query into terms and operators"""
        tokens = []
        current_token = ''
        in_quotes = False
        quote_char = None
        
        for char in query:
            if char in '"\'':
                if not in_quotes:
                    if current_token:
                        tokens.extend(current_token.split())
                        current_token = ''
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    tokens.append(current_token)
                    current_token = ''
                    in_quotes = False
                else:
                    current_token += char
            elif char in '()':
                if current_token:
                    tokens.extend(current_token.split())
                    current_token = ''
                tokens.append(char)
            elif in_quotes:
                current_token += char
            else:
                current_token += char
            
        if current_token:
            tokens.extend(current_token.split())
            
        return tokens

    def calculate_relevance_score(self, text, start, end):
        """Calculate relevance score for a text snippet"""
        score = 0.0
        text_lower = text.lower()
        text_norm = self.normalize_text(text_lower)
        
        # Term frequency scoring
        term_counts = defaultdict(int)
        for term in self.search_terms:
            term_norm = self.normalize_text(term.lower())
            count = len(re.findall(r'\b' + re.escape(term_norm) + r'\b', text_norm))
            term_counts[term] = count
            
            # Base score from term frequency
            score += count * 0.1
            
            # Bonus for exact phrase matches
            if f'"{term}"' in self.original_query:
                score += count * 0.2
        
        # Proximity scoring
        words = text_norm.split()
        term_positions = defaultdict(list)
        
        for i, word in enumerate(words):
            for term in self.search_terms:
                term_norm = self.normalize_text(term.lower())
                if term_norm in self.normalize_text(word):
                    term_positions[term].append(i)
        
        # Calculate proximity bonuses
        if len(term_positions) > 1:
            min_distance = float('inf')
            for term1, pos1 in term_positions.items():
                for term2, pos2 in term_positions.items():
                    if term1 != term2:
                        for p1 in pos1:
                            for p2 in pos2:
                                distance = abs(p1 - p2)
                                min_distance = min(min_distance, distance)
            
            if min_distance != float('inf'):
                proximity_bonus = 1.0 / (1.0 + min_distance)
                score += proximity_bonus
        
        # Context relevance bonus
        context_length = end - start
        if context_length < 200:  # Bonus for more focused matches
            score *= 1.2
        
        # Bonus for matches appearing early in the text
        if start < len(text) / 3:
            score *= 1.1
        
        return score

    def extract_context(self, text, window_size=150):
        """Extract relevant context around matching terms"""
        best_context = ""
        best_score = 0
        best_start = 0
        best_end = 0
        text_norm = self.normalize_text(text)
        
        # Find all term positions
        positions = []
        for term in self.search_terms:
            term_norm = self.normalize_text(term)
            for match in re.finditer(r'\b' + re.escape(term_norm) + r'\b', text_norm, re.IGNORECASE):
                positions.append((match.start(), match.end()))
        
        if not positions:
            return text[:200], 0, 200
            
        # Find best context window
        positions.sort()
        for start_pos, _ in positions:
            window_start = max(0, start_pos - window_size)
            window_end = min(len(text), start_pos + window_size)
            
            # Adjust to word boundaries
            while window_start > 0 and text[window_start].isalnum():
                window_start -= 1
            while window_end < len(text) and text[window_end-1].isalnum():
                window_end += 1
                
            context = text[window_start:window_end]
            score = self.calculate_relevance_score(context, window_start, window_end)
            
            if score > best_score:
                best_score = score
                best_context = context
                best_start = window_start
                best_end = window_end
        
        # Clean up context
        best_context = re.sub(r'\s+', ' ', best_context).strip()
        best_context = unicodedata.normalize('NFKC', best_context)
        
        # Highlight matching terms
        for term in sorted(self.search_terms, key=len, reverse=True):
            term_norm = self.normalize_text(term)
            best_context = re.sub(
                f'(?i)\\b{re.escape(term_norm)}\\b',
                f'**{term}**',
                best_context
            )
        
        return best_context, best_start, best_end
        
def save_results(output_file, results, show_duplicates=False):
    """Save search results with improved formatting and duplicate handling"""
    with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write("Search Results\n")
        f.write("=" * 50 + "\n\n")
        
        # Group by file
        grouped_results = defaultdict(list)
        seen_content = set()
        
        for result in results:
            # Check for duplicate content
            content_key = (result.context, result.query)
            if not show_duplicates and content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            grouped_results[result.filepath].append(result)
            
        # Write results for each file
        for filepath, file_results in grouped_results.items():
            f.write(f"File: {filepath}\n")
            f.write("-" * 50 + "\n")
            
            # Sort by relevance score
            file_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            for result in file_results:
                f.write(f"  Search Query: {result.query}\n")
                f.write(f"  Relevance Score: {result.relevance_score:.2f}\n")
                f.write("  " + "-" * 40 + "\n")
                f.write(f"    Location: {result.location}\n")
                f.write(f"    Context: {result.context}\n")
                f.write("\n")
            f.write("=" * 50 + "\n\n")
def search_in_text_file(filepath, boolean_parser, context_size):
    """Search in text files with context tracking"""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            text = file.read()
            matches, score = boolean_parser.evaluate(text)
            if matches:
                context, start, end = boolean_parser.extract_context(text, context_size)
                results.append(SearchResult(
                    filepath=filepath,
                    query=boolean_parser.original_query,
                    location=1,
                    context=context,
                    relevance_score=score,
                    snippet_start=start,
                    snippet_end=end
                ))
    except Exception as e:
        print(f"Error reading text file {filepath}: {e}")
    return results

def search_in_pdf_file(filepath, boolean_parser, context_size):
    """Search in PDF files with detailed logging"""
    results = []
    try:
        print(f"\nProcessing PDF: {filepath}")
        reader = PdfReader(filepath)
        print(f"Total pages found: {len(reader.pages)}")
        
        for page_num, page in enumerate(reader.pages, 1):
            try:
                print(f"\nProcessing page {page_num}")
                
                # Try normal text extraction first
                text = page.extract_text()
                print(f"Text extraction result length: {len(text) if text else 0}")
                
                # If no text is extracted, try OCR
                if not text or text.isspace():
                    print("No text found, attempting OCR...")
                    try:
                        from pdf2image import convert_from_path
                        print("Converting PDF page to image...")
                        images = convert_from_path(filepath, first_page=page_num, last_page=page_num)
                        
                        if images:
                            print("Successfully converted page to image, performing OCR...")
                            text = extract_text_from_image(images[0])
                            print(f"OCR result length: {len(text) if text else 0}")
                        else:
                            print("No image converted from PDF page")
                            
                    except Exception as e:
                        print(f"Error during OCR processing: {str(e)}")
                
                if text:
                    print("Evaluating extracted text...")
                    matches, score = boolean_parser.evaluate(text)
                    if matches:
                        print(f"Match found with score: {score}")
                        context, start, end = boolean_parser.extract_context(text, context_size)
                        results.append(SearchResult(
                            filepath=filepath,
                            query=boolean_parser.original_query,
                            location=page_num,
                            context=context,
                            relevance_score=score,
                            snippet_start=start,
                            snippet_end=end
                        ))
                    else:
                        print("No matches in this page")
                else:
                    print("No text to evaluate")
                    
            except Exception as e:
                print(f"Error processing page {page_num}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
    
    print(f"\nResults found for this PDF: {len(results)}")
    return results

def search_in_docx_file(filepath, boolean_parser, context_size):
    """Search in Word documents with context tracking"""
    results = []
    try:
        doc = Document(filepath)
        for para_num, paragraph in enumerate(doc.paragraphs, 1):
            text = paragraph.text
            if text:
                matches, score = boolean_parser.evaluate(text)
                if matches:
                    context, start, end = boolean_parser.extract_context(text, context_size)
                    results.append(SearchResult(
                        filepath=filepath,
                        query=boolean_parser.original_query,
                        location=para_num,
                        context=context,
                        relevance_score=score,
                        snippet_start=start,
                        snippet_end=end
                    ))
    except Exception as e:
        print(f"Error reading DOCX file {filepath}: {e}")
    return results

def search_in_html_file(filepath, boolean_parser, context_size):
    """Search in HTML files with context tracking"""
    results = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Search in different HTML elements
            for element_num, element in enumerate(soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']), 1):
                text = element.get_text(strip=True)
                if text:
                    matches, score = boolean_parser.evaluate(text)
                    if matches:
                        context, start, end = boolean_parser.extract_context(text, context_size)
                        results.append(SearchResult(
                            filepath=filepath,
                            query=boolean_parser.original_query,
                            location=element_num,
                            context=context,
                            relevance_score=score,
                            snippet_start=start,
                            snippet_end=end
                        ))
    except Exception as e:
        print(f"Error reading HTML file {filepath}: {e}")
    return results

def search_in_image_file(filepath, boolean_parser, context_size):
    """Search in images using OCR with context tracking"""
    results = []
    try:
        text = extract_text_from_image(filepath)
        if text:
            matches, score = boolean_parser.evaluate(text)
            if matches:
                context, start, end = boolean_parser.extract_context(text, context_size)
                results.append(SearchResult(
                    filepath=filepath,
                    query=boolean_parser.original_query,
                    location=1,
                    context=context,
                    relevance_score=score,
                    snippet_start=start,
                    snippet_end=end
                ))
    except Exception as e:
        print(f"Error reading image file {filepath}: {e}")
    return results

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy using advanced techniques.
    """
    try:
        # Converter para escala de cinza se necessário
        if image.mode != 'L':
            image = image.convert('L')

        # Aumentar o contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(3.0)  # Aumentando mais o contraste

        # Converter para array NumPy para OpenCV
        img_array = np.array(image)

        # Aplicar desfoque gaussiano para redução de ruído
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0)

        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_array = clahe.apply(img_array)

        # Aplicar um filtro bilateral (suaviza sem perder bordas)
        img_array = cv2.bilateralFilter(img_array, 5, 75, 75)

        # Aplicar binarização Otsu
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Aplicar operações morfológicas para remover ruído (Dilatação e Erosão)
        kernel = np.ones((1, 1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Converter de volta para imagem PIL
        return Image.fromarray(binary)

    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return image  # Retorna a imagem original se houver erro



def deskew_image(image):
    """
    Detect and correct skew in an image using OpenCV.
    """
    try:
        # Convert PIL image to numpy array (grayscale)
        img_array = np.array(image.convert('L'))

        # Apply thresholding to obtain a binary image
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find non-zero pixel coordinates
        coords = np.column_stack(np.where(binary > 0))

        # Compute the angle of the text
        angle = cv2.minAreaRect(coords)[-1]

        # Adjust angle
        if angle < -45:
            angle = 90 + angle
        else:
            angle = -angle

        # Get image dimensions
        (h, w) = img_array.shape[:2]
        center = (w // 2, h // 2)

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Apply affine transformation to correct skew
        rotated = cv2.warpAffine(img_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # Convert back to PIL image
        return Image.fromarray(rotated)

    except Exception as e:
        print(f"Error deskewing image: {e}")
        return image  # Return the original image if an error occurs


def extract_text_from_image(image_input):
    """
    Enhanced text extraction from images using preprocessing and configured OCR.
    Accepts either a file path or a PIL Image object.
    """
    try:
        # Se for um caminho, abrir a imagem
        if isinstance(image_input, str):  # Caminho do arquivo
            image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):  # Objeto PIL Image
            image = image_input
        else:
            raise ValueError("Invalid image input type")

        # Ajuste de DPI se necessário
        try:
            dpi = image.info.get('dpi', (300, 300))[0]
        except:
            dpi = 300

        # Redimensionamento se DPI for baixo
        if dpi < 300:
            scale_factor = 300 / dpi
            new_size = tuple(int(dim * scale_factor) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Pré-processamento para melhorar OCR
        processed_image = preprocess_image(image)
        processed_image = deskew_image(processed_image)

        # Configuração do OCR
        custom_config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'

        # Tentativa com diferentes modos de segmentação
        psm_modes = [3, 6, 4, 1]
        text = ''
        for psm in psm_modes:
            try:
                config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(
                    processed_image,
                    config=config,
                    lang='eng+fra+por+spa'  # Adicione outros idiomas conforme necessário
                )
                if text.strip():  # Se capturou texto, parar a busca
                    break
            except Exception as e:
                print(f"Erro com PSM {psm}: {e}")
                continue

        # Pós-processamento do texto extraído
        if text:
            text = re.sub(r'\s+', ' ', text)
            text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
            text = text.replace('|', 'I').replace('1', 'l')
            return text.strip()
        else:
            print("Nenhum texto extraído")
            return ""

    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        return ""


def search_in_files(directory, boolean_query, file_types, min_relevance=0.1, context_size=150):
    """Main search function that coordinates the search across all file types"""
    results = []
    boolean_parser = BooleanSearchParser(boolean_query)
    content_hash = ContentHash()
    
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            lower_file = file.lower()
            
            try:
                if lower_file.endswith('.txt') and file_types['txt']:
                    results.extend(search_in_text_file(filepath, boolean_parser, context_size))
                elif lower_file.endswith('.pdf') and file_types['pdf']:
                    results.extend(search_in_pdf_file(filepath, boolean_parser, context_size))
                elif lower_file.endswith('.docx') and file_types['docx']:
                    results.extend(search_in_docx_file(filepath, boolean_parser, context_size))
                elif lower_file.endswith(('.html', '.htm')) and file_types['html']:
                    results.extend(search_in_html_file(filepath, boolean_parser, context_size))
                elif lower_file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')) and file_types['image']:
                    results.extend(search_in_image_file(filepath, boolean_parser, context_size))
            except Exception as e:
                print(f"Error processing file {filepath}: {e}")
                continue
    
    # Filter results by relevance score and remove duplicates
    filtered_results = []
    seen_content = set()
    
    for result in results:
        if result.relevance_score >= min_relevance:
            # Create a unique key for the content
            content_key = (result.context, result.query)
            
            # Check if this is duplicate content
            if not content_hash.is_duplicate(result.context, result.filepath):
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    filtered_results.append(result)
    
    # Sort results by relevance score
    filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
    
    return filtered_results


def run_interface():
    def select_directory():
        directory = filedialog.askdirectory(title="Select Directory with Documents")
        directory_var.set(directory)

    def select_output_file():
        file_path = filedialog.asksaveasfilename(
            title="Select Output File",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        output_file_var.set(file_path)

    def show_help():
        help_text = """
Boolean Search Syntax:
- Use AND, OR, NOT operators in CAPS
- Use parentheses for grouping
- Use quotes for exact phrases

Examples:
  * term1 AND term2
  * (term1 OR term2) AND term3
  * term1 AND NOT term2
  * "exact phrase" AND term

Advanced Features:
- Relevance scoring helps rank results
- Duplicate detection removes identical content
- Context window size affects excerpt length
- Special character handling (é, è, etc.)
"""
        messagebox.showinfo("Boolean Search Help", help_text)

    def start_search():
        directory = directory_var.get()
        output_file = output_file_var.get()
        search_query = terms_var.get()
        min_relevance = float(relevance_var.get())
        context_size = int(context_var.get())
        show_duplicates = show_duplicates_var.get()
        
        file_types = {
            'txt': txt_var.get(),
            'pdf': pdf_var.get(),
            'docx': docx_var.get(),
            'html': html_var.get(),
            'image': image_var.get()
        }

        if not directory:
            messagebox.showerror("Error", "No directory selected.")
            return
        if not output_file:
            messagebox.showerror("Error", "No output file selected.")
            return
        if not search_query:
            messagebox.showerror("Error", "No search query provided.")
            return

        try:
            results = search_in_files(directory, search_query, file_types, 
                                    min_relevance=min_relevance,
                                    context_size=context_size)
            
            if results:
                save_results(output_file, results, show_duplicates=show_duplicates)
                messagebox.showinfo("Results Found", 
                                  f"Found {len(results)} results.\nSaved to: {output_file}")
            else:
                messagebox.showinfo("No Results", "No matches found for the given query.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Initialize Tkinter
    root = Tk()
    root.title("Enhanced Document Search Tool")
    root.geometry("700x900")

    # Style configuration
    style = ttk.Style()
    style.configure('TFrame', padding=5)
    style.configure('Header.TLabel', font=('TkDefaultFont', 10, 'bold'))
    style.configure('Accent.TButton', font=('TkDefaultFont', 10, 'bold'))

    # Main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.grid(row=0, column=0, sticky=(N, W, E, S))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # Variables
    directory_var = StringVar()
    output_file_var = StringVar()
    terms_var = StringVar()
    txt_var = BooleanVar(value=True)
    pdf_var = BooleanVar(value=True)
    docx_var = BooleanVar(value=True)
    html_var = BooleanVar(value=True)
    image_var = BooleanVar(value=True)
    relevance_var = StringVar(value="0.1")
    context_var = StringVar(value="150")
    show_duplicates_var = BooleanVar(value=False)

    # Directory selection
    ttk.Label(main_frame, text="Directory:", style='Header.TLabel').grid(column=0, row=0, sticky=W)
    ttk.Button(main_frame, text="Select Directory", command=select_directory).grid(column=1, row=0, pady=5)
    ttk.Label(main_frame, textvariable=directory_var, wraplength=500).grid(column=0, row=1, columnspan=2, sticky=W)

    # Output file selection
    ttk.Label(main_frame, text="Output File:", style='Header.TLabel').grid(column=0, row=2, sticky=W, pady=(10,0))
    ttk.Button(main_frame, text="Select Output File", command=select_output_file).grid(column=1, row=2, pady=5)
    ttk.Label(main_frame, textvariable=output_file_var, wraplength=500).grid(column=0, row=3, columnspan=2, sticky=W)

    # Search query
    ttk.Label(main_frame, text="Boolean Search Query:", style='Header.TLabel').grid(column=0, row=4, sticky=W, pady=(10,0))
    ttk.Entry(main_frame, textvariable=terms_var, width=60).grid(column=0, row=5, columnspan=2, sticky=(W,E), pady=5)
    ttk.Button(main_frame, text="Search Help", command=show_help).grid(column=0, row=6, sticky=W)

    # Advanced options frame
    advanced_frame = ttk.LabelFrame(main_frame, text="Advanced Options", padding="5")
    advanced_frame.grid(column=0, row=7, columnspan=2, sticky=(W,E), pady=10)

    # Relevance score threshold
    ttk.Label(advanced_frame, text="Minimum Relevance Score (0.0-1.0):").grid(column=0, row=0, sticky=W)
    ttk.Entry(advanced_frame, textvariable=relevance_var, width=10).grid(column=1, row=0, sticky=W, padx=5)

    # Context window size
    ttk.Label(advanced_frame, text="Context Window Size (characters):").grid(column=0, row=1, sticky=W, pady=5)
    ttk.Entry(advanced_frame, textvariable=context_var, width=10).grid(column=1, row=1, sticky=W, padx=5)

    # Show duplicates option
    ttk.Checkbutton(advanced_frame, text="Show Duplicate Results", variable=show_duplicates_var).grid(column=0, row=2, columnspan=2, sticky=W)

    # File type selection frame
    file_type_frame = ttk.LabelFrame(main_frame, text="File Types to Search", padding="5")
    file_type_frame.grid(column=0, row=8, columnspan=2, sticky=(W,E), pady=10)

    ttk.Checkbutton(file_type_frame, text="Text Files (.txt)", variable=txt_var).grid(column=0, row=0, sticky=W)
    ttk.Checkbutton(file_type_frame, text="PDF Files (.pdf)", variable=pdf_var).grid(column=0, row=1, sticky=W)
    ttk.Checkbutton(file_type_frame, text="Word Files (.docx)", variable=docx_var).grid(column=0, row=2, sticky=W)
    ttk.Checkbutton(file_type_frame, text="HTML Files (.html, .htm)", variable=html_var).grid(column=0, row=3, sticky=W)
    ttk.Checkbutton(file_type_frame, text="Image Files (.png, .jpg, .jpeg, .tiff, .bmp)", variable=image_var).grid(column=0, row=4, sticky=W)

    # Search button
    ttk.Button(main_frame, text="Start Search", command=start_search, style='Accent.TButton').grid(column=0, row=9, columnspan=2, pady=20)

    # Status bar
    status_var = StringVar()
    status_bar = ttk.Label(main_frame, textvariable=status_var, relief='sunken')
    status_bar.grid(column=0, row=10, columnspan=2, sticky=(W,E))

    # Configure grid weights
    main_frame.columnconfigure(1, weight=1)
    
    # Add padding to all children of main_frame
    for child in main_frame.winfo_children():
        child.grid_configure(padx=5)
        
    # Add padding to all children of advanced_frame
    for child in advanced_frame.winfo_children():
        child.grid_configure(padx=5)
        
    # Add padding to all children of file_type_frame
    for child in file_type_frame.winfo_children():
        child.grid_configure(padx=5)

    root.mainloop()

if __name__ == "__main__":
    run_interface()            