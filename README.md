# Document Search Tool

This project is an **advanced document search tool** that supports searching through text files, PDF, DOCX, HTML, and image files (via OCR). It provides a **Boolean search parser** (`AND`, `OR`, `NOT`), relevance scoring, duplicate detection, and includes both a **Tkinter GUI** and a **Flask web interface**.

## Main Features

1. **Boolean Text Search**  
   - Supports `.txt`, `.pdf`, `.docx`, `.html`, `.htm`, and image-based text search.
   - Uses **direct text extraction** for PDFs and DOCX files.
   - Applies **OCR (Tesseract)** to extract text from scanned PDFs and image files.
   - Extracts text from HTML files based on common tags (`<p>`, `<div>`, `<h1>`-`<h6>`).

2. **Advanced Boolean Parsing**  
   - Supports `AND`, `OR`, `NOT` operators and parentheses.
   - Recognizes **quoted phrases** (e.g., "exact phrase").
   - Uses **relevance scoring** based on term frequency and proximity.
   - Displays contextual snippets around matched text.

3. **Metadata Extraction**  
   - Extracts **file properties** (size, creation date, modification date).
   - Retrieves **PDF metadata** (author, title, keywords).
   - Reads **DOCX file properties** (e.g., author information).

4. **Duplicate Detection**  
   - Identical results are flagged as duplicates and can be filtered.
   - Uses **MD5 hashing** to detect duplicate files.

5. **OCR and Preprocessing**  
   - Uses **Tesseract OCR** when native text extraction is unavailable.
   - Includes preprocessing steps (deskewing, contrast enhancement, binarization) to improve OCR accuracy.

6. **Graphical User Interface (Tkinter)**  
   - Provides an intuitive **desktop interface** for selecting files and queries.
   - Allows users to set **minimum relevance score, context size, and file types**.
   - Displays **search progress** with a status bar.

7. **Web-Based Interface (Flask)**  
   - Users can access a web-based search tool via **http://127.0.0.1:5000/**.
   - The `/search` API endpoint allows programmatic queries.
   - JSON responses provide structured search results.

8. **Report Generation**  
   - Saves a detailed report of all matches, grouped by file.
   - Sorts results by **relevance score**.
   - Includes a **contextual snippet** around each match.

## Installation

### **1. Install Tesseract OCR**  
- **Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```
- **Windows:** Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).
- **macOS (Homebrew):**
  ```bash
  brew install tesseract
  ```

### **2. Install Python Dependencies**  
Run the following command to install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### **GUI Mode (Tkinter)**
1. Run the script:
   ```bash
   python docseeker_v2.py gui
   ```
2. Use the GUI to:
   - Select the **directory** containing documents.
   - Choose the **output file** for results.
   - Enter a **Boolean search query** (e.g., `"machine learning" AND Python NOT Java`).
   - Adjust **relevance score, context size, and duplicate filtering**.
   - Select the **file types** to include in the search.
   - Click **Start Search** to begin.
3. A report will be generated with the results.

### **Web Mode (Flask)**
1. Run the script:
   ```bash
   python docseeker_v2.py web
   ```
2. Open **http://127.0.0.1:5000/** in a browser.
3. Enter a search query and select the document types.
4. View structured search results.

## Tips

- Use uppercase `AND`, `OR`, `NOT` in your queries to ensure correct parsing.
- Put exact phrases in quotes, e.g., `"machine learning"`.
- Group complex expressions with parentheses, e.g., `(Python OR Java) AND "data analytics"`.
- If you need advanced OCR, ensure **Tesseract** is installed and correctly recognized by the system.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**. 
For full license details, see the [`LICENSE.md`](LICENSE.md) file.

---

## Acknowledgments

This software was developed by Luís Raimundo, as part of a broader study on Music Analysis
**DOI 10.54499/2020.08817.BD 8D** (https://doi.org/10.54499/2020.08817.BD) 

and was funded by:

**Foundation for Science and Technology (FCT)** - Portugal

And also supported by:

**Universidade NOVA de Lisboa**

**Centre for the Study of Sociology and Musical Aesthetics** (CESEM)

**Contemporary Music Group Investigation** (GIMC)

**In2Past**

---

## Contact
If you have any questions or suggestions, please open an issue on the GitHub repository or contact the maintainer.

**Luís Miguel da Luz Raimundo**
ORCID Profile: https://orcid.org/0000-0003-1712-6358

## Email Addresses:

lmr.2020@outlook.pt

luisraimundo@fcsh.unl.pt


