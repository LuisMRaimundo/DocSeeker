**README.md**

This project is an **advanced document search tool** that supports searching through text files, PDF, DOCX, HTML, and image files (via OCR). It features a **Boolean search parser** for queries (using `AND`, `OR`, `NOT`, and quoted phrases), extraction of text from documents, relevancy scoring, duplicate detection, and a simple **Tkinter GUI**.

## Main Features

1. **Boolean Text Search**  
   - Supports `.txt`, `.pdf`, `.docx`, `.html`, and `.htm` files.  
   - For PDF files, it tries direct text extraction. If that fails, it falls back to OCR-based text extraction.  
   - For DOCX files, it reads each paragraph to search for matches.  
   - For HTML files, it processes text from selected tags (`<p>`, `<div>`, `<h1>`-`<h6>`).  
   - For images (`.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`), it uses OCR to extract text.

2. **Advanced Boolean Parsing**  
   - Supports `AND`, `OR`, `NOT` operators and parentheses for grouping.  
   - Quoted phrases, e.g. `"exact phrase"`.  
   - Relevancy scoring with term frequency and proximity measures.  
   - Generates a context snippet highlighting matched terms.

3. **Duplicate Detection**  
   - Identical results from multiple files or repeated sections are flagged as duplicates and can be omitted from the final report.

4. **OCR and Preprocessing**  
   - Automatically uses **Tesseract OCR** when native text extraction is not available.  
   - Preprocessing steps (deskew, contrast enhancements, binarization, etc.) to improve OCR accuracy.

5. **Graphical User Interface (Tkinter)**  
   - Select the directory containing documents and the output file location for results.  
   - Specify the Boolean query, relevancy threshold, context size, and file types to search.  
   - Start and monitor the search process within a user-friendly GUI.

6. **Report Generation**  
   - Saves a detailed report of all matches, grouped by file, sorted by relevance score.  
   - Shows a snippet of text where the match was found.

## Installation

1. **Install Tesseract OCR**  
   - **Linux (Ubuntu/Debian)**:  
     ```bash
     sudo apt-get update
     sudo apt-get install tesseract-ocr
     ```  
   - **Windows**: Download and install from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).  
   - **macOS** (Homebrew):  
     ```bash
     brew install tesseract
     ```

2. **(Optional) Create a Virtual Environment**  
   ```bash
   python -m venv venv
   # Linux/Mac
   source venv/bin/activate
   # or Windows
   .\venv\Scripts\activate
   ```

3. **Install Python Dependencies**  
   - See `requirements.txt` for the list of dependencies.  
   - Install them with:
     ```bash
     pip install -r requirements.txt
     ```

## Usage

1. **Clone or Download** this repository.  
2. **Run the Script**:
   ```bash
   python gpt_doc_scrab_v12.py
   ```
   This will open the **Tkinter interface**.

3. **In the GUI**:
   - **Directory**: select the folder containing your files (TXT, PDF, DOCX, HTML, images).  
   - **Output File**: choose or type a path for the output file (e.g., `search_results.txt`).  
   - **Boolean Search Query**: type something like `"machine learning" AND (Python OR R) AND NOT Java`.  
   - **Advanced Options** (optional):
     - **Minimum Relevance Score**: skip results below this score (default 0.1).  
     - **Context Window Size**: how many characters to show around the match (default 150).  
     - **Show Duplicate Results**: toggle whether duplicates should appear in the final report.  
   - **File Types to Search**: select among TXT, PDF, DOCX, HTML, Image.  
   - Click **Start Search** to begin.  
   - A success message will appear with the number of results. Open your output file to review the details.

## Tips

- Use uppercase `AND`, `OR`, `NOT` in your queries to ensure correct parsing.  
- Put exact phrases in quotes, e.g. `"machine learning"`.  
- Group complex expressions with parentheses, e.g. `(Python OR Java) AND "data analytics"`.  
- If you need advanced OCR, ensure Tesseract is properly installed and recognized by the system.

**LICENSE.md**

This package is licensed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License**.  
To view the full text of this license, visit: [https://creativecommons.org/licenses/by-nc-nd/4.0/](https://creativecommons.org/licenses/by-nc-nd/4.0/)


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
Please feel free to open an issue on this repository if you encounter problems or want to request more/new features to the code

**Luís Miguel da Luz Raimundo**
ORCID Profile: https://orcid.org/0000-0003-1712-6358

## Email Addresses:

lmr.2020@gmail.com

luisraimundo@fcsh.unl.pt


