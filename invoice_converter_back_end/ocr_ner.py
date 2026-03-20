import pytesseract
from PIL import Image
import spacy
import cv2
import numpy as np
from pdf2image import convert_from_path
import os
import stanza  # Türkçe NER için

# Tesseract Türkçe dil desteği için
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows için gerekli

# Dil modellerini yükle
try:
    # İngilizce için spaCy
    en_nlp = spacy.load("en_core_web_lg")
    
    # Türkçe için Stanza (spaCy'nin Türkçe modeli de kullanılabilir)
    stanza.download("tr")  # Sadece ilk çalıştırmada gerekli
    tr_nlp = stanza.Pipeline("tr")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    en_nlp = None
    tr_nlp = None

# Görseli OCR için ön işlemeye al (Türkçe karakterler için optimize edildi)
def preprocess_image(image):
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Türkçe karakterler için daha iyi sonuç veren filtreleme
        gray = cv2.fastNlMeansDenoising(gray, h=30)
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    except Exception as e:
        print(f"Görüntü işleme hatası: {e}")
        return image

# Görselden metin çıkar (Türkçe+İngilizce)
def extract_text_from_image(image_path, lang='tur+eng'):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Görüntü yüklenemedi")
            
        processed_image = preprocess_image(image)
        
        # Türkçe için özel config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config, lang=lang)
        return text.strip()
    except Exception as e:
        print(f"Metin çıkarma hatası: {e}")
        return ""

# PDF'den metin çıkar (Türkçe+İngilizce)
def extract_text_from_pdf(pdf_path, lang='tur+eng'):
    try:
        images = convert_from_path(pdf_path)
        all_text = ""
        for img in images:
            image_np = np.array(img)
            processed_image = preprocess_image(image_np)
            
            # Türkçe için özel config
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed_image, config=custom_config, lang=lang)
            all_text += text + "\n"
        return all_text.strip()
    except Exception as e:
        print(f"PDF işleme hatası: {e}")
        return ""

# Dosya türüne göre yönlendir
def extract_text(file_path, lang='tur+eng'):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        return extract_text_from_image(file_path, lang)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path, lang)
    else:
        return ""

# Metni sınıflandır (Türkçe ve İngilizce destekli)
def classify_text(text, lang='tr'):
    entities = []
    
    if not text.strip():
        return entities
    
    try:
        if lang == 'tr' and tr_nlp:
            # Türkçe NER işlemi
            doc = tr_nlp(text)
            for ent in doc.ents:
                entities.append((ent.text, ent.type))
        elif lang == 'en' and en_nlp:
            # İngilizce NER işlemi
            doc = en_nlp(text)
            for ent in doc.ents:
                entities.append((ent.text, ent.label_))
    except Exception as e:
        print(f"Metin sınıflandırma hatası: {e}")
    
    return entities

# Dil tespiti yap (basit versiyon)
def detect_language(text):
    tr_chars = set('ğüşıöçĞÜŞİÖÇ')
    if any((c in tr_chars) for c in text):
        return 'tr'
    return 'en'

# Ana işlem fonksiyonu (OCR + NER)
def process_invoice(file_path):
    # 1. Metni çıkar (Türkçe+İngilizce)
    text = extract_text(file_path, lang='tur+eng')
    
    if not text:
        return "Metin çıkarılamadı", []
    
    # 2. Dil tespiti yap
    lang = detect_language(text)
    print(f"Tespit edilen dil: {lang}")
    
    # 3. Metni sınıflandır
    entities = classify_text(text, lang=lang)
    
    return text, entities

# Örnek kullanım
if __name__ == "__main__":
    # Test edilecek dosya yolu
    file_path = "turkce_fatura.jpg"  # veya .pdf
    
    # İşlemi çalıştır
    extracted_text, entities = process_invoice(file_path)
    
    # Sonuçları yazdır
    print("\n--- Çıkarılan Metin ---")
    print(extracted_text)
    
    print("\n--- Tanınan Varlıklar ---")
    for entity, label in entities:
        print(f"{label}: {entity}")