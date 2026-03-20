from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import zipfile
import pandas as pd
from ocr_ner import process_invoice
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/uploadfiles/")
async def upload_files(files: list[UploadFile] = File(...)):
    file_paths = []
    output_file_paths = []

    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        file_paths.append(file_path)

    for file_path in file_paths:
        processed_text, entities = process_invoice(file_path)

        # .txt çıktısı
        output_txt_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path) + "_output.txt")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("=== Extracted Text ===\n")
            f.write(processed_text + "\n\n")
            f.write("--- spaCy Entities ---\n")
            for entity, label in entities:
                f.write(f"{entity} - {label}\n")
        output_file_paths.append(output_txt_path)

        # .csv çıktısı
        csv_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file_path) + "_entities.csv")
        df = pd.DataFrame(entities, columns=["text", "label"])
        df.to_csv(csv_path, index=False)
        output_file_paths.append(csv_path)

    # Zip dosyası oluştur
    zip_filename = "output_files.zip"
    zip_filepath = os.path.join(OUTPUT_FOLDER, zip_filename)
    with zipfile.ZipFile(zip_filepath, 'w') as zipf:
        for file_path in output_file_paths:
            zipf.write(file_path, os.path.basename(file_path))

    print(f"Zip dosyası oluşturuldu: {zip_filepath}")

    # Geçici dosyaları sil (sadece inputlar)
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)

    return FileResponse(zip_filepath, media_type='application/zip', filename=zip_filename)
