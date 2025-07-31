import os
import openai
import base64
import csv
from PIL import Image
from Levenshtein import distance as levenshtein_distance

# === Setup LM Studio ===
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "lm-studio"  # Gak perlu autentikasi beneran

# === Encode gambar ke base64 ===
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# === Hitung Character Error Rate (CER) ===
def calculate_cer(predicted, truth):
    dist = levenshtein_distance(predicted, truth)
    return dist / len(truth) if len(truth) > 0 else 1.0

# === Kirim OCR ke LM Studio ===
def ocr_lmstudio(image_path):
    encoded_image = encode_image(image_path)
    response = openai.ChatCompletion.create(
        model="llava",  # Bisa diganti sesuai model yang kamu load
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the license plate number shown in this image? Respond only with the plate number."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content'].strip()

# === Jalankan proses OCR untuk semua gambar ===
def run_ocr(dataset_root, output_file):
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "ground_truth", "prediction", "CER_score"])

        for subdir, dirs, files in os.walk(dataset_root):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_path = os.path.join(subdir, file)
                    txt_path = os.path.splitext(image_path)[0] + ".txt"

                    if not os.path.exists(txt_path):
                        continue

                    with open(txt_path, "r") as f:
                        ground_truth = f.read().strip()

                    try:
                        prediction = ocr_lmstudio(image_path)
                        cer = calculate_cer(prediction, ground_truth)

                        # === Output terminal gaya kamu ===
                        print(f"➜ Memproses {file}...")
                        print(f"   ➜ Ground Truth : {ground_truth}")
                        print(f"   ➜ Prediksi     : {prediction}")
                        print(f"   ➜ CER          : {cer:.4f}")
                        print("-" * 50)

                        writer.writerow([file, ground_truth, prediction, cer])

                    except Exception as e:
                        print(f"[ERROR] {file} gagal diproses: {e}")

# === Main eksekusi ===
if __name__ == "__main__":
    dataset_folder = "test"         # Ganti jika lokasi folder beda
    output_csv = "results.csv"
    run_ocr(dataset_folder, output_csv)
    print("✅ Proses selesai. Hasil disimpan ke results.csv")
