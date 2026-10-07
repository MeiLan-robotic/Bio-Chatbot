import os
import re
import sys
import json
import time
import pymupdf as fitz  # PyMuPDF (tên mới thay cho import fitz)
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
# Fix encoding trên Windows CMD / PowerShell
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# =====================================================================
# CẤU HÌNH
# =====================================================================
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
GEMINI_MODEL   = "gemini-3.8-flash"   # model duy nhất hoạt động với API key này

PDF_PATH    = "data_raw/chuong1_ditruyenphantu.pdf"
IMAGE_DIR   = "static/images"
OUTPUT_JSON = "processed_data/biology_chunks.json"

# Kích thước tối thiểu để coi là sơ đồ/hình thật sự (px)
MIN_IMG_WIDTH  = 150
MIN_IMG_HEIGHT = 150

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)


# =====================================================================
# EXCEPTION RIÊNG CHO HẾT QUOTA NGÀY
# =====================================================================
class QuotaExhaustedError(Exception):
    """Raise khi API trả 429 RESOURCE_EXHAUSTED – hết quota ngày.
    Khác với 503 (quá tải tạm thời), lỗi này không thể giải quyết bằng retry
    ngắn → toàn bộ việc sinh caption sẽ bị tắt để tiết kiệm thời gian.
    """
    pass



# =====================================================================
# BƯỚC 1 ‑ LÀM SẠCH & PHÂN MỤC VĂN BẢN THUẦN
# (tương ứng nhánh trên của sơ đồ: "Văn bản thuần → Làm sạch & phân mục")
# =====================================================================
# Tiêu đề thường gặp trong SGK Sinh học 12:
HEADING_PATTERNS = [
    r"^(CHƯƠNG\s+\w+)",
    r"^(Bài\s+\d+[\.\:])",
    r"^(I{1,3}|IV|V|VI{0,3}|IX|X)\.\s+",   # La mã: I. II. III. ...
    r"^(\d+\.\s+[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ])",  # 1. Tên mục viết hoa
]

def clean_text(raw: str) -> str:
    """
    Làm sạch văn bản thô từ PyMuPDF:
    - Xoá ký tự đặc biệt rác (hyphen cuối dòng, dấu lạ)
    - Nối lại từ bị ngắt dòng giữa chừng
    - Chuẩn hoá khoảng trắng
    """
    # Nối dòng bị ngắt giữa từ (dấu '-' cuối dòng)
    text = re.sub(r"-\n(?=[a-záàảãạăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ])", "", raw)
    # Thay nhiều newline liên tiếp bằng 1 dòng trống
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Xoá khoảng trắng đầu dòng thừa
    text = re.sub(r"[ \t]+\n", "\n", text)
    # Xoá ký tự điều khiển lạ
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


def detect_section(line: str) -> str | None:
    """
    Phát hiện dòng là tiêu đề mục → trả về tên mục, ngược lại trả None.
    """
    line = line.strip()
    for pat in HEADING_PATTERNS:
        if re.match(pat, line, re.IGNORECASE):
            return line
    return None


def split_into_sections(text: str) -> list[dict]:
    """
    Phân mục văn bản thành các đoạn có cấu trúc:
    [{"section": "Tên mục", "content": "nội dung..."}, ...]
    Nếu không có tiêu đề nào thì trả về 1 chunk duy nhất.
    """
    sections = []
    current_heading = "Nội dung chung"
    current_lines = []

    for line in text.splitlines():
        heading = detect_section(line)
        if heading:
            # Lưu đoạn trước
            if current_lines:
                sections.append({
                    "section": current_heading,
                    "content": "\n".join(current_lines).strip()
                })
            current_heading = heading
            current_lines = []
        else:
            current_lines.append(line)

    # Đoạn cuối
    if current_lines:
        sections.append({
            "section": current_heading,
            "content": "\n".join(current_lines).strip()
        })

    return [s for s in sections if s["content"]]


# =====================================================================
# BƯỚC 2 ‑ SINH CAPTION CHO ẢNH / SƠ ĐỒ BẰNG GEMINI (VLM)
# (tương ứng nhánh dưới: "Hình ảnh & Sơ đồ → VLM sinh Caption")
# =====================================================================
def generate_image_caption(image_bytes: bytes,
                           max_retries: int = 3,
                           base_delay: float = 5.0) -> str:
    """
    Gửi ảnh lên Gemini (VLM) và nhận mô tả ngữ nghĩa Sinh học.

    Phân biệt 2 loại lỗi:
    - 503 UNAVAILABLE (quá tải tạm thời) → retry với exponential backoff (5s, 10s, 20s)
    - 429 RESOURCE_EXHAUSTED hết quota ngày → raise QuotaExhaustedError ngay, không retry
    - Lỗi khác (404, 400...) → trả fallback ngay
    """
    prompt = (
        "Bạn là chuyên gia Sinh học. "
        "Hãy mô tả chi tiết sơ đồ/hình ảnh này bằng tiếng Việt: "
        "nêu tên sơ đồ (nếu có), các thành phần chú thích chính (enzim, "
        "phân tử, chiều mạch, sản phẩm...), và các bước cơ chế diễn ra. "
        "Viết từ 3 đến 5 câu, ngắn gọn, phù hợp làm ngữ cảnh cho hệ thống RAG."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
            )
            caption = response.text.strip()
            if not caption:
                raise ValueError("Gemini trả về chuỗi rỗng")
            return caption

        except Exception as e:
            err_str = str(e)

            # ── Hết quota ngày: không retry, báo lỗi nghiêm trọng để dừng hẳn
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                print(f"    [!] HẾT QUOTA NGÀY – không retry. Bỏ qua caption ảnh này.")
                raise QuotaExhaustedError(err_str)

            # ── Quá tải tạm thời 503: retry với backoff
            if ("503" in err_str or "UNAVAILABLE" in err_str) and attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))   # 5 → 10 → 20
                print(f"    [!] 503 tạm thời. Retry {attempt}/{max_retries} sau {delay:.0f}s...")
                time.sleep(delay)
                continue

            # ── Lỗi khác (404, 400, v.v.): trả fallback
            print(f"    [!] Lỗi Gemini caption: {err_str[:120]}")
            return "[Caption lỗi – cần xem lại API key hoặc ảnh không hợp lệ]"



# =====================================================================
# BƯỚC 3 ‑ ĐÓNG GÓI THÀNH TEXT CHUNKS + METADATA (IMAGE)
# (đầu ra cuối cùng của sơ đồ: "Text Chunks + Metadata (Image)")
# =====================================================================
def build_chunk(page_num: int,
                section: str,
                content: str,
                images_metadata: list[dict]) -> dict:
    """
    Tạo 1 chunk hoàn chỉnh gồm:
      page_content : văn bản thuần + mô tả ảnh ghép vào (để BM25 & embed tìm được)
      metadata     : thông tin trang, mục, đường dẫn ảnh
    """
    page_content = content

    # Ghép caption ảnh vào text để retrieval có thể tìm thấy qua nội dung ảnh
    if images_metadata:
        page_content += "\n\n[HÌNH ẢNH & SƠ ĐỒ TRONG TRANG]:\n"
        for item in images_metadata:
            page_content += f"- {item['image_path']}: {item['caption']}\n"

    return {
        "page_content": page_content.strip(),
        "metadata": {
            "trang"    : page_num,
            "muc"      : section,
            "has_image": len(images_metadata) > 0,
            "images"   : images_metadata,   # [{image_path, caption}]
        },
    }


# =====================================================================
# HÀM CHÍNH – CHẠY TOÀN BỘ PIPELINE
# =====================================================================
def process_pdf(max_pages: int | None = None):
    """
    Pipeline tiền xử lý SGK PDF theo sơ đồ:
      PDF SGK
        → Phân tách Layout (PyMuPDF)
            → Văn bản thuần  → Làm sạch & Phân mục
            → Hình ảnh & Sơ đồ → VLM sinh Caption
        → Text Chunks + Metadata(Image)
        → Lưu JSON

    Tham số:
        max_pages : giới hạn số trang (None = xử lý hết). Dùng max_pages=5 khi test.
    """
    if not os.path.exists(PDF_PATH):
        print(f"[LỖI] Không tìm thấy file: {PDF_PATH}")
        return

    doc = fitz.open(PDF_PATH)
    total = len(doc)
    limit = min(max_pages, total) if max_pages else total
    print(f"[INFO] Mở '{PDF_PATH}' – {total} trang. Sẽ xử lý {limit} trang.\n")

    all_chunks: list[dict] = []
    quota_exhausted = False   # True khi hết 20 req/ngày → bỏ qua gọi Gemini

    for page_index in range(limit):
        page    = doc[page_index]
        page_num = page_index + 1
        print(f"── Trang {page_num}/{limit} ──────────────────────────")

        # ------------------------------------------------------------------
        # A. PHÂN TÁCH LAYOUT: lấy text theo blocks để tránh lỗi dính cột
        # ------------------------------------------------------------------
        blocks = page.get_text("blocks")   # [(x0,y0,x1,y1, text, block_no, block_type)]
        raw_text = ""
        for b in sorted(blocks, key=lambda b: (b[1], b[0])):  # sắp theo top→bottom, left→right
            txt = b[4].strip()
            if txt:
                raw_text += txt + "\n"

        # ------------------------------------------------------------------
        # B. NHÁNH VĂN BẢN: Làm sạch → Phân mục
        # ------------------------------------------------------------------
        clean = clean_text(raw_text)
        sections = split_into_sections(clean)
        print(f"  Text: {len(clean)} ký tự  |  {len(sections)} đoạn/mục")

        # ------------------------------------------------------------------
        # C. NHÁNH HÌNH ẢNH: Trích xuất → VLM sinh Caption
        # ------------------------------------------------------------------
        image_list = page.get_images(full=True)
        images_metadata: list[dict] = []

        for img_idx, img_info in enumerate(image_list):
            xref       = img_info[0]
            base_image = doc.extract_image(xref)
            img_bytes  = base_image["image"]
            width      = base_image["width"]
            height     = base_image["height"]

            if width < MIN_IMG_WIDTH or height < MIN_IMG_HEIGHT:
                continue  # bỏ qua icon/logo nhỏ

            img_filename = f"trang_{page_num}_hinh_{img_idx + 1}.png"
            img_path     = os.path.join(IMAGE_DIR, img_filename)

            with open(img_path, "wb") as f_img:
                f_img.write(img_bytes)

            print(f"  Ảnh [{img_idx+1}]: {img_filename} ({width}x{height})", end="")

            if quota_exhausted:
                caption = "[Bỏ qua – hết quota API ngày hôm nay]"
                print(" → bỏ qua (quota hết)")
            else:
                print(" → gọi Gemini...")
                try:
                    caption = generate_image_caption(img_bytes)
                    print(f"    Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}")
                except QuotaExhaustedError:
                    quota_exhausted = True
                    caption = "[Bỏ qua – hết quota API ngày hôm nay]"
                    print("    [!] Đã hết quota – tắt Gemini cho các ảnh còn lại trong lần chạy này.")
                    print("    [!] Các ảnh vẫn được lưu ra đĩa. Chạy lại vào ngày mai để sinh caption.")


            images_metadata.append({
                "image_path": img_path.replace("\\", "/"),
                "caption"   : caption,
            })

        # ------------------------------------------------------------------
        # D. ĐÓNG GÓI CHUNKS
        # Mỗi "mục" trong trang → 1 chunk riêng; ảnh gắn vào chunk cuối
        # (hoặc chunk duy nhất nếu trang không có tiêu đề mục)
        # ------------------------------------------------------------------
        for s_idx, sec in enumerate(sections):
            # Chỉ gắn ảnh vào chunk CUỐI của trang (ảnh thuộc về trang đó)
            imgs = images_metadata if s_idx == len(sections) - 1 else []
            chunk = build_chunk(page_num, sec["section"], sec["content"], imgs)
            all_chunks.append(chunk)
            print(f"  ✓ Chunk: [{sec['section'][:40]}] – {len(chunk['page_content'])} ký tự")

        # Trang không có text (trang ảnh thuần) → vẫn tạo chunk cho ảnh
        if not sections and images_metadata:
            chunk = build_chunk(page_num, "Hình ảnh & Sơ đồ", "", images_metadata)
            all_chunks.append(chunk)

    # ------------------------------------------------------------------
    # E. LƯU KẾT QUẢ RA FILE JSON
    # ------------------------------------------------------------------
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[THÀNH CÔNG] Xuất {len(all_chunks)} chunks → {OUTPUT_JSON}")


# =====================================================================
if __name__ == "__main__":
    # Đổi max_pages=None để chạy toàn bộ PDF
    process_pdf(max_pages=5)