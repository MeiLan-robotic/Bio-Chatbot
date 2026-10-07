import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Đảm bảo in tiếng Việt chuẩn trên Windows PowerShell / CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# 1. Thêm đường dẫn module biochat_retrieval vào sys.path để import HybridRetriever
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RETRIEVAL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "biochat_retrieval"))
if RETRIEVAL_DIR not in sys.path:
    sys.path.insert(0, RETRIEVAL_DIR)

from hybrid_retriever import HybridRetriever
from google import genai

# =====================================================================
# CẤU HÌNH GEMINI API
# =====================================================================
# Dùng API Key đã cấu hình từ giai đoạn 1
api_key = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def build_rag_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Xây dựng prompt chuẩn hóa theo kỹ thuật RAG:
    - Bổ sung chỉ thị chống ảo giác (Anti-hallucination).
    - Cung cấp dữ liệu ngữ cảnh trích xuất từ SGK kèm số trang.
    """
    context_sections = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        meta = chunk.get("metadata", {})
        trang = meta.get("trang", "Không xác định")
        content = chunk.get("page_content", "").strip()
        context_sections.append(
            f"--- [TRÍCH ĐOẠN {idx} | SGK TRANG {trang}] ---\n{content}"
        )

    context_str = "\n\n".join(context_sections)

    prompt = f"""Bạn là Gia sư Sinh học 12 thông thái, chuẩn mực và tận tâm.
Nhiệm vụ của bạn là giải đáp thắc mắc của học sinh bằng cách DỰA HOÀN TOÀN vào tài liệu Sách giáo khoa (SGK) được cung cấp dưới đây.

[QUY TẮC BẮT BUỘC]:
1. CHỈ sử dụng thông tin có trong mục [NGỮ CẢNH TỪ SÁCH GIÁO KHOA]. Tuyệt đối KHÔNG tự bịa đặt hoặc suy diễn kiến thức ngoài tài liệu.
2. Nếu câu hỏi KHÔNG có câu trả lời trong ngữ cảnh, hãy thông báo lịch sự: "Rất tiếc, nội dung Sách giáo khoa trong phạm vi này chưa đề cập đến câu hỏi của bạn."
3. Mọi câu trả lời PHẢI trích dẫn rõ ràng xuất xứ ở cuối câu hoặc đoạn (Ví dụ: "Theo SGK Sinh học 12, trang X...").
4. Nếu tài liệu có nhắc đến sơ đồ hoặc hình ảnh minh họa (ví dụ: Hình 1.2, Hình 1.3), hãy hướng dẫn học sinh quan sát sơ đồ đó để hiểu sâu hơn.
5. Trình bày khoa học: dùng gạch đầu dòng, in đậm các thuật ngữ quan trọng (enzyme, chiều mạch 5' -> 3', Okazaki...).

[NGỮ CẢNH TỪ SÁCH GIÁO KHOA]:
{context_str}

[CÂU HỎI CỦA HỌC SINH]:
{query}

[CÂU TRẢ LỜI CỦA GIA SƯ]:"""

    return prompt


class BioChatPipeline:
    """
    Pipeline tích hợp toàn diện:
    HybridRetriever (Giai đoạn 2) -> RAG Prompt -> Gemini API (gemini-3.8-flash)
    """

    def __init__(self):
        print("[Pipeline] Khởi động Gemini Client...")
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        print("[Pipeline] Khởi động Hybrid Retriever...")
        # Đường dẫn tới ChromaDB cục bộ
        persist_dir = os.path.join(RETRIEVAL_DIR, "chroma_db")
        json_path = os.path.join(
            CURRENT_DIR,
            "..",
            "biochat_preprocessing",
            "processed_data",
            "biology_chunks.json",
        )
        self.retriever = HybridRetriever(
            json_path=json_path, persist_dir=persist_dir
        )
        print("[Pipeline] BioChat Pipeline đã sẵn sàng phục vụ!\n")

    def ask(self, query: str, top_n: int = 2) -> dict:
        """
        Nhận câu hỏi từ học sinh, trả về:
        - answer: Câu trả lời chi tiết chuẩn SGK
        - sources: Danh sách các trang SGK liên quan
        - images: Danh sách file ảnh/sơ đồ và caption
        - retrieved_chunks: Toàn bộ dữ liệu thô đã truy xuất
        """
        # Bước 1: Truy xuất tài liệu liên quan bằng Hybrid RRF
        retrieved_chunks = self.retriever.retrieve(query, top_n=top_n)

        if not retrieved_chunks:
            return {
                "answer": "Không tìm thấy tài liệu phù hợp trong cơ sở tri thức.",
                "sources": [],
                "images": [],
                "retrieved_chunks": [],
            }

        # Bước 2: Thu thập metadata (trang, danh sách ảnh)
        sources = []
        all_images = []
        seen_images = set()

        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            trang = meta.get("trang")
            if trang and trang not in sources:
                sources.append(trang)

            for img in meta.get("images", []):
                img_path = img.get("image_path")
                if img_path and img_path not in seen_images:
                    seen_images.add(img_path)
                    all_images.append(img)

        # Bước 3: Tạo Prompt RAG và gọi Gemini (có Retry tự động cho 503)
        rag_prompt = build_rag_prompt(query, retrieved_chunks)
        answer_text = ""

        import time
        candidate_models = [GEMINI_MODEL, "gemini-3.5-flash-lite", "gemini-3.8-flash"]
        # Loại bỏ trùng lặp giữ nguyên thứ tự
        unique_models = list(dict.fromkeys(candidate_models))
        
        for model_to_use in unique_models:
            try:
                response = self.client.models.generate_content(
                    model=model_to_use,
                    contents=rag_prompt,
                )
                answer_text = response.text.strip()
                break
            except Exception as e:
                err_str = str(e)
                print(f"[!] Lỗi khi gọi Gemini ({model_to_use}): {e}")
                if "503" in err_str or "UNAVAILABLE" in err_str or "404" in err_str:
                    time.sleep(1.0)
                    continue
                else:
                    answer_text = f"Đã xảy ra lỗi khi kết nối với AI: {e}"
                    break
        else:
            if not answer_text:
                answer_text = "Hệ thống AI hiện đang bận do lượng truy cập cao. Vui lòng thử lại sau vài giây."

        return {
            "answer": answer_text,
            "sources": sorted(sources),
            "images": all_images,
            "retrieved_chunks": retrieved_chunks,
        }


# =====================================================================
# CHẠY TEST ĐỘC LẬP KHI THỰC THI TRỰC TIẾP FILE
# =====================================================================
if __name__ == "__main__":
    pipeline = BioChatPipeline()

    test_question = "Cơ chế nhân đôi DNA tại chạc sao chép chữ Y diễn ra như thế nào? Chiều tổng hợp mạch mới là gì?"
    print(f"[CÂU HỎI THỬ NGHIỆM]: {test_question}")
    print("=" * 75)

    result = pipeline.ask(test_question, top_n=2)

    print("\n💡 [CÂU TRẢ LỜI CỦA BIOCHAT]:")
    print(result["answer"])

    print("\n📚 [NGUỒN TRÍCH DẪN SGK]:", f"Trang {', '.join(map(str, result['sources']))}")

    if result["images"]:
        print(f"\n🖼️ [HÌNH ẢNH / SƠ ĐỒ ĐÍNH KÈM ({len(result['images'])} ảnh)]:")
        for idx, img in enumerate(result["images"], start=1):
            print(f"  ({idx}) File: {img.get('image_path')}")
            print(f"      Mô tả sơ đồ: {img.get('caption')[:120]}...")
    print("=" * 75)
