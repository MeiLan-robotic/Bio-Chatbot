import os
import re
import sys
import json
from rank_bm25 import BM25Okapi

# Đảm bảo in đúng tiếng Việt trên Console Windows PowerShell / CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """
    Tách từ chuyên dụng cho môn Sinh học:
    - Chuẩn hóa chữ thường (lowercase)
    - Giữ lại các ký hiệu chiều mạch: 3', 5', 3'-5', 5'-3'
    - Bắt chính xác tên enzyme và thuật ngữ: Okazaki, helicase, ligase, DNA, RNA, chạc Y...
    """
    text = text.lower()
    # Regex nhận diện từ ghép, chiều mạch có dấu nháy đơn ' và gạch nối
    pattern = r"\b[a-z0-9à-ỹ]+(?:['\-\/][a-z0-9à-ỹ]+)*\b|\d+'"
    tokens = re.findall(pattern, text)
    return tokens


class BM25Searcher:
    """
    Bộ tìm kiếm từ khóa thưa (Sparse Search) sử dụng thuật toán BM25Okapi.
    Tối ưu hóa cho truy vấn thuật ngữ chuyên ngành môn Sinh học.
    """

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: list[dict] = []

    def fit(self, chunks: list[dict]):
        """
        Lập chỉ mục BM25 từ danh sách chunks.
        Trích xuất trường 'page_content' để phân tích từ khóa.
        """
        if not chunks:
            raise ValueError("[BM25] Danh sách chunks rỗng!")

        self.chunks = chunks
        corpus = [chunk["page_content"] for chunk in chunks]
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"[BM25] Đã lập chỉ mục thành công cho {len(chunks)} chunks.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Tìm kiếm các chunk có độ khớp từ khóa cao nhất với câu hỏi.
        Trả về top_k chunks kèm theo 'bm25_score' và 'bm25_rank'.
        """
        if not self.bm25 or not self.chunks:
            raise ValueError("[BM25] Chưa có dữ liệu! Hãy gọi fit(chunks) trước khi search.")

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        # Sắp xếp các chỉ số chunk theo thứ tự điểm giảm dần
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            if scores[idx] > 0:  # Chỉ giữ lại chunk có chứa ít nhất 1 từ khóa khớp
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["bm25_score"] = float(scores[idx])
                chunk_copy["bm25_rank"] = rank
                results.append(chunk_copy)

        return results


# =====================================================================
# CHẠY TEST ĐỘC LẬP KHI THỰC THI TRỰC TIẾP FILE
# =====================================================================
if __name__ == "__main__":
    # Đường dẫn tới file dữ liệu tiền xử lý từ Giai đoạn 1
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "biochat_preprocessing", "processed_data", "biology_chunks.json"
    )

    if not os.path.exists(DATA_PATH):
        print(f"[LỖI] Không tìm thấy file dữ liệu tại: {DATA_PATH}")
        sys.exit(1)

    print(f"[BM25] Đang nạp dữ liệu từ: {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_chunks = json.load(f)

    # 1. Khởi tạo và lập chỉ mục
    searcher = BM25Searcher()
    searcher.fit(data_chunks)

    # 2. Câu hỏi kiểm thử thực tế
    test_query = "Enzyme helicase và ligase có vai trò gì trong tái bản DNA?"
    print(f"\n[QUERY]: '{test_query}'")
    print("=" * 70)

    # 3. Thực hiện tìm kiếm
    top_results = searcher.search(test_query, top_k=2)

    if not top_results:
        print("[!] Không tìm thấy đoạn nào chứa từ khóa phù hợp.")
    else:
        for res in top_results:
            trang = res["metadata"].get("trang", "N/A")
            score = res["bm25_score"]
            rank = res["bm25_rank"]
            has_img = res["metadata"].get("has_image", False)
            imgs = res["metadata"].get("images", [])

            print(f"▶ HẠNG {rank} | Điểm BM25: {score:.4f} | Thuộc Trang: {trang}")
            print(f"  Trích đoạn: {res['page_content'][:220]}...")
            if has_img:
                print(f"  Ảnh/Sơ đồ liên quan ({len(imgs)} hình):")
                for img in imgs:
                    print(f"    - File: {img.get('image_path')}")
            print("-" * 70)
