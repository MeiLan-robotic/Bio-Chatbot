import os
import sys
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Đảm bảo in đúng tiếng Việt trên Console Windows PowerShell / CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


class ChromaSearcher:
    """
    Bộ tìm kiếm ngữ nghĩa đặc (Dense Search) sử dụng ChromaDB kết hợp
    mô hình nhúng đa ngôn ngữ intfloat/multilingual-e5-small.
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        self.persist_dir = persist_dir
        self.model_name = model_name
        
        print(f"[Dense] Đang khởi tạo ChromaDB client tại: {persist_dir}...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name="biology_sgk")

        print(f"[Dense] Đang tải mô hình embedding: {model_name}...")
        # SentenceTransformer sẽ tự động sử dụng CPU hoặc GPU nếu có
        self.embed_model = SentenceTransformer(model_name)
        print("[Dense] Khởi tạo mô hình embedding hoàn tất.")

    def index_chunks(self, chunks: list[dict], force_reload: bool = False):
        """
        Lập chỉ mục vector cho danh sách chunks vào ChromaDB.
        - Tuân thủ tiền tố 'passage: ' theo chuẩn của mô hình Multilingual-E5.
        - Tuần tự hóa trường metadata 'images' thành chuỗi JSON string vì ChromaDB
          chỉ hỗ trợ kiểu dữ liệu nguyên thủy (str, int, float, bool).
        """
        current_count = self.collection.count()
        if current_count > 0 and not force_reload:
            print(f"[Dense] Collection đã có {current_count} vector. Bỏ qua bước tạo lại (dùng cache).")
            return

        if force_reload and current_count > 0:
            print("[Dense] Thực hiện force_reload: Xóa sạch collection cũ...")
            self.client.delete_collection(name="biology_sgk")
            self.collection = self.client.create_collection(name="biology_sgk")

        print(f"[Dense] Bắt đầu tính toán embedding cho {len(chunks)} chunks...")
        documents = []
        metadatas = []
        ids = []

        # Chuẩn bị dữ liệu theo chuẩn Multilingual-E5 (cần tiền tố 'passage: ')
        passages_to_embed = []
        for idx, chunk in enumerate(chunks):
            content = chunk.get("page_content", "").strip()
            meta = chunk.get("metadata", {})

            # Xử lý metadata: convert list/dict images thành JSON string
            serialized_meta = {
                "trang": int(meta.get("trang", 0)),
                "muc": str(meta.get("muc", "Chung")),
                "has_image": bool(meta.get("has_image", False)),
                "images_json": json.dumps(meta.get("images", []), ensure_ascii=False),
                "chunk_id": idx,
            }

            passages_to_embed.append(f"passage: {content}")
            documents.append(content)
            metadatas.append(serialized_meta)
            ids.append(f"chunk_{idx}")

        # Tính toán embedding
        embeddings = self.embed_model.encode(
            passages_to_embed,
            normalize_embeddings=True,
            show_progress_bar=True
        ).tolist()

        # Thêm vào ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        print(f"[Dense] Đã lưu thành công {len(chunks)} vector vào ChromaDB!")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Tìm kiếm ngữ nghĩa các chunk gần nhất với câu hỏi.
        - Tuân thủ tiền tố 'query: ' theo chuẩn của Multilingual-E5.
        - Trả về danh sách chunks kèm 'distance' và 'chroma_rank'.
        """
        # Multilingual-E5 yêu cầu tiền tố 'query: ' cho câu truy vấn
        query_text = f"query: {query.strip()}"
        query_embedding = self.embed_model.encode(
            [query_text],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        formatted_results = []
        if not results or not results["ids"] or not results["ids"][0]:
            return formatted_results

        matched_docs = results["documents"][0]
        matched_metas = results["metadatas"][0]
        matched_distances = results["distances"][0]

        for rank, (doc, meta, dist) in enumerate(
            zip(matched_docs, matched_metas, matched_distances), start=1
        ):
            # Khôi phục trường images từ chuỗi JSON
            images = []
            if "images_json" in meta:
                try:
                    images = json.loads(meta["images_json"])
                except Exception:
                    images = []

            chunk_item = {
                "page_content": doc,
                "metadata": {
                    "trang": meta.get("trang"),
                    "muc": meta.get("muc"),
                    "has_image": meta.get("has_image"),
                    "images": images,
                    "chunk_id": meta.get("chunk_id"),
                },
                "chroma_distance": float(dist),
                "chroma_rank": rank,
            }
            formatted_results.append(chunk_item)

        return formatted_results


# =====================================================================
# CHẠY TEST ĐỘC LẬP KHI THỰC THI TRỰC TIẾP FILE
# =====================================================================
if __name__ == "__main__":
    DATA_PATH = os.path.join(
        os.path.dirname(__file__), "..", "biochat_preprocessing", "processed_data", "biology_chunks.json"
    )

    if not os.path.exists(DATA_PATH):
        print(f"[LỖI] Không tìm thấy file dữ liệu tại: {DATA_PATH}")
        sys.exit(1)

    print(f"[Dense] Đang đọc dữ liệu từ: {DATA_PATH}...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data_chunks = json.load(f)

    # 1. Khởi tạo ChromaSearcher
    chroma_searcher = ChromaSearcher(persist_dir="./chroma_db")

    # 2. Lập chỉ mục dữ liệu vào ChromaDB
    chroma_searcher.index_chunks(data_chunks)

    # 3. Câu hỏi kiểm thử về mặt ngữ nghĩa (không dùng từ khóa trực diện)
    test_query = "Cơ chế bảo tồn thông tin di truyền qua các thế hệ tế bào"
    print(f"\n[QUERY THỬ NGHIỆM]: '{test_query}'")
    print("=" * 70)

    # 4. Tìm kiếm top 2 chunks
    top_results = chroma_searcher.search(test_query, top_k=2)

    for res in top_results:
        trang = res["metadata"].get("trang", "N/A")
        dist = res["chroma_distance"]
        rank = res["chroma_rank"]
        imgs = res["metadata"].get("images", [])

        print(f"▶ HẠNG {rank} | Khoảng cách Vector (Distance): {dist:.4f} | Thuộc Trang: {trang}")
        print(f"  Trích đoạn: {res['page_content'][:220]}...")
        if res["metadata"].get("has_image", False):
            print(f"  Ảnh/Sơ đồ liên quan ({len(imgs)} hình):")
            for img in imgs:
                print(f"    - File: {img.get('image_path')}")
        print("-" * 70)
