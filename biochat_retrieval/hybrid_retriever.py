import os
import sys
import json
from collections import defaultdict

# Đảm bảo in đúng tiếng Việt trên Console Windows PowerShell / CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from bm25_indexer import BM25Searcher
from chroma_indexer import ChromaSearcher


class HybridRetriever:
    """
    Bộ truy xuất lai (Hybrid Retriever) kết hợp:
    1. Sparse Search: BM25Okapi (bắt từ khóa chuyên ngành, ký hiệu khoa học)
    2. Dense Search: ChromaDB + Multilingual-E5 (bắt ngữ nghĩa câu hỏi tự nhiên)
    3. Thuật toán tái xếp hạng: Reciprocal Rank Fusion (RRF, k=60)
    """

    def __init__(
        self,
        json_path: str | None = None,
        persist_dir: str = "./chroma_db",
        model_name: str = "intfloat/multilingual-e5-small",
    ):
        # Xác định đường dẫn file chunks nếu không truyền vào
        if json_path is None:
            json_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "biochat_preprocessing",
                "processed_data",
                "biology_chunks.json",
            )

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[Hybrid] Không tìm thấy file dữ liệu: {json_path}")

        print(f"[Hybrid] Đang nạp cơ sở tri thức từ: {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Gán chunk_id định danh duy nhất cho từng chunk nếu chưa có
        for idx, chunk in enumerate(self.chunks):
            if "metadata" not in chunk:
                chunk["metadata"] = {}
            if "chunk_id" not in chunk["metadata"]:
                chunk["metadata"]["chunk_id"] = idx

        # Khởi tạo hai bộ tìm kiếm
        print("\n--- [1/2] Khởi tạo Sparse Engine (BM25) ---")
        self.bm25_searcher = BM25Searcher()
        self.bm25_searcher.fit(self.chunks)

        print("\n--- [2/2] Khởi tạo Dense Engine (ChromaDB) ---")
        self.chroma_searcher = ChromaSearcher(
            persist_dir=persist_dir, model_name=model_name
        )
        self.chroma_searcher.index_chunks(self.chunks)
        print("[Hybrid] Hệ thống Hybrid Retriever đã sẵn sàng hoạt động!\n")

    def rrf_fusion(
        self,
        bm25_results: list[dict],
        dense_results: list[dict],
        k: int = 60,
        top_n: int = 3,
    ) -> list[dict]:
        """
        Thuật toán Reciprocal Rank Fusion (RRF):
        RRF_Score(d) = sum( 1 / (k + rank_i(d)) ) với k=60 theo chuẩn quốc tế.
        """
        rrf_scores = defaultdict(float)
        chunk_map = {}
        source_ranks = defaultdict(dict)

        # 1. Tính điểm từ danh sách BM25
        for res in bm25_results:
            cid = res["metadata"]["chunk_id"]
            rank = res.get("bm25_rank", 999)
            rrf_scores[cid] += 1.0 / (k + rank)
            chunk_map[cid] = res
            source_ranks[cid]["bm25_rank"] = rank
            source_ranks[cid]["bm25_score"] = res.get("bm25_score", 0.0)

        # 2. Tính điểm từ danh sách Dense (ChromaDB)
        for res in dense_results:
            cid = res["metadata"]["chunk_id"]
            rank = res.get("chroma_rank", 999)
            rrf_scores[cid] += 1.0 / (k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = res
            source_ranks[cid]["chroma_rank"] = rank
            source_ranks[cid]["chroma_distance"] = res.get("chroma_distance", 999.0)

        # 3. Sắp xếp các chunk theo điểm RRF giảm dần
        sorted_chunk_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )[:top_n]

        # 4. Đóng gói kết quả cuối cùng
        fused_results = []
        for final_rank, cid in enumerate(sorted_chunk_ids, start=1):
            item = dict(chunk_map[cid])
            item["rrf_score"] = rrf_scores[cid]
            item["rrf_rank"] = final_rank
            item["retrieval_details"] = source_ranks[cid]
            fused_results.append(item)

        return fused_results

    def retrieve(
        self, query: str, top_k_each: int = 5, top_n: int = 3
    ) -> list[dict]:
        """
        Quy trình Hybrid Retrieval:
        1. Gọi BM25 lấy top_k_each
        2. Gọi ChromaDB lấy top_k_each
        3. Áp dụng RRF Fusion (k=60) để tái xếp hạng và trả về top_n
        """
        # 1. Sparse Search
        bm25_res = self.bm25_searcher.search(query, top_k=top_k_each)

        # 2. Dense Search
        dense_res = self.chroma_searcher.search(query, top_k=top_k_each)

        # 3. RRF Fusion
        fused_res = self.rrf_fusion(bm25_res, dense_res, k=60, top_n=top_n)

        return fused_res


# =====================================================================
# CHẠY TEST ĐỘC LẬP KHI THỰC THI TRỰC TIẾP FILE
# =====================================================================
if __name__ == "__main__":
    retriever = HybridRetriever()

    test_query = "Cấu trúc xoắn kép của phân tử DNA và các liên kết hóa học giữa hai mạch"
    print(f"[QUERY KIỂM THỬ HYBRID]: '{test_query}'")
    print("=" * 75)

    final_chunks = retriever.retrieve(test_query, top_k_each=5, top_n=3)

    for res in final_chunks:
        trang = res["metadata"].get("trang", "N/A")
        score = res["rrf_score"]
        rank = res["rrf_rank"]
        details = res["retrieval_details"]
        imgs = res["metadata"].get("images", [])

        bm25_info = f"Rank {details.get('bm25_rank', 'None')} (Điểm: {details.get('bm25_score', 0):.2f})"
        dense_info = f"Rank {details.get('chroma_rank', 'None')} (Dist: {details.get('chroma_distance', 0):.3f})"

        print(f"🏆 HẠNG {rank} (RRF Score: {score:.5f}) | SGK Trang: {trang}")
        print(f"   📊 Nguồn đóng góp: BM25 -> {bm25_info} | Chroma -> {dense_info}")
        print(f"   📖 Trích đoạn: {res['page_content'][:230]}...")

        if res["metadata"].get("has_image", False):
            print(f"   🖼️  Ảnh / Sơ đồ đính kèm ({len(imgs)} ảnh):")
            for img in imgs:
                print(f"      - {img.get('image_path')}")
        print("-" * 75)
