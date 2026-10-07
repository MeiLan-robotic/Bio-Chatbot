import os
import sys

# Đảm bảo in tiếng Việt chuẩn trên Console Windows PowerShell / CMD
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from hybrid_retriever import HybridRetriever


def run_evaluation_suite():
    print("=" * 80)
    print("        BỘ KIỂM THỬ ĐÁNH GIÁ NĂNG LỰC TRUY XUẤT - BIOCHAT RETRIEVAL")
    print("=" * 80)

    # 1. Khởi tạo Retriever
    print("\n[INIT] Đang khởi động Hybrid Retriever...")
    retriever = HybridRetriever()
    print("[INIT] Khởi động thành công! Bắt đầu kiểm thử các kịch bản...\n")

    # 2. Định nghĩa danh sách các bài test case
    test_cases = [
        {
            "id": "TC-01",
            "type": "Keyword-Heavy (Từ khóa chuyên sâu)",
            "query": "Enzyme ligase có vai trò gì trong việc nối các đoạn Okazaki?",
            "target": "Kiểm tra độ nhạy BM25 với các thuật ngữ: ligase, Okazaki",
        },
        {
            "id": "TC-02",
            "type": "Semantic-Heavy (Ngữ nghĩa & Khái niệm)",
            "query": "Nguyên tắc bán bảo toàn trong quá trình nhân đôi ADN là gì?",
            "target": "Kiểm tra Dense Search hiểu khái niệm bán bảo toàn, truyền đạt thông tin",
        },
        {
            "id": "TC-03",
            "type": "Diagram-Focused (Truy vấn sơ đồ & hình ảnh)",
            "query": "Sơ đồ chạc sao chép chữ Y và chiều tổng hợp mạch mới 5' đến 3'",
            "target": "Kiểm tra khả năng bắt trúng sơ đồ/hình ảnh mô tả chạc chữ Y",
        },
    ]

    # 3. Lặp và đánh giá từng test case
    for idx, tc in enumerate(test_cases, start=1):
        print("+" + "-" * 78 + "+")
        print(f"| TEST CASE {idx}: [{tc['id']}] - {tc['type']}")
        print(f"| Mục tiêu: {tc['target']}")
        print(f"| Câu hỏi : \"{tc['query']}\"")
        print("+" + "-" * 78 + "+")

        results = retriever.retrieve(tc["query"], top_k_each=5, top_n=2)

        if not results:
            print("  ❌ [THẤT BẠI] Không tìm thấy kết quả phù hợp nào.\n")
            continue

        for res in results:
            rank = res.get("rrf_rank", 1)
            score = res.get("rrf_score", 0.0)
            meta = res.get("metadata", {})
            trang = meta.get("trang", "Không xác định")
            muc = meta.get("muc", "Chung")
            details = res.get("retrieval_details", {})
            has_image = meta.get("has_image", False)
            images = meta.get("images", [])

            bm25_rank = details.get("bm25_rank", "N/A")
            chroma_rank = details.get("chroma_rank", "N/A")

            print(f"\n  🎯 [TOP {rank}] -> Điểm RRF: {score:.5f} | SGK Trang {trang} ({muc})")
            print(f"     ├── Nguồn xếp hạng : BM25 Hạng {bm25_rank} | Chroma Hạng {chroma_rank}")

            # Rút gọn nội dung hiển thị sạch đẹp
            content_preview = res["page_content"].replace("\n", " ")
            if len(content_preview) > 220:
                content_preview = content_preview[:220] + "..."
            print(f"     ├── Trích đoạn nội dung: {content_preview}")

            # Thông tin hình ảnh / sơ đồ đính kèm
            if has_image and images:
                print(f"     └── 🖼️  Dữ liệu hình ảnh đính kèm ({len(images)} sơ đồ/ảnh):")
                for img_idx, img in enumerate(images, start=1):
                    cap = img.get("caption", "Không có caption").replace("\n", " ")
                    if len(cap) > 100:
                        cap = cap[:100] + "..."
                    print(f"         [{img_idx}] Đường dẫn: {img.get('image_path')}")
                    print(f"             Caption : {cap}")
            else:
                print("     └── (Không có sơ đồ đính kèm)")

        print("\n" + "=" * 80 + "\n")

    print("[KẾT THÚC] Toàn bộ kịch bản kiểm thử truy xuất đã hoàn tất!")


if __name__ == "__main__":
    run_evaluation_suite()
