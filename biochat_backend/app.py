import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Đảm bảo in tiếng Việt chuẩn trên Windows Console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from rag_pipeline import BioChatPipeline

# =====================================================================
# KHỞI TẠO FASTAPI APP
# =====================================================================
app = FastAPI(
    title="BioChat - Sinh Học 12 RAG API",
    version="1.0.0",
    description="Hệ thống hỏi đáp Sách giáo khoa Sinh học 12 ứng dụng Hybrid RAG (BM25 + ChromaDB + Multilingual-E5) & Gemini AI",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Cấu hình CORS Middleware để Frontend gọi sang mà không bị chặn
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount thư mục tĩnh phục vụ xem ảnh/sơ đồ trực tiếp qua URL
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "biochat_preprocessing", "static", "images")
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    print(f"[FastAPI] Đã mount thư mục static tại: {STATIC_DIR}")

if os.path.exists(STATIC_IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=STATIC_IMAGES_DIR), name="images")
    print(f"[FastAPI] Đã mount thư mục ảnh tĩnh tại: {STATIC_IMAGES_DIR}")
else:
    print(f"[!] Cảnh báo: Không tìm thấy thư mục ảnh tại: {STATIC_IMAGES_DIR}")

# =====================================================================
# PYDANTIC SCHEMAS
# =====================================================================
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        description="Câu hỏi của học sinh về kiến thức Sinh học 12",
        example="Cơ chế nhân đôi DNA tại chạc sao chép chữ Y diễn ra như thế nào?",
    )
    top_n: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Số lượng đoạn tài liệu SGK tối ưu được trích xuất làm ngữ cảnh",
    )


class ImageItem(BaseModel):
    image_path: str = Field(..., description="Đường dẫn file ảnh cục bộ")
    url: str = Field(..., description="Đường dẫn URL web để hiển thị ảnh trên frontend")
    caption: str = Field(..., description="Mô tả nội dung học thuật của hình ảnh/sơ đồ")


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Câu trả lời chi tiết chuẩn SGK từ Gemini")
    sources: list[int] = Field(..., description="Danh sách các trang SGK được trích dẫn")
    images: list[ImageItem] = Field(..., description="Danh sách hình ảnh và sơ đồ minh họa")


# =====================================================================
# KHỞI TẠO PIPELINE (SINGLETON)
# =====================================================================
pipeline: BioChatPipeline | None = None


@app.on_event("startup")
def startup_event():
    """Khởi động và nạp các mô hình vào bộ nhớ khi server bắt đầu chạy."""
    global pipeline
    print("\n[Server Startup] Đang nạp BioChat RAG Pipeline...")
    pipeline = BioChatPipeline()
    print("[Server Startup] BioChat Pipeline đã sẵn sàng nhận request!\n")


# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.get("/", tags=["Thông tin chung"])
def root_ui():
    """Giao diện người dùng BioChat Web."""
    index_file = os.path.join(BASE_DIR, "static", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {
        "project": "BioChat - Trợ lý Sinh học 12",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "description": "API phục vụ hỏi đáp Sinh học 12 bám sát SGK bằng công nghệ Hybrid RAG.",
    }


@app.get("/api/info", tags=["Thông tin chung"])
def api_info():
    """Endpoint giới thiệu và thông tin hệ thống."""
    return {
        "project": "BioChat - Trợ lý Sinh học 12",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
        "description": "API phục vụ hỏi đáp Sinh học 12 bám sát SGK bằng công nghệ Hybrid RAG.",
    }


@app.get("/health", tags=["Thông tin chung"])
def health_check():
    """Kiểm tra tình trạng sức khỏe của server."""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Hỏi đáp RAG"])
def chat_endpoint(request_data: ChatRequest, request: Request):
    """
    Endpoint chính phục vụ hỏi đáp:
    1. Nhận câu hỏi từ học sinh.
    2. Chạy Hybrid Retrieval (BM25 + ChromaDB).
    3. Gemini sinh câu trả lời chuẩn SGK.
    4. Trả về kết quả kèm danh sách ảnh và link URL trực tiếp.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline chưa được khởi tạo hoàn tất.")

    try:
        result = pipeline.ask(query=request_data.query, top_n=request_data.top_n)

        # Xây dựng Base URL để tạo link ảnh đầy đủ (ví dụ: http://127.0.0.1:8000/images/...)
        base_url = str(request.base_url).rstrip("/")

        formatted_images = []
        for img in result.get("images", []):
            raw_path = img.get("image_path", "")
            # Lấy tên file ảnh (ví dụ: trang_3_hinh_1.png)
            filename = os.path.basename(raw_path)
            image_url = f"{base_url}/images/{filename}"

            formatted_images.append(
                ImageItem(
                    image_path=raw_path,
                    url=image_url,
                    caption=img.get("caption", ""),
                )
            )

        return ChatResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            images=formatted_images,
        )

    except Exception as e:
        print(f"[Server Error] Lỗi khi xử lý chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Đã xảy ra lỗi nội bộ trong quá trình xử lý: {str(e)}",
        )


# =====================================================================
# THỰC THI TRỰC TIẾP
# =====================================================================
if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🚀 KHỞI ĐỘNG FASTAPI SERVER TRÊN http://127.0.0.1:8000")
    print("📖 TÀI LIỆU SWAGGER UI TẠI  : http://127.0.0.1:8000/docs")
    print("=" * 65 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
