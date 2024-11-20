Bước 1: 
    - download python về máy, dùng phiên bản python 3.12
    - cài đặt môi trường cho python để chạy trên máy tính(gắn đường dẫn path).
Bước 2:
    - download vscode.
    - vào extentions để cài môi trường chạy cương trình.
        + cài html, css.
        + cài javascript
        + cài python.
    - vào terminal chạy lệnh "pip install -r requirements.txt" để cài đặt các thư viện để chạy chương trình.
Bước 3: 
    - sau khi cài đặt hết thư viện về vscode thì dùng terminal chạy câu lệnh "uvicorn main:pp" để chạy chương trình.
    - sau khi chạy lệnh "uvicorn main:pp" chương trình sẽ đưa ra đường dấn  "http://127.0.0.1:8000".
    - copy đường dẫn đó vào trình duyệt để mở giao diện chương trình.

Các thư mục & tệp tin :
- folder data/video: các video giao thông để chạy chương trình
- folder static: frame web
- folder model: tập kết quả training
	+ best.pt: nhận diện phương tiện
	+ best_helmet_end.pt: nhận diện mũ bảo hiểm
- folder template: các tabpage của giao diện, trọng tâm:
	+ camera.html: xử lý tabpage liên quan về xử lý hình ảnh, bắt lỗi vi phạm
	+ tables.html: bảng dữ liệu vi phạm
- folder violation: lưu trữ hình ảnh vi phạm
- file statics.py : lưu trữ thông tin về tổng số lượng vi phạm
- file violation_daily.py: lưu trỹ thông tin vi phạm theo ngày
- file requirements.txt: các thư viện cần cài đặt

nguồn dữ liệu lấy tại:
- tải ứng dụng iHaNoi về điện thoại 
- đăng ký tài khoản, có thể chọn quận Hà Đông, phường Lam Khê
- đăng nhập
- Tiện ích đô thị thông minh -> Giao thông -> Camera giao thông
