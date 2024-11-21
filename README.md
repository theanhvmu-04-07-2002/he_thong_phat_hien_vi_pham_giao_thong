**Hướng dẫn cài đặt chi tiết**

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

Các thư mục & tệp tin:

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

Nguồn dữ liệu lấy tại:

- tải ứng dụng iHaNoi về điện thoại 
- đăng ký tài khoản, có thể chọn quận Hà Đông, phường Lam Khê
- đăng nhập
- Tiện ích đô thị thông minh -> Giao thông -> Camera giao thông

-------------------------------------------------------------------------------------------------------------------------
**Demo phát hiện phương tiện**

![2](https://github.com/user-attachments/assets/82eb8155-592d-4e05-b017-ca38fe6efacf)


-------------------------------------------------------------------------------------------------------------------------
**Demo bắt lỗi vượt đèn đỏ**
  ( chú ý xử lý lỗi tại điều kiện: đèn đỏ được rẽ phải )

**Phát hiện phương tiện vượt đèn đỏ**

![image](https://github.com/user-attachments/assets/0808782e-1b5a-4cd0-ac55-e4758df67422)

**Lưu hình ảnh phương tiện vượt đèn đỏ**

![image](https://github.com/user-attachments/assets/d196d2fe-a714-48a4-b851-7439801250e4)


-------------------------------------------------------------------------------------------------------------------------
**Demo bắt lỗi đi sai làn**

**Phát hiện phương tiện đi sai làn**

![image](https://github.com/user-attachments/assets/f12bee8d-893a-4ebb-9862-36de0dc5e99c)

**Lưu hình ảnh phương tiện đi sai làn**

![image](https://github.com/user-attachments/assets/4041664e-403e-49b3-81bf-e3b71b77616f)

-------------------------------------------------------------------------------------------------------------------------
**Demo bắt lỗi không đội mũ bảo hiểm**

![19](https://github.com/user-attachments/assets/286d415d-da83-426c-a120-b1983cacc5b6)


