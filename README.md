# DoAnMLinCV
# Để tạo models máy học
  `python3 classifierDaSua.p A B C` \
  Ví dụ:\
  `python3 classifierDaSua.py ../DataSet/FaceData/processed/ ../ModelsPD/20180402-114759.pb ../Models/facePKL` \
  Trong đó: \
  | A | Là đường dẫn đến thư mục chứa dataset đã xử lý |
  | B | Là đường dẫn đến models mạng ... |
  | C | Là đường đẫn đến nơi lưu models đã học được |
  
  Khi chạy file này thì sẽ xuất hiện thư mục tempLuu. Trong thư mục chứ các file sau: \
  | A | Là đường dẫn đến thư mục chứa dataset đã xử lý |
  | B | Là đường dẫn đến models mạng ... |
  | C | Là đường đẫn đến nơi lưu models đã học được |
     
# Để đo độ chính xác của models đã tạo được ở trên
  `python3 test.py A B C` \
  Ví dụ: \   
  `python3 test.py ../Models/ ../ModelsPD/20180402-114759.pb GaussianNB` \
  Trong đó: \
  | A | Là đường dẫn đến thư mục chưa models đã học được |
  | B | Là đường dẫn đến models mạng ... |
  | C | Là tên phương pháp máy học muốn đặt tên |
  
  Khi chạy file này thì sẽ xuất hiện thư mục tempLuu. Trong thư mục chứ các file sau: \
  | A | Là đường dẫn đến thư mục chứa dataset đã xử lý |
  | B | Là đường dẫn đến models mạng ... |
  | C | Là đường đẫn đến nơi lưu models đã học được |
  
# Để chạy toàn bộ hệ thống từ ảnh ra gợi ý hoạt động
  `python3 GetHtml.py --u A --p B --l C` \
  Ví dụ: \
  `python3 GetHtml.py --u 17521234 --p 123456789 -- l ../DataSet/FaceData/Nguyen.jpg` \
  Trong đó: \
  | --u A | A là username đăng nhập đrl có (có quyền bí thư / lớp trưởng) |
  | --p B | B là password đăng nhập đrl có (có quyền bí thư / lớp trưởng |
  | --l C | C là đường dẫn đến 1 ảnh muốn thử |
  

  
