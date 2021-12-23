# kai2021

# 컴파일 환경
 * Microsoft Visual Studio Professional 2019

# 필수 라이브러리 (2021-07-21 수정)
 * CUDA Toolkit 11.3 Update 1 (설치 필수) : https://developer.nvidia.com/cuda-toolkit-archive
 * ~~boost 1.76.0 : https://www.boost.org/users/history/~~
 * ~~eigen 3.3.9 : http://eigen.tuxfamily.org/index.php?title=Main_Page~~
 * OpenCV 4.5.2 : https://sourceforge.net/projects/opencvlibrary/files/4.5.2/opencv-4.5.2-vc14_vc15.exe/download
 * MySQL Connector C++ 8.0.2 : https://dev.mysql.com/downloads/installer/

# Visual Studio 연동 시 주의 사항
 * ~~VS 상에서 GitLab clone 수행 시, CMakeLists.txt 파일이 포함된 모든 프로젝트 폴더에 캐시를 생성하므로
   clone 전에 반드시 아래와 같이 설정해야 함~~
 * ~~Tools > Options > CMake > General > When cache is out of date 설정을
   "Never run configure step automatically"로 변경
