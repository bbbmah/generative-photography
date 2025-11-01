import difflib
import sys
import os

def compare_python_files(file1_path, file2_path, output_html='diff_report.html'):
    """
    두 파이썬 파일의 내용을 비교하여 차이점을 HTML 파일로 생성합니다.
    """
    try:
        # 파일 내용을 줄 단위로 읽어옵니다.
        with open(file1_path, 'r', encoding='utf-8') as file1:
            text1 = file1.readlines()
        with open(file2_path, 'r', encoding='utf-8') as file2:
            text2 = file2.readlines()

    except FileNotFoundError:
        print("오류: 지정된 파일 중 하나를 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # HtmlDiff 객체를 생성합니다.
    # context=True (기본값)는 차이점 주변의 문맥만 표시하고, False는 전체 파일을 표시합니다.
    diff = difflib.HtmlDiff(tabsize=4)
    
    # 비교를 수행하고 HTML 문자열을 생성합니다.
    html_output = diff.make_file(
        text1, 
        text2, 
        fromdesc=file1_path, 
        todesc=file2_path, 
        context=True
    )

    # HTML 보고서를 파일로 저장합니다.
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"\n✅ 파일 비교가 완료되었습니다.")
    print(f"👉 차이점은 '{output_html}' 파일에 HTML 보고서로 저장되었습니다.")
    print("웹 브라우저로 이 파일을 열어 결과를 확인하세요.")

# --- 스크립트 실행 ---

if __name__ == "__main__":
    # 비교할 두 파일 경로를 지정합니다.
    # 테스트를 위해 아래 경로를 실제 파일 경로로 변경해야 합니다.
    file_a = 'camera_adaptor.py'
    file_b = 'camera_adaptor_1.py'
    
    # **********************************************
    # 실제 파일 경로를 여기에 입력하세요 (예: r'/Users/user/code/script1.py')
    # **********************************************
    
    # 예시를 위해 파일이 없으면 더미 파일을 생성
    if not os.path.exists(file_a):
        with open(file_a, 'w', encoding='utf-8') as f:
            f.write("def func_a():\n    # 이것은 첫 번째 파일입니다.\n    x = 10\n    return x * 2\n")
            
    if not os.path.exists(file_b):
        with open(file_b, 'w', encoding='utf-8') as f:
            f.write("def func_a():\n    # 이것은 두 번째 파일입니다.\n    y = 20  # 변수 이름을 변경했습니다.\n    return y * 2\n# 새 함수 추가\ndef func_b():\n    pass\n")

    compare_python_files(file_a, file_b)