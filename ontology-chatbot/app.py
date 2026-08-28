"""``streamlit run ontology-chatbot/app.py``를 위한 최소 진입점.

실제 화면 구성과 이벤트 처리는 :mod:`views.streamlit_app`에 둔다. 이 파일은
Streamlit이 실행할 안정적인 경로만 제공하여 View 구현과 실행 명령을 분리한다.
"""

from views.streamlit_app import main

if __name__ == "__main__":
    main()
