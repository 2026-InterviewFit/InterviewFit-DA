import requests
import logging

logger = logging.getLogger(__name__)

# Spring Boot 서버 주소
SPRING_BOOT_URL = "http://localhost:8080/api/speech/report"

def send_to_spring_server(report_data: dict) -> bool:
    try:
        response = requests.post(SPRING_BOOT_URL, json=report_data, timeout=10)
        response.raise_for_status()
        logger.info("Spring Boot 서버로 데이터 전송 성공")
        return True
    except Exception as e:
        logger.error(f"Spring Boot 통신 실패: {e}")
        return False