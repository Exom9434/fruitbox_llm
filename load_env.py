import os
from dotenv import load_dotenv

def load_environment(env_path=".env"):
    """
    Loads a .env file and sets its contents as environment variables.

    Args:
        env_path (str, optional): The path to the .env file. 
                                  Defaults to the root .env file.
    """
    if not os.path.exists(env_path):
        raise FileNotFoundError(f"{env_path} 파일을 찾을 수 없습니다.")

    load_dotenv(dotenv_path=env_path)
    print(f"[SUCCESS] .env 환경변수 로드 완료 from {env_path}")
