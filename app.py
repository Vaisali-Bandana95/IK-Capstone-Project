from src.config import settings
from src.ui.chat import create_demo


if __name__ == "__main__":
    create_demo().launch(
        server_name="0.0.0.0",
        server_port=settings.port,
    )
