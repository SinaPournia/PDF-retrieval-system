import dotenv
from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings

# Load environment variables from .env file
dotenv.load_dotenv()

class Settings(BaseSettings):
    vespa_app_name: str = Field(default="default_app_name")  # Vespa app name
    ranking_profile_name: str = Field(default="default")  # Ranking profile name
    image_resize: int = Field(default=640)  # Default image resize dimension
    vespa_host: str = Field(default="localhost")  # Vespa URL
    vespa_protocol: str = Field(default="http")  # Vespa URL
    vespa_port: int = Field(default=8080)  # Vespa port
    model_name: str = Field(default="impactframes/colqwen2-v0.1")  # Model name
    batch_size: int = Field(default=1)  # Batch size for DataLoader

    model_config = SettingsConfigDict(
        env_prefix="MYAPP_",         # Prefix for env variables
        env_file=".env",             # Use .env file
        env_ignore_empty=True,       # Ignore empty env variables
        extra="ignore",              # Ignore undefined fields
        protected_namespaces = ('settings_',)
    )

    @property
    def vespa_url(self) -> str:
        """Dynamically construct the Vespa URL."""
        return f"{self.vespa_protocol}://{self.vespa_host}:{self.vespa_port}/"



