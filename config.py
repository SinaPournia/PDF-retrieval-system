import dotenv
from pydantic import Field
from pydantic_settings import SettingsConfigDict ,BaseSettings

# Load environment variables from .env file
dotenv.load_dotenv()

class Settings(BaseSettings):
    vespa_app_name: str = Field(default="default_app_ame", env="VESPA_APP_NAME")  # Vespa app name
    ranking_profile_name: str = Field(default="default", env="RANKING_PROFILE_NAME")  # Ranking profile name
    device_model: str = Field(default="cpu", env="_MODEL_DEVICE")  # Device to run model on, defaulting to CPU
    batch_size: int = Field(default=1, env="BATCH_SIZE")  # Default batch size for DataLoader
    image_resize: int = Field(default=640, env="IMAGE_RESIZE")  # Default image resize dimension
    vespa_url: str = Field(default="http://localhost", env="VESPA_URL")  # Vespa URL
    vespa_port: int = Field(default=8080, env="VESPA_PORT")  # Vespa port
    colpali_model_name: str = Field(default="impactframes/colqwen2-v0.1", env="_MODEL_NAME")  # Model name
    
    device_map: str = Field("cuda:0", env="DEVICE_MAP")  # Device mapping for model
   
    batch_size: int = Field(default=1, env="BATCH_SIZE")  # Batch size for DataLoader
    image_resize: int = Field(default=800, env="IMAGE_RESIZE")
    model_config = SettingsConfigDict(
        env_prefix="MYAPP_",         # Prefix for env variables
        env_file=".env",             # Use .env file
        env_ignore_empty=True,       # Ignore empty env variables
        extra="ignore"               # Ignore undefined fields
    )





# Instantiate the settings to be used in the app
settings = Settings()
