import yaml, os
from types import SimpleNamespace

# Load YAML once
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

# Decide environment dynamically
env = os.getenv("RENATE_ENV", "test")

cfg = config[env]

# Create a dot-access object
CONFIG = SimpleNamespace(
    BASE_URL=cfg["base_url"],
    USER=cfg["credentials"]["user"],
    PASSWORD=cfg["credentials"]["password"],
    COLLECTIONS_ENDPOINT=f"{cfg['base_url']}{cfg['endpoints']['collections']}",
    COLLECTION_ITEMS_ENDPOINT=f"{cfg['base_url']}{cfg['endpoints']['collection_items']}",
    UPLOAD_BITSTREAMS_ENDPOINT=f"{cfg['base_url']}{cfg['endpoints']['upload_bitstreams']}",
    ITEMS_ENDPOINT=f"{cfg['base_url']}{cfg['endpoints']['items']}",
    UPDATE_ITEMS_METADATA=f"{cfg['base_url']}{cfg['endpoints']['update_item_metadata']}",
    LOGIN_ENDPOINT=f"{cfg['base_url']}{cfg['endpoints']['login']}",
)