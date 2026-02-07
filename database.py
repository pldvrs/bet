import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client, Client

# Charger .env depuis le dossier du projet (override=True pour forcer relecture)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

url: str = (os.environ.get("SUPABASE_URL") or "").strip()
key: str = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()

# Singleton pour éviter de recréer la connexion à chaque fois
_supabase_client = None


def get_client() -> Client:
    global _supabase_client
    if not url or not key:
        print("⚠️  ATTENTION : Clés Supabase manquantes dans le .env")
        return None
    if _supabase_client is None:
        _supabase_client = create_client(url, key)
    return _supabase_client
