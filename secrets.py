"""Centralized secrets management using HashiCorp Vault."""

import os
import hvac
from typing import Optional

class SecretsManager:
    """A singleton class to manage secrets from HashiCorp Vault."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecretsManager, cls).__new__(cls)
            cls._instance.client = None
            try:
                vault_addr = os.environ.get("VAULT_ADDR")
                role_id = os.environ.get("VAULT_ROLE_ID")
                secret_id = os.environ.get("VAULT_SECRET_ID")

                if vault_addr and role_id and secret_id:
                    cls._instance.client = hvac.Client(url=vault_addr)
                    cls._instance.client.auth.approle.login(
                        role_id=role_id,
                        secret_id=secret_id
                    )
                    print("Successfully authenticated with Vault.")
                else:
                    print("Vault environment variables not set. Secrets manager will be disabled.")
            except Exception as e:
                print(f"Failed to authenticate with Vault: {e}")
                cls._instance.client = None
        return cls._instance

    def get_secret(self, path: str, key: str) -> Optional[str]:
        """
        Retrieve a secret from a given path and key in Vault.
        
        Example: get_secret("kv/oncotarget-lite/database", "password")
        """
        if self.client and self.client.is_authenticated():
            try:
                response = self.client.secrets.kv.v2.read_secret_version(
                    path=path,
                )
                return response['data']['data'][key]
            except Exception as e:
                print(f"Failed to retrieve secret '{key}' from path '{path}': {e}")
                return None
        return None

# Singleton instance
secrets_manager = SecretsManager()

