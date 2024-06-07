from fusion import Fusion

client = Fusion("config/client_credentials_bearer.json")
client.list_catalogs()
client.list_datasets()

fs = client.get_fusion_filesystem()
print(fs.ls('common'))