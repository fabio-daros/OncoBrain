import os
from synapseclient import Synapse

def sync_from_synapse(syn: Synapse, entity_id: str, path: str):
    children = syn.getChildren(entity_id)
    for child in children:
        child_type = child['type']
        child_id = child['id']
        name = child['name']
        full_path = os.path.join(path, name)

        if child_type == 'org.sagebionetworks.repo.model.FileEntity':
            if not os.path.exists(full_path):
                print(f"Downloading file: {name}")
                syn.get(child_id, downloadLocation=path)
        elif child_type == 'org.sagebionetworks.repo.model.Folder':
            os.makedirs(full_path, exist_ok=True)
            sync_from_synapse(syn, child_id, full_path)
