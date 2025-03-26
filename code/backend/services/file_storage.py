import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

class FileStorage:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def _get_file_path(self, collection: str) -> Path:
        """Get the file path for a collection"""
        return self.data_dir / f"{collection}.json"
    
    def _load_data(self, collection: str) -> Dict[str, Any]:
        """Load data from a JSON file"""
        file_path = self._get_file_path(collection)
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_data(self, collection: str, data: Dict[str, Any]) -> None:
        """Save data to a JSON file"""
        file_path = self._get_file_path(collection)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """Insert a single document into a collection"""
        data = self._load_data(collection)
        doc_id = str(len(data) + 1)
        data[doc_id] = document
        self._save_data(collection, data)
        return doc_id
    
    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document in a collection"""
        data = self._load_data(collection)
        for doc_id, doc in data.items():
            if all(doc.get(k) == v for k, v in query.items()):
                return {"_id": doc_id, **doc}
        return None
    
    def find(self, collection: str, query: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Find documents in a collection"""
        data = self._load_data(collection)
        results = []
        for doc_id, doc in data.items():
            if query is None or all(doc.get(k) == v for k, v in query.items()):
                results.append({"_id": doc_id, **doc})
        return results
    
    def update_one(self, collection: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """Update a single document in a collection"""
        data = self._load_data(collection)
        for doc_id, doc in data.items():
            if all(doc.get(k) == v for k, v in query.items()):
                data[doc_id].update(update)
                self._save_data(collection, data)
                return True
        return False
    
    def delete_one(self, collection: str, query: Dict[str, Any]) -> bool:
        """Delete a single document from a collection"""
        data = self._load_data(collection)
        for doc_id, doc in list(data.items()):
            if all(doc.get(k) == v for k, v in query.items()):
                del data[doc_id]
                self._save_data(collection, data)
                return True
        return False

# Create a singleton instance
file_storage = FileStorage() 