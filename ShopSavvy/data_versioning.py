import pandas as pd
import json
import os
from datetime import datetime
import hashlib

class DataVersionManager:
    """
    Manages dataset versions and tracks scraper runs with metadata logging.
    """
    
    def __init__(self, version_file='data_versions.json'):
        self.version_file = version_file
        self.versions = self._load_versions()
    
    def _load_versions(self):
        """Load version history from JSON file"""
        if os.path.exists(self.version_file):
            with open(self.version_file, 'r') as f:
                return json.load(f)
        return {'datasets': [], 'scraper_runs': []}
    
    def _save_versions(self):
        """Save version history to JSON file"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def _calculate_hash(self, filepath):
        """Calculate MD5 hash of a file"""
        hash_md5 = hashlib.md5()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return None
    
    def register_dataset(self, filepath, description='', source='generated'):
        """
        Register a new dataset version.
        
        Args:
            filepath: Path to the dataset file
            description: Description of the dataset
            source: Source of data (generated, scraped, manual)
        
        Returns:
            Version ID
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        file_hash = self._calculate_hash(filepath)
        
        version_id = len(self.versions['datasets']) + 1
        
        version_info = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'filepath': filepath,
            'description': description,
            'source': source,
            'num_records': len(df),
            'num_features': len(df.columns),
            'columns': list(df.columns),
            'file_hash': file_hash,
            'file_size_kb': os.path.getsize(filepath) / 1024
        }
        
        self.versions['datasets'].append(version_info)
        self._save_versions()
        
        return version_id
    
    def register_scraper_run(self, scraper_name, url, records_scraped, success=True, error_msg=''):
        """
        Log a web scraper execution.
        
        Args:
            scraper_name: Name of the scraper function
            url: URL that was scraped
            records_scraped: Number of records obtained
            success: Whether scraping was successful
            error_msg: Error message if failed
        
        Returns:
            Run ID
        """
        run_id = len(self.versions['scraper_runs']) + 1
        
        run_info = {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'scraper_name': scraper_name,
            'url': url,
            'records_scraped': records_scraped,
            'success': success,
            'error_message': error_msg
        }
        
        self.versions['scraper_runs'].append(run_info)
        self._save_versions()
        
        return run_id
    
    def get_dataset_history(self):
        """Get all dataset versions as DataFrame"""
        if not self.versions['datasets']:
            return pd.DataFrame()
        return pd.DataFrame(self.versions['datasets'])
    
    def get_scraper_history(self):
        """Get all scraper runs as DataFrame"""
        if not self.versions['scraper_runs']:
            return pd.DataFrame()
        return pd.DataFrame(self.versions['scraper_runs'])
    
    def get_latest_dataset(self):
        """Get information about the most recent dataset"""
        if not self.versions['datasets']:
            return None
        return self.versions['datasets'][-1]
    
    def compare_datasets(self, version_id_1, version_id_2):
        """
        Compare two dataset versions.
        
        Args:
            version_id_1: First version ID
            version_id_2: Second version ID
        
        Returns:
            Dictionary with comparison results
        """
        datasets = self.versions['datasets']
        
        v1 = next((d for d in datasets if d['version_id'] == version_id_1), None)
        v2 = next((d for d in datasets if d['version_id'] == version_id_2), None)
        
        if not v1 or not v2:
            return None
        
        comparison = {
            'version_1': version_id_1,
            'version_2': version_id_2,
            'record_difference': v2['num_records'] - v1['num_records'],
            'feature_difference': v2['num_features'] - v1['num_features'],
            'columns_added': list(set(v2['columns']) - set(v1['columns'])),
            'columns_removed': list(set(v1['columns']) - set(v2['columns'])),
            'file_size_change_kb': v2['file_size_kb'] - v1['file_size_kb'],
            'identical_hash': v1['file_hash'] == v2['file_hash']
        }
        
        return comparison
    
    def export_report(self, output_file='version_report.txt'):
        """Export version history report to text file"""
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("DATA VERSION CONTROL REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total Datasets: {len(self.versions['datasets'])}\n")
            f.write(f"Total Scraper Runs: {len(self.versions['scraper_runs'])}\n\n")
            
            f.write("DATASET HISTORY:\n")
            f.write("-"*60 + "\n")
            for ds in self.versions['datasets']:
                f.write(f"\nVersion {ds['version_id']}:\n")
                f.write(f"  Timestamp: {ds['timestamp']}\n")
                f.write(f"  File: {ds['filepath']}\n")
                f.write(f"  Records: {ds['num_records']}\n")
                f.write(f"  Features: {ds['num_features']}\n")
                f.write(f"  Source: {ds['source']}\n")
                if ds['description']:
                    f.write(f"  Description: {ds['description']}\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("SCRAPER RUN HISTORY:\n")
            f.write("-"*60 + "\n")
            for run in self.versions['scraper_runs']:
                f.write(f"\nRun {run['run_id']}:\n")
                f.write(f"  Timestamp: {run['timestamp']}\n")
                f.write(f"  Scraper: {run['scraper_name']}\n")
                f.write(f"  URL: {run['url']}\n")
                f.write(f"  Records: {run['records_scraped']}\n")
                f.write(f"  Status: {'Success' if run['success'] else 'Failed'}\n")
                if run['error_message']:
                    f.write(f"  Error: {run['error_message']}\n")
        
        return output_file


if __name__ == "__main__":
    manager = DataVersionManager()
    
    if os.path.exists('historical_fifa_data.csv'):
        version_id = manager.register_dataset(
            'historical_fifa_data.csv',
            description='Historical FIFA World Cup data (2006-2022)',
            source='generated'
        )
        print(f"Registered dataset version: {version_id}")
    
    manager.register_scraper_run(
        scraper_name='scrape_fifa_rankings',
        url='https://en.wikipedia.org/wiki/FIFA_World_Rankings',
        records_scraped=20,
        success=True
    )
    
    print("\nDataset History:")
    print(manager.get_dataset_history())
    
    print("\nScraper History:")
    print(manager.get_scraper_history())
    
    report_file = manager.export_report()
    print(f"\nReport exported to: {report_file}")
