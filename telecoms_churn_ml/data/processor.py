"""Data processing and cleaning functionality."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Union
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles cleaning, processing, and unifying messy telecoms and billing data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            config: Configuration dictionary for processing parameters
        """
        self.config = config or self._get_default_config()
        logger.info("DataProcessor initialized")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for data processing."""
        return {
            'missing_value_threshold': 0.5,
            'outlier_method': 'iqr',
            'encoding_method': 'label',
            'scaling_method': 'standard',
            'date_format': '%Y-%m-%d',
            'customer_id_col': 'customer_id',
            'churn_col': 'churn'
        }
    
    def load_raw_data(self, data_path: Union[str, Path]) -> Dict[str, pd.DataFrame]:
        """
        Load raw data files from the specified directory.
        
        Args:
            data_path: Path to directory containing raw data files
            
        Returns:
            Dictionary of DataFrames with filename as key
        """
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data path {data_path} does not exist")
        
        dataframes = {}
        supported_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        
        for file_path in data_path.iterdir():
            if file_path.suffix.lower() in supported_formats:
                try:
                    df = self._load_file(file_path)
                    dataframes[file_path.stem] = df
                    logger.info(f"Loaded {file_path.name}: {df.shape}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
        
        if not dataframes:
            logger.warning("No data files found or loaded successfully")
        
        return dataframes
    
    def _load_file(self, file_path: Path) -> pd.DataFrame:
        """Load a single file based on its extension."""
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif suffix == '.json':
            return pd.read_json(file_path)
        elif suffix == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def clean_dataframe(self, df: pd.DataFrame, df_name: str = "") -> pd.DataFrame:
        """
        Clean a single DataFrame.
        
        Args:
            df: DataFrame to clean
            df_name: Name of the DataFrame for logging
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning DataFrame {df_name}: {df.shape}")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Clean column names
        df_clean = self._clean_column_names(df_clean)
        
        # Handle missing values
        df_clean = self._handle_missing_values(df_clean)
        
        # Clean data types
        df_clean = self._clean_data_types(df_clean)
        
        # Remove duplicates
        df_clean = self._remove_duplicates(df_clean)
        
        # Handle outliers
        df_clean = self._handle_outliers(df_clean)
        
        logger.info(f"Cleaned DataFrame {df_name}: {df_clean.shape}")
        return df_clean
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column names."""
        df.columns = (df.columns
                     .str.lower()
                     .str.replace(' ', '_')
                     .str.replace('-', '_')
                     .str.replace('[^a-z0-9_]', '', regex=True))
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Remove columns with too many missing values
        threshold = self.config['missing_value_threshold']
        missing_pct = df.isnull().sum() / len(df)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        
        if len(cols_to_drop) > 0:
            logger.info(f"Dropping columns with >{threshold*100}% missing: {list(cols_to_drop)}")
            df = df.drop(columns=cols_to_drop)
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                # Categorical: fill with mode or 'Unknown'
                mode_val = df[col].mode()
                fill_val = mode_val[0] if len(mode_val) > 0 else 'Unknown'
                df[col] = df[col].fillna(fill_val)
            else:
                # Numerical: fill with median
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data types."""
        for col in df.columns:
            # Try to convert to numeric if possible
            if df[col].dtype == 'object':
                # Check if it's a date column
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except (ValueError, TypeError):
                        pass
                else:
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df[col] = numeric_series
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        initial_shape = df.shape
        df = df.drop_duplicates()
        if df.shape[0] < initial_shape[0]:
            logger.info(f"Removed {initial_shape[0] - df.shape[0]} duplicate rows")
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers in numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.config['outlier_method'] == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def unify_data(self, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Unify multiple DataFrames into a single dataset.
        
        Args:
            dataframes: Dictionary of DataFrames to unify
            
        Returns:
            Unified DataFrame
        """
        if not dataframes:
            raise ValueError("No dataframes provided for unification")
        
        logger.info(f"Unifying {len(dataframes)} DataFrames")
        
        # Find common customer identifier
        customer_id_col = self._find_customer_id_column(dataframes)
        
        if customer_id_col is None:
            # If no common customer ID, concatenate all data
            logger.warning("No common customer ID found, concatenating all data")
            unified_df = pd.concat(dataframes.values(), ignore_index=True)
        else:
            # Merge on customer ID
            unified_df = self._merge_on_customer_id(dataframes, customer_id_col)
        
        logger.info(f"Unified DataFrame shape: {unified_df.shape}")
        return unified_df
    
    def _find_customer_id_column(self, dataframes: Dict[str, pd.DataFrame]) -> Optional[str]:
        """Find the common customer ID column across DataFrames."""
        possible_id_cols = ['customer_id', 'id', 'cust_id', 'user_id', 'account_id']
        
        for id_col in possible_id_cols:
            if all(id_col in df.columns for df in dataframes.values()):
                return id_col
        
        return None
    
    def _merge_on_customer_id(self, dataframes: Dict[str, pd.DataFrame], 
                             customer_id_col: str) -> pd.DataFrame:
        """Merge DataFrames on customer ID."""
        df_list = list(dataframes.values())
        unified_df = df_list[0]
        
        for df in df_list[1:]:
            unified_df = unified_df.merge(df, on=customer_id_col, how='outer', 
                                        suffixes=('', '_dup'))
            
            # Remove duplicate columns
            dup_cols = [col for col in unified_df.columns if col.endswith('_dup')]
            unified_df = unified_df.drop(columns=dup_cols)
        
        return unified_df
    
    def clean_and_unify_data(self, data_path: Union[str, Path]) -> pd.DataFrame:
        """
        Complete pipeline: load, clean, and unify data.
        
        Args:
            data_path: Path to directory containing raw data files
            
        Returns:
            Cleaned and unified DataFrame
        """
        logger.info("Starting data cleaning and unification pipeline")
        
        # Load raw data
        raw_dataframes = self.load_raw_data(data_path)
        
        # Clean each DataFrame
        clean_dataframes = {}
        for name, df in raw_dataframes.items():
            clean_dataframes[name] = self.clean_dataframe(df, name)
        
        # Unify data
        unified_df = self.unify_data(clean_dataframes)
        
        logger.info("Data cleaning and unification pipeline completed")
        return unified_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path],
                           format: str = 'csv') -> None:
        """
        Save processed data to file.
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            format: Output format ('csv', 'parquet', 'excel')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'parquet':
            df.to_parquet(output_path, index=False)
        elif format == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved processed data to {output_path}")
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get comprehensive summary of the DataFrame.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {}
        }
        
        # Add categorical summary
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        return summary
    
    def unify_multi_table_data(self, data_path: Union[str, Path], output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Unify multiple telecoms data tables into a single dataset for ML training.
        
        Args:
            data_path: Path to directory containing separate CSV files
            output_path: Optional path to save the unified dataset
            
        Returns:
            Unified DataFrame with all customer data
        """
        data_path = Path(data_path)
        logger.info(f"Unifying multi-table data from {data_path}")
        
        # Define expected tables and their join keys
        tables_config = {
            'customers': {'key': 'customer_id', 'required': True},
            'billing': {'key': 'customer_id', 'required': True},
            'usage': {'key': 'customer_id', 'required': False},
            'customer_care': {'key': 'customer_id', 'required': False},
            'crm': {'key': 'customer_id', 'required': False},
            'social': {'key': 'customer_id', 'required': False},
            'network': {'key': 'customer_id', 'required': False}
        }
        
        loaded_tables = {}
        
        # Load available tables
        for table_name, config in tables_config.items():
            file_path = data_path / f"{table_name}.csv"
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    loaded_tables[table_name] = df
                    logger.info(f"Loaded {table_name}: {df.shape}")
                except Exception as e:
                    logger.warning(f"Failed to load {table_name}: {e}")
                    if config['required']:
                        raise
            elif config['required']:
                raise FileNotFoundError(f"Required table {table_name} not found at {file_path}")
        
        # Start with customers table as base
        if 'customers' not in loaded_tables:
            raise ValueError("Customers table is required but not found")
        
        unified_df = loaded_tables['customers'].copy()
        logger.info(f"Starting with customers table: {unified_df.shape}")
        
        # Join other tables
        for table_name, df in loaded_tables.items():
            if table_name == 'customers':
                continue
                
            join_key = tables_config[table_name]['key']
            
            # Ensure join key exists in both tables
            if join_key not in unified_df.columns:
                logger.warning(f"Join key {join_key} not found in unified dataset, skipping {table_name}")
                continue
            if join_key not in df.columns:
                logger.warning(f"Join key {join_key} not found in {table_name}, skipping")
                continue
            
            # Perform left join to preserve all customers
            before_shape = unified_df.shape
            unified_df = unified_df.merge(df, on=join_key, how='left', suffixes=('', f'_{table_name}'))
            after_shape = unified_df.shape
            
            logger.info(f"Joined {table_name}: {before_shape} -> {after_shape}")
        
        logger.info(f"Final unified dataset shape: {unified_df.shape}")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            unified_df.to_csv(output_path, index=False)
            logger.info(f"Saved unified dataset to {output_path}")
        
        return unified_df
    
    def unify_generated_data(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Unify all generated data tables into a single dataset for ML training.
        
        Args:
            data_dir: Directory containing the generated data files
            
        Returns:
            Unified DataFrame ready for ML training
        """
        data_dir = Path(data_dir)
        logger.info(f"Unifying data from {data_dir}")
        
        # Load all data tables
        tables = {}
        expected_files = [
            'customers.csv', 'billing.csv', 'usage.csv', 
            'customer_care.csv', 'crm.csv', 'social.csv', 'network.csv'
        ]
        
        for filename in expected_files:
            file_path = data_dir / filename
            if file_path.exists():
                tables[filename.replace('.csv', '')] = pd.read_csv(file_path)
                logger.info(f"Loaded {filename} with shape {tables[filename.replace('.csv', '')].shape}")
            else:
                logger.warning(f"File {filename} not found in {data_dir}")
        
        if 'customers' not in tables:
            raise ValueError("customers.csv is required as the base table")
        
        # Start with customers as the base table
        unified_df = tables['customers'].copy()
        logger.info(f"Starting with customers table: {unified_df.shape}")
        
        # Join other tables on customer_id
        for table_name, df in tables.items():
            if table_name == 'customers':
                continue
            
            if 'customer_id' not in df.columns:
                logger.warning(f"customer_id not found in {table_name}, skipping")
                continue
            
            # Create prefixed column names to avoid conflicts
            df_prefixed = df.copy()
            non_id_cols = [col for col in df.columns if col != 'customer_id']
            prefix_mapping = {col: f"{table_name}_{col}" for col in non_id_cols}
            df_prefixed = df_prefixed.rename(columns=prefix_mapping)
            
            # Merge with the unified dataset
            before_shape = unified_df.shape
            unified_df = unified_df.merge(
                df_prefixed, 
                on='customer_id', 
                how='left'
            )
            logger.info(f"Merged {table_name}: {before_shape} -> {unified_df.shape}")
        
        logger.info(f"Final unified dataset shape: {unified_df.shape}")
        logger.info(f"Columns: {list(unified_df.columns)}")
        
        return unified_df
