"""Data validation functionality."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from loguru import logger
from pydantic import BaseModel, validator
import warnings

warnings.filterwarnings('ignore')


class DataSchema(BaseModel):
    """Pydantic model for data validation schema."""
    
    customer_id: str
    churn: int
    
    class Config:
        extra = "allow"  # Allow additional fields
    
    @validator('churn')
    def validate_churn(cls, v):
        """Validate churn is binary (0 or 1)."""
        if v not in [0, 1]:
            raise ValueError('Churn must be 0 or 1')
        return v


class DataValidator:
    """
    Validates data quality and schema compliance for telecoms churn data.
    """
    
    def __init__(self, schema: Optional[Dict] = None):
        """
        Initialize the DataValidator.
        
        Args:
            schema: Expected data schema dictionary
        """
        self.schema = schema or self._get_default_schema()
        self.validation_results = {}
        logger.info("DataValidator initialized")
    
    def _get_default_schema(self) -> Dict:
        """Get default schema for telecoms churn data."""
        return {
            'required_columns': [
                'customer_id',
                'churn'
            ],
            'optional_columns': [
                'tenure',
                'monthly_charges',
                'total_charges',
                'contract_type',
                'payment_method',
                'internet_service',
                'phone_service',
                'multiple_lines',
                'online_security',
                'online_backup',
                'device_protection',
                'tech_support',
                'streaming_tv',
                'streaming_movies',
                'paperless_billing',
                'gender',
                'senior_citizen',
                'partner',
                'dependents'
            ],
            'data_types': {
                'customer_id': 'object',
                'churn': 'int64',
                'tenure': 'int64',
                'monthly_charges': 'float64',
                'total_charges': 'float64',
                'senior_citizen': 'int64'
            },
            'value_ranges': {
                'tenure': (0, 100),
                'monthly_charges': (0, 1000),
                'total_charges': (0, 10000),
                'churn': (0, 1),
                'senior_citizen': (0, 1)
            },
            'categorical_values': {
                'contract_type': ['Month-to-month', 'One year', 'Two year'],
                'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
                'internet_service': ['DSL', 'Fiber optic', 'No'],
                'gender': ['Male', 'Female']
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame, strict: bool = False) -> Dict[str, Any]:
        """
        Validate a DataFrame against the schema.
        
        Args:
            df: DataFrame to validate
            strict: Whether to enforce strict validation
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating DataFrame with shape {df.shape}")
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # Check basic structure
        structure_results = self._validate_structure(df, strict)
        results['errors'].extend(structure_results['errors'])
        results['warnings'].extend(structure_results['warnings'])
        
        # Check data types
        dtype_results = self._validate_data_types(df, strict)
        results['errors'].extend(dtype_results['errors'])
        results['warnings'].extend(dtype_results['warnings'])
        
        # Check value ranges
        range_results = self._validate_value_ranges(df)
        results['errors'].extend(range_results['errors'])
        results['warnings'].extend(range_results['warnings'])
        
        # Check categorical values
        categorical_results = self._validate_categorical_values(df)
        results['errors'].extend(categorical_results['errors'])
        results['warnings'].extend(categorical_results['warnings'])
        
        # Check data quality
        quality_results = self._validate_data_quality(df)
        results['warnings'].extend(quality_results['warnings'])
        
        # Set overall validity
        results['is_valid'] = len(results['errors']) == 0
        
        # Generate summary
        results['summary'] = self._generate_validation_summary(df, results)
        
        self.validation_results = results
        logger.info(f"Validation completed. Valid: {results['is_valid']}")
        
        return results
    
    def _validate_structure(self, df: pd.DataFrame, strict: bool) -> Dict[str, List[str]]:
        """Validate DataFrame structure."""
        errors = []
        warnings = []
        
        # Check required columns
        missing_required = set(self.schema['required_columns']) - set(df.columns)
        if missing_required:
            errors.append(f"Missing required columns: {list(missing_required)}")
        
        # Check for unexpected columns if strict mode
        if strict:
            expected_cols = set(self.schema['required_columns'] + self.schema['optional_columns'])
            unexpected_cols = set(df.columns) - expected_cols
            if unexpected_cols:
                warnings.append(f"Unexpected columns found: {list(unexpected_cols)}")
        
        # Check for empty DataFrame
        if df.empty:
            errors.append("DataFrame is empty")
        
        # Check for duplicate columns
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            errors.append(f"Duplicate columns found: {duplicates}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_data_types(self, df: pd.DataFrame, strict: bool) -> Dict[str, List[str]]:
        """Validate data types."""
        errors = []
        warnings = []
        
        for col, expected_dtype in self.schema['data_types'].items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if strict and actual_dtype != expected_dtype:
                    errors.append(f"Column '{col}' has type {actual_dtype}, expected {expected_dtype}")
                elif not strict and not self._is_compatible_dtype(actual_dtype, expected_dtype):
                    warnings.append(f"Column '{col}' has type {actual_dtype}, expected {expected_dtype}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """Check if data types are compatible."""
        numeric_types = ['int64', 'float64', 'int32', 'float32']
        
        if expected in numeric_types and actual in numeric_types:
            return True
        if expected == 'object' and actual in ['object', 'string', 'category']:
            return True
        
        return actual == expected
    
    def _validate_value_ranges(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate value ranges for numeric columns."""
        errors = []
        warnings = []
        
        for col, (min_val, max_val) in self.schema['value_ranges'].items():
            if col in df.columns:
                out_of_range = df[(df[col] < min_val) | (df[col] > max_val)]
                if len(out_of_range) > 0:
                    pct_out_of_range = len(out_of_range) / len(df) * 100
                    if pct_out_of_range > 5:  # More than 5% out of range
                        errors.append(f"Column '{col}' has {pct_out_of_range:.1f}% values outside range [{min_val}, {max_val}]")
                    else:
                        warnings.append(f"Column '{col}' has {len(out_of_range)} values outside range [{min_val}, {max_val}]")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_categorical_values(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate categorical values."""
        errors = []
        warnings = []
        
        for col, expected_values in self.schema['categorical_values'].items():
            if col in df.columns:
                unique_values = set(df[col].dropna().unique())
                unexpected_values = unique_values - set(expected_values)
                
                if unexpected_values:
                    warnings.append(f"Column '{col}' has unexpected values: {list(unexpected_values)}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate general data quality."""
        warnings = []
        
        # Check missing values
        missing_pct = df.isnull().sum() / len(df) * 100
        high_missing = missing_pct[missing_pct > 20]
        if len(high_missing) > 0:
            warnings.append(f"Columns with >20% missing values: {high_missing.to_dict()}")
        
        # Check duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            warnings.append(f"Found {dup_count} duplicate rows ({dup_count/len(df)*100:.1f}%)")
        
        # Check constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Constant columns found: {constant_cols}")
        
        # Check high cardinality columns
        high_card_cols = []
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.8:  # More than 80% unique values
                high_card_cols.append(col)
        if high_card_cols:
            warnings.append(f"High cardinality columns: {high_card_cols}")
        
        return {'warnings': warnings}
    
    def _generate_validation_summary(self, df: pd.DataFrame, results: Dict) -> Dict:
        """Generate validation summary."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'error_count': len(results['errors']),
            'warning_count': len(results['warnings']),
            'data_types': df.dtypes.value_counts().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a detailed validation report.
        
        Args:
            output_path: Optional path to save the report
            
        Returns:
            Validation report as string
        """
        if not self.validation_results:
            return "No validation results available. Run validate_dataframe() first."
        
        results = self.validation_results
        
        report = []
        report.append("# Data Validation Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary
        report.append("## Summary")
        report.append(f"Overall Status: {'✅ VALID' if results['is_valid'] else '❌ INVALID'}")
        report.append(f"Total Rows: {results['summary']['total_rows']:,}")
        report.append(f"Total Columns: {results['summary']['total_columns']}")
        report.append(f"Errors: {results['summary']['error_count']}")
        report.append(f"Warnings: {results['summary']['warning_count']}")
        report.append(f"Memory Usage: {results['summary']['memory_usage_mb']:.2f} MB")
        report.append("")
        
        # Errors
        if results['errors']:
            report.append("## Errors")
            for i, error in enumerate(results['errors'], 1):
                report.append(f"{i}. {error}")
            report.append("")
        
        # Warnings
        if results['warnings']:
            report.append("## Warnings")
            for i, warning in enumerate(results['warnings'], 1):
                report.append(f"{i}. {warning}")
            report.append("")
        
        # Data Types
        report.append("## Data Types Distribution")
        for dtype, count in results['summary']['data_types'].items():
            report.append(f"- {dtype}: {count} columns")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text
