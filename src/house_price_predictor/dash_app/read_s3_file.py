import os
import boto3
import pandas as pd
from io import BytesIO
from loguru import logger

def read_from_s3(bucket_name, file_key, file_type='csv'):
    """
    Read data directly from S3
    Args:
        bucket_name (str): Name of the S3 bucket
        file_key (str): Path to the file in the bucket
        file_type (str): Type of file to read ('csv' or 'parquet')
    """
    try:
        logger.info(f"Reading from S3 bucket: {bucket_name}, file key: {file_key}, file type: {file_type}")
        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('aws_access_key_id'),
            aws_secret_access_key=os.getenv('aws_secret_access_key'),
            aws_session_token=os.getenv('aws_session_token')
            
        )
        
        # Get the object from S3
        obj = s3_client.get_object(
            Bucket=bucket_name,
            Key=file_key
        )
        
        # Read the data stream based on file type
        if file_type.lower() == 'csv':
            df = pd.read_csv(obj['Body'])
        elif file_type.lower() == 'parquet':
            buffer = BytesIO(obj['Body'].read())
            df = pd.read_parquet(buffer)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'parquet'")
        
        return df
    

    except Exception as e:
        logger.error(f"Error reading from S3: {str(e)}")
        return None
# Example usage:
# For CSV:
# df = read_from_s3(HOUSING_BUCKET, HOUSING_S3_KEY, file_type='csv')
# For Parquet:
# df = read_from_s3(HOUSING_BUCKET, HOUSING_S3_KEY, file_type='parquet')