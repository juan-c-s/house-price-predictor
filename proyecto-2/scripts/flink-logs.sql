-- Step 1: Create source table mapping to your existing Kinesis Data Stream
CREATE TABLE orders_stream (
    InvoiceNo VARCHAR(32),
    StockCode VARCHAR(32),
    Description VARCHAR(255),
    Quantity INT,
    InvoiceDate TIMESTAMP,
    UnitPrice DECIMAL(10,2),
    CustomerID VARCHAR(32),
    Country VARCHAR(32),
    -- Add watermark to handle late events
    WATERMARK FOR InvoiceDate AS InvoiceDate - INTERVAL '5' SECONDS
) WITH (
    'connector' = 'kinesis',
    'stream' = 'acmecoOrders',
    'aws.region' = 'us-east-1',
    'scan.stream.initpos' = 'LATEST',
    'format' = 'csv'
);

-- Step 2: Create output stream for alerts
CREATE TABLE low_stock_alerts (
    StockCode VARCHAR(32),
    Description VARCHAR(255),
    CurrentQuantity INT,
    AverageQuantity DOUBLE,
    AlertTime TIMESTAMP
) WITH (
    'connector' = 'aws-lambda',
    'function.name' = 'inventory-alert-function',    -- We'll create this Lambda function next
    'aws.region' = 'us-east-1'
);

-- Step 3: Insert anomaly detection logic
INSERT INTO low_stock_alerts
SELECT 
    StockCode,
    Description,
    Quantity as CurrentQuantity,
    AVG(Quantity) OVER (
        PARTITION BY StockCode 
        ORDER BY InvoiceDate 
        RANGE BETWEEN INTERVAL '1' HOUR PRECEDING AND CURRENT ROW
    ) as AverageQuantity,
    CURRENT_TIMESTAMP as AlertTime
FROM orders_stream
-- Filter for low stock conditions
WHERE Quantity < 10;