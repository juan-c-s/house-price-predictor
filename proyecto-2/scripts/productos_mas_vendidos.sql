SELECT col2 AS producto, SUM(CAST(col3 AS INT)) AS cantidad_total
FROM proyecto2db."22"
GROUP BY col2
ORDER BY cantidad_total DESC
LIMIT 10;
