SELECT col2 AS producto,
       SUM(CAST(col3 AS DOUBLE) * CAST(col5 AS DOUBLE)) AS ingreso_total
FROM proyecto2db."22"
GROUP BY col2
ORDER BY ingreso_total DESC;
