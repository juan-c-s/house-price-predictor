SELECT col7 AS pais,
       SUM(CAST(col3 AS DOUBLE) * CAST(col5 AS DOUBLE)) AS ingreso_total
FROM proyecto2db."22"
GROUP BY col7
ORDER BY ingreso_total DESC;
