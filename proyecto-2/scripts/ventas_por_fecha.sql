SELECT 
  DATE_PARSE(col4, '%m/%d/%Y %H:%i') AS fecha_hora,
  DATE(DATE_PARSE(col4, '%m/%d/%Y %H:%i')) AS fecha,
  SUM(CAST(col3 AS DOUBLE) * CAST(col5 AS DOUBLE)) AS ingreso_diario
FROM proyecto2db."22"
GROUP BY 
  DATE_PARSE(col4, '%m/%d/%Y %H:%i'),
  DATE(DATE_PARSE(col4, '%m/%d/%Y %H:%i'))
ORDER BY 
  DATE(DATE_PARSE(col4, '%m/%d/%Y %H:%i'));
