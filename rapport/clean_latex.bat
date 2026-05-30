@echo off
echo Nettoyage des fichiers temporaires LaTeX...

del /s /f /q *.aux *.log *.out *.toc *.nav *.snm *.synctex.gz *.bbl *.blg *.lof *.lot *.idx *.ind *.ilg *.bcf *.run.xml *.fdb_latexmk *.fls

echo Nettoyage termine.
pause