for (( ; ; ))
do
   twint --username realDonaldTrump --limit 20 --csv -o prova.csv
   sleep 1
   python pyScript.py 
   sleep 1
   rm prova.csv
   sleep 1
done
