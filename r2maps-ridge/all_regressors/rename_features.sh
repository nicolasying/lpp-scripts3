#! /bin/bash

set +x
#for b in 1 2 3 4 5 6 7 8 9;
#do
#	cd Block${b}
	for f in *.csv ; do g=lstm_${f/${1}_TimeFeat_}; cp $f ${g%_reg.csv}.csv; done
	for f in lstm_?.csv; do mv $f lstm_000${f/lstm_}; done
	for f in lstm_??.csv; do mv $f lstm_00${f/lstm_}; done
	for f in lstm_???.csv; do mv $f lstm_0${f/lstm_}; done
#	cd ..
#done
