#!/bin/bash

cd $1
rm -fr $1/tmp_GOLD
rm -fr $1/tmp_SYSTEM
rm -fr $1/tmp_OUTPUT
mkdir -p $1/tmp_GOLD
mkdir -p $1/tmp_SYSTEM

python $ABS/DUC/make_rouge.py --base $1 --gold tmp_GOLD --system tmp_SYSTEM --input input.txt
perl $ABS/DUC/prepare4rouge-simple.pl tmp_SYSTEM tmp_GOLD tmp_OUTPUT

cd tmp_OUTPUT
export PERL5LIB=/data/users/sashar/summary/duc/RELEASE-1.5.5/

echo "FULL LENGTH"
perl $ROUGE/ROUGE-1.5.5.pl -m -n 2 -w 1.2 -e $ROUGE -a settings.xml


echo "LIMITED LENGTH"
perl $ROUGE/ROUGE-1.5.5.pl -m -b 75 -n 2 -w 1.2 -e $ROUGE -a settings.xml
