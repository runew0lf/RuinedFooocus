#!/bin/sh
# Bash script to build a small python_embed for Windows.
URL=https://www.python.org/ftp/python/3.14.4/python-3.14.4-embed-amd64.zip
RESULT=python_embeded
DL=dl
[ -d $RESULT ] && echo "$RESULT already exists" && exit 1
[ -d $DL ] || mkdir $DL
mkdir $RESULT
curl -L -o $DL/embed.zip $URL
unzip -x $DL/embed.zip -d $RESULT
rename 's/_pth/pth/' $RESULT/python*._pth
echo "import site" >> $RESULT/python*.pth
curl -L -o $RESULT/get-pip.py https://bootstrap.pypa.io/get-pip.py
(proton-call -r $RESULT/python.exe $RESULT/get-pip.py)
(proton-call -r $RESULT/Scripts/pip.exe install wheel packaging pygit2 setuptools==80.9.0 cffi==2.0.0)
